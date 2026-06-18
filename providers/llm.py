"""
providers/llm.py
────────────────
LLM provider with:
  - Gemini primary (via LangChain + direct streaming)
  - Cloudflare AI fallback
  - Token-by-token streaming via llm_stream()
  - Rate-limit tracking with automatic recovery
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

import httpx
import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config.settings import settings

log = structlog.get_logger()

# ── Rate-limit tracker ────────────────────────────────────────────────────────
_rate_limited_until: dict[str, float] = {}


def _is_rate_limited(provider: str) -> bool:
    return time.time() < _rate_limited_until.get(provider, 0)


def _mark_rate_limited(provider: str, ttl_seconds: int = 60) -> None:
    _rate_limited_until[provider] = time.time() + ttl_seconds
    log.warning("provider_rate_limited", provider=provider, ttl_seconds=ttl_seconds)


# ── Gemini ────────────────────────────────────────────────────────────────────

def coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


class SafeChatGoogleGenerativeAI:
    """Thin wrapper around LangChain's ChatGoogleGenerativeAI with content sanitisation."""

    def __init__(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=api_key,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            convert_system_message_to_human=True,
            max_retries=0,
        )

    def _clean(self, msg: BaseMessage) -> BaseMessage:
        if hasattr(msg, "content"):
            clean_str = coerce_content(msg.content)
            try:
                msg.content = clean_str
            except Exception:
                pass
            if isinstance(msg.content, list):
                if hasattr(msg, "model_copy"):
                    msg = msg.model_copy(update={"content": clean_str})
                else:
                    try:
                        cls = msg.__class__
                        kwargs = {}
                        if hasattr(cls, "model_fields"):
                            for field_name in cls.model_fields:
                                if field_name != "content" and hasattr(msg, field_name):
                                    kwargs[field_name] = getattr(msg, field_name)
                        else:
                            kwargs = {k: v for k, v in msg.__dict__.items() if k != "content"}
                        msg = cls(content=clean_str, **kwargs)
                    except Exception:
                        import copy
                        msg = copy.copy(msg)
                        msg.content = clean_str
        return msg

    async def ainvoke(self, msgs, **kwargs) -> BaseMessage:
        resp = await self._llm.ainvoke(msgs, **kwargs)
        return self._clean(resp)

    async def astream(self, msgs, **kwargs) -> AsyncIterator[str]:
        """Yield text tokens one by one via LangChain streaming."""
        async for chunk in self._llm.astream(msgs, **kwargs):
            if chunk.content:
                content = chunk.content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, str):
                            yield part
                        elif isinstance(part, dict) and "text" in part:
                            yield part["text"]
                elif isinstance(content, str):
                    yield content


@lru_cache(maxsize=1)
def get_gemini_llm() -> SafeChatGoogleGenerativeAI:
    return SafeChatGoogleGenerativeAI()


# ── Cloudflare AI ─────────────────────────────────────────────────────────────

async def cloudflare_complete(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Call Cloudflare Workers AI REST API as LLM fallback."""
    if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
        raise RuntimeError("Cloudflare credentials not configured")

    if settings.cloudflare_gateway_id:
        url = (
            f"https://gateway.ai.cloudflare.com/v1/"
            f"{settings.cloudflare_account_id}/{settings.cloudflare_gateway_id}"
            f"/workers-ai/{settings.cloudflare_model}"
        )
    else:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{settings.cloudflare_account_id}/ai/run/{settings.cloudflare_model}"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.cloudflare_api_token}",
                "Content-Type": "application/json",
            },
            json={"messages": messages, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle API error response format
        if not data.get("success", True) or "result" not in data:
            err_msg = data.get("errors") or data
            raise RuntimeError(f"Cloudflare AI API error: {err_msg}")

        result = data.get("result")
        if not result:
            raise RuntimeError(f"Cloudflare AI API returned empty result: {data}")

        if isinstance(result, dict):
            if "response" in result:
                return result["response"]
            elif "choices" in result and isinstance(result["choices"], list) and result["choices"]:
                choice = result["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content")
                if content:
                    return content.strip()
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    return reasoning.strip()
            elif "text" in result:
                return result["text"]
        return str(result)


# ── Unified blocking invoke ───────────────────────────────────────────────────

async def llm_invoke(
    system: str,
    user: str,
    callback=None,
    max_tokens: int = 1024,
) -> str:
    """
    Call LLM with provider fallback chain: Gemini → Cloudflare.
    Returns plain text string.
    """
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    cfg = {"callbacks": [callback]} if callback else {}

    # ── 1. Gemini ──
    if settings.gemini_api_key and not _is_rate_limited("gemini"):
        try:
            import asyncio
            llm = get_gemini_llm()
            resp = await asyncio.wait_for(
                llm.ainvoke(msgs, **({"config": cfg} if cfg else {})),
                timeout=30.0,
            )
            content_str = coerce_content(resp.content)
            return content_str.strip() if content_str else ""
        except Exception as exc:
            is_quota = any(k in str(exc) for k in ("RESOURCE_EXHAUSTED", "429", "quota"))
            if is_quota:
                _mark_rate_limited("gemini", ttl_seconds=60)
                log.warning("gemini_rate_limited", error=str(exc)[:100])
            else:
                log.warning("gemini_failed", error=str(exc)[:100])

    # ── 2. Cloudflare ──
    if settings.use_cloudflare_fallback and not _is_rate_limited("cloudflare"):
        try:
            log.info("llm_fallback_cloudflare")
            return await cloudflare_complete(user, system=system, max_tokens=max_tokens)
        except Exception as exc:
            log.warning("cloudflare_failed", error=str(exc)[:100])

    raise RuntimeError("All LLM providers exhausted or rate-limited")


# ── Token streaming ───────────────────────────────────────────────────────────

async def llm_stream(
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """
    Token-by-token streaming. Provider priority:
      1. Gemini  — via LangChain astream (true token streaming)
      2. Cloudflare — no streaming API, yields whole response as one chunk

    Usage:
        async for token in llm_stream(system, user):
            yield token
    """
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]

    # ── 1. Gemini streaming ──
    if settings.gemini_api_key and not _is_rate_limited("gemini"):
        try:
            llm = get_gemini_llm()
            import asyncio
            async def _stream_with_timeout():
                iterator = llm.astream(msgs).__aiter__()
                while True:
                    try:
                        token = await asyncio.wait_for(iterator.__anext__(), timeout=15.0)
                        yield token
                    except StopAsyncIteration:
                        break

            async for token in _stream_with_timeout():
                yield token
            return
        except Exception as exc:
            is_quota = any(k in str(exc) for k in ("RESOURCE_EXHAUSTED", "429", "quota"))
            if is_quota:
                _mark_rate_limited("gemini", ttl_seconds=60)
            log.warning("gemini_stream_failed", error=str(exc)[:100])

    # ── 2. Cloudflare (non-streaming — yield as single chunk) ──
    if settings.use_cloudflare_fallback:
        try:
            log.info("stream_fallback_cloudflare")
            text = await cloudflare_complete(user, system=system, max_tokens=max_tokens)
            yield text
            return
        except Exception as exc:
            log.warning("cloudflare_stream_failed", error=str(exc)[:100])

    raise RuntimeError("All LLM stream providers exhausted")
