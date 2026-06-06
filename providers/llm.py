"""
providers/llm.py
────────────────
LLM provider with Gemini primary + Cloudflare AI fallback.
"""
from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

import httpx
import structlog
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

log = structlog.get_logger()


class SafeChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def _clean_message(self, message: BaseMessage) -> BaseMessage:
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                message.content = "".join(parts)
        return message

    async def ainvoke(self, *args, **kwargs):
        res = await super().ainvoke(*args, **kwargs)
        if isinstance(res, BaseMessage):
            self._clean_message(res)
        return res


@lru_cache(maxsize=1)
def get_gemini_llm() -> SafeChatGoogleGenerativeAI:
    return SafeChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_tokens,
        convert_system_message_to_human=True,
    )


async def cloudflare_complete(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """
    Call Cloudflare Workers AI REST API as LLM fallback.
    Endpoint: POST /accounts/{id}/ai/run/{model}
    """
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

    async with httpx.AsyncClient(timeout=60) as client:
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
        # Cloudflare returns {"result": {"response": "..."}, "success": true}
        return data["result"]["response"]


async def llm_invoke(
    system: str,
    user: str,
    callback=None,
    max_tokens: int = 1024,
) -> str:
    """
    Call Gemini first; fall back to Cloudflare AI on rate-limit/quota errors.
    Returns plain text string.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_gemini_llm()
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    cfg = {"callbacks": [callback]} if callback else {}

    try:
        resp = await llm.ainvoke(msgs, config=cfg)
        return resp.content.strip() if resp.content else ""
    except Exception as exc:
        is_quota = any(k in str(exc) for k in ("RESOURCE_EXHAUSTED", "429", "quota"))
        if is_quota and settings.use_cloudflare_fallback:
            log.warning("gemini_quota_fallback_cloudflare", error=str(exc)[:120])
            return await cloudflare_complete(user, system=system, max_tokens=max_tokens)
        raise
