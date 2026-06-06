"""
workers/search_worker.py
────────────────────────
ARQ worker: executes web search off the request path.
Handles Serper + DuckDuckGo fallback with distributed locking.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from config.settings import settings
from core.cache import (
    acquire_lock, release_lock,
    get_cached_search, set_cached_search,
)

log = structlog.get_logger()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


async def search_web(ctx: dict, query: str, mode: str = "serper") -> dict:
    """
    ARQ job: search_web(query, mode)
    Returns serializable dict matching SearchOutput schema.
    """
    lock_key = f"web_search:{_hash(query)}"

    # Check cache first
    cached = await get_cached_search(query)
    if cached:
        log.info("search_cache_hit", query=query[:60])
        return {**cached, "cache_hit": True}

    # Distributed lock — prevent duplicate concurrent searches
    if not await acquire_lock(lock_key, ttl=20):
        # Another worker is searching; wait briefly and check cache
        import asyncio
        await asyncio.sleep(2)
        cached = await get_cached_search(query)
        if cached:
            return {**cached, "cache_hit": True}

    try:
        result = await _do_search(query, mode)
        await set_cached_search(query, result)
        return result
    finally:
        await release_lock(lock_key)


async def _do_search(query: str, mode: str) -> dict:
    t0 = time.perf_counter()
    results = []

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                settings.serper_endpoint,
                headers={
                    "X-API-KEY": settings.serper_api_key.get_secret_value(),
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": settings.max_search_results},
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("organic", [])[:settings.max_search_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serper",
                })
        provider = "serper"
    except Exception as exc:
        log.warning("serper_failed_trying_ddg", error=str(exc))
        try:
            from duckduckgo_search import AsyncDDGS
            async with AsyncDDGS() as ddgs:
                async for r in ddgs.atext(query, max_results=settings.max_search_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "duckduckgo",
                    })
            provider = "duckduckgo"
        except Exception as exc2:
            log.error("both_search_providers_failed", error=str(exc2))
            provider = "none"

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("search_done", query=query[:60], results=len(results), latency_ms=round(latency_ms, 1))
    return {
        "sub_query": query,
        "results": results,
        "latency_ms": latency_ms,
        "provider_used": provider,
        "cache_hit": False,
    }
