"""
tools/tavily_search.py
----------------------
Tavily search API — purpose-built for LLM retrieval.
Returns pre-cleaned content so the scraper can be skipped.
"""
from __future__ import annotations

import structlog
from tavily import AsyncTavilyClient

from config.settings import settings
from models.schemas import SearchResult

log = structlog.get_logger()


async def tavily_search(query: str, num: int | None = None) -> list[SearchResult]:
    """
    Call Tavily Search API. Returns SearchResult list with pre-cleaned content.
    Set TAVILY_API_KEY in .env.
    """
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not configured")

    n = num or settings.max_search_results
    client = AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    try:
        data = await client.search(
            query=query,
            max_results=n,
            search_depth="advanced",
            include_raw_content=False,
            timeout=6.0,
        )
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed: {exc}")

    results: list[SearchResult] = []
    for item in data.get("results", [])[:n]:
        # Tavily returns pre-cleaned content — use it directly as scraped_text
        raw_content = item.get("raw_content") or item.get("content") or ""
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:300],
                scraped_text=raw_content[:settings.max_scraped_chars] if raw_content else None,
                source="web",
            )
        )

    log.info("tavily_search_ok", query=query[:60], count=len(results))
    return results
