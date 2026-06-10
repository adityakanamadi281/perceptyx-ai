"""
tools/firecrawl_search.py
--------------------------
Firecrawl Search API — returns web search results with deep-scraped
markdown content. Replaces Brave Search in the aggregator.
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from config.settings import settings
from models.schemas import SearchResult

log = structlog.get_logger()

_FIRECRAWL_SEARCH_ENDPOINT = "https://api.firecrawl.dev/v1/search"


async def firecrawl_search(query: str, num: int | None = None) -> list[SearchResult]:
    """
    Call Firecrawl Search API using the official SDK. Returns results with clean markdown content.
    Set FIRECRAWL_API_KEY in .env.
    """
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not configured")

    n = num or settings.max_search_results

    def _sync_search():
        from firecrawl import Firecrawl
        fc = Firecrawl(api_key=settings.firecrawl_api_key.get_secret_value())
        return fc.search(query=query, limit=n)

    try:
        search_data = await asyncio.to_thread(_sync_search)
    except Exception as exc:
        raise RuntimeError(f"Firecrawl SDK search failed: {exc}")

    results: list[SearchResult] = []
    web_results = getattr(search_data, "web", None) or []
    for item in web_results[:n]:
        meta = getattr(item, "metadata", None)
        meta_url = getattr(meta, "url", "") if meta else ""
        meta_title = getattr(meta, "title", "") if meta else ""
        meta_desc = getattr(meta, "description", "") if meta else ""

        url = getattr(item, "url", None) or meta_url or ""
        title = getattr(item, "title", None) or meta_title or ""
        description = getattr(item, "description", None) or meta_desc or ""
        markdown_content = getattr(item, "markdown", None) or ""

        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=description[:300],
                # Firecrawl returns clean markdown — use it directly
                scraped_text=(
                    markdown_content[:settings.max_scraped_chars]
                    if markdown_content else None
                ),
                source="web",
            )
        )

    log.info("firecrawl_search_ok", query=query[:60], count=len(results))
    return results


async def firecrawl_scrape(url: str) -> str:
    """
    Scrape a single URL with Firecrawl. Returns clean markdown.
    Used as a premium scraper fallback.
    """
    if not settings.firecrawl_api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not configured")

    scrape_endpoint = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": url,
        "formats": ["markdown"],
    }
    headers = {
        "Authorization": f"Bearer {settings.firecrawl_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(scrape_endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    markdown = data.get("data", {}).get("markdown", "")
    return markdown[:settings.max_scraped_chars]
