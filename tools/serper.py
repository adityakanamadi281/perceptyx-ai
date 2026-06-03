"""
tools/serper.py
---------------
Async Serper.dev wrapper.
Returns a list of SearchResult-compatible dicts.
"""

from __future__ import annotations

import httpx

from config.settings import settings
from models.schemas import SearchResult


async def serper_search(query: str, num: int | None = None) -> list[SearchResult]:
    """
    Call the Serper Google Search API and return structured results.

    Args:
        query: The search query string.
        num:   Number of results to request (defaults to settings.max_search_results).

    Returns:
        List of SearchResult objects (without scraped_text — that's the scraper's job).

    Raises:
        httpx.HTTPStatusError: on non-2xx responses.
    """
    n = num or settings.max_search_results
    headers = {
        "X-API-KEY": settings.serper_api_key.get_secret_value(),
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": n, "gl": "us", "hl": "en"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(settings.serper_endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results: list[SearchResult] = []
    for item in data.get("organic", [])[:n]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
        )
    return results
