"""
tools/duckduckgo.py
-------------------
DuckDuckGo fallback search via duckduckgo-search.
Used automatically when Serper fails or returns no results.
"""

from __future__ import annotations

from models.schemas import SearchResult


async def ddg_search(query: str, num: int = 5) -> list[SearchResult]:
    """
    Search DuckDuckGo and return structured results.
    This is a sync library wrapped in a thread executor.
    """
    import asyncio
    from functools import partial

    from duckduckgo_search import DDGS

    def _sync_search() -> list[SearchResult]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo",
                    )
                )
        return results

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_search)
