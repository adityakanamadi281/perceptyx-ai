"""
agents/search.py  
----------------------
Search agent with automatic Serper → DuckDuckGo fallback.
"""

from __future__ import annotations

import time

from config.settings import settings
from core.observability import get_logger
from models.schemas import PipelineTrace, SearchOutput, SearchResult
from tools.duckduckgo import ddg_search
from tools.scraper import scrape_urls
from tools.serper import serper_search


async def run_search_agent(
    sub_query: str,
    trace: PipelineTrace,
) -> SearchOutput:
    logger = get_logger("search_agent", trace.run_id)
    t0 = time.perf_counter()
    provider = "serper"

    # ── 1. Try Serper first ───────────────────────────────────────────────────
    try:
        raw_results = await serper_search(sub_query)
        if not raw_results:
            raise ValueError("Serper returned empty results")
        logger.info("serper_ok", count=len(raw_results))
    except Exception as exc:
        logger.warning("serper_failed_fallback_ddg", error=str(exc))
        try:
            raw_results = await ddg_search(sub_query, num=settings.max_search_results)
            provider = "duckduckgo"
            logger.info("ddg_ok", count=len(raw_results))
        except Exception as exc2:
            logger.error("both_search_failed", error=str(exc2))
            raw_results = []

    # ── 2. Scrape URLs concurrently ───────────────────────────────────────────
    urls = [r.url for r in raw_results if r.url]
    scraped = await scrape_urls(urls)

    enriched: list[SearchResult] = []
    for r in raw_results:
        if r.url in scraped:
            text, ts = scraped[r.url]
            enriched.append(r.model_copy(update={"scraped_text": text, "scraped_at": ts}))
        else:
            enriched.append(r)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("search_agent_done", provider=provider, latency_ms=round(latency_ms, 1))

    return SearchOutput(
        sub_query=sub_query,
        results=enriched,
        latency_ms=latency_ms,
        provider_used=provider,
    )

