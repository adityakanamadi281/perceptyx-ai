"""
agents/search.py
----------------
Search agent: for a given sub-query,
  1. Calls Serper to get organic results
  2. Scrapes each result URL in parallel
  3. Returns a SearchOutput with filled-in scraped_text fields
"""

from __future__ import annotations

import time

from config.settings import settings
from core.observability import get_logger
from models.schemas import PipelineTrace, SearchOutput, SearchResult
from tools.scraper import scrape_urls
from tools.serper import serper_search


async def run_search_agent(
    sub_query: str,
    trace: PipelineTrace,
) -> SearchOutput:
    """
    Execute the full search-and-scrape cycle for one sub-query.

    Args:
        sub_query: A focused query string from the planner.
        trace:     Shared PipelineTrace (search agent emits no LLM tokens
                   so we only log timing here).

    Returns:
        SearchOutput with scraped content attached to each result.
    """
    logger = get_logger("search_agent", trace.run_id)
    t0 = time.perf_counter()

    logger.info("search_start", sub_query=sub_query)

    # ── 1. Retrieve organic results ──────────────────────────────────────────
    raw_results = await serper_search(sub_query)
    logger.info("serper_done", count=len(raw_results))

    # ── 2. Scrape URLs concurrently ───────────────────────────────────────────
    urls = [r.url for r in raw_results]
    scraped = await scrape_urls(urls)
    logger.info("scrape_done", scraped=len(scraped), total=len(urls))

    # ── 3. Merge scraped text back into results ───────────────────────────────
    enriched: list[SearchResult] = []
    for r in raw_results:
        if r.url in scraped:
            text, ts = scraped[r.url]
            enriched.append(r.model_copy(update={"scraped_text": text, "scraped_at": ts}))
        else:
            enriched.append(r)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("search_agent_done", latency_ms=round(latency_ms, 1))

    return SearchOutput(
        sub_query=sub_query,
        results=enriched,
        latency_ms=latency_ms,
    )
