"""
agents/search.py
----------------
Search agent with multi-provider aggregation (Serper + Tavily + Firecrawl)
and cross-encoder reranking.
"""
from __future__ import annotations

import time

from config.settings import settings
from core.observability import get_logger
from models.schemas import PipelineTrace, SearchOutput, SearchResult
from tools.scraper import scrape_urls


async def run_search_agent(
    sub_query: str,
    trace: PipelineTrace,
) -> SearchOutput:
    logger = get_logger("search_agent", trace.run_id)
    t0 = time.perf_counter()
    provider = "aggregator"

    # ── 1. Multi-provider search ──────────────────────────────────────────────
    try:
        from tools.search_aggregator import multi_provider_search
        raw_results = await multi_provider_search(sub_query, n=settings.max_search_results)
        if not raw_results:
            raise ValueError("All search providers returned empty results")
        logger.info("aggregator_ok", count=len(raw_results))
    except Exception as exc:
        logger.warning("aggregator_failed_fallback_ddg", error=str(exc))
        try:
            from tools.duckduckgo import ddg_search
            raw_results = await ddg_search(sub_query, num=settings.max_search_results)
            provider = "duckduckgo"
            logger.info("ddg_ok", count=len(raw_results))
        except Exception as exc2:
            logger.error("all_search_failed", error=str(exc2))
            raw_results = []

    # ── 2. Scrape URLs that don't already have content (Tavily/Firecrawl pre-fill) ──
    urls_to_scrape = [
        r.url for r in raw_results
        if r.url and not (r.scraped_text and len(r.scraped_text) > 200)
    ]
    scraped = await scrape_urls(urls_to_scrape)

    enriched: list[SearchResult] = []
    for r in raw_results:
        if r.scraped_text and len(r.scraped_text) > 200:
            # Already has content from Tavily / Firecrawl
            enriched.append(r)
        elif r.url in scraped:
            text, ts = scraped[r.url]
            enriched.append(r.model_copy(update={"scraped_text": text, "scraped_at": ts}))
        else:
            enriched.append(r)

    # ── 3. Cross-encoder reranking ────────────────────────────────────────────
    if settings.enable_reranking and enriched:
        try:
            from rag.reranker import rerank_search_results
            enriched = await rerank_search_results(sub_query, enriched, top_k=min(5, len(enriched)))
            logger.info("reranked", count=len(enriched))
        except Exception as exc:
            logger.warning("rerank_failed", error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("search_agent_done", provider=provider, latency_ms=round(latency_ms, 1))

    return SearchOutput(
        sub_query=sub_query,
        results=enriched,
        latency_ms=latency_ms,
        provider_used=provider if provider == "duckduckgo" else "serper",
    )
