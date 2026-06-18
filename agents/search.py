"""
agents/search.py
----------------
Search agent with parallel web search and LLM knowledge extraction,
semantic memory recall (500ms cap), and cross-encoder reranking.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

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

    # ── 1. Parallel search & LLM parametric knowledge extraction ──────────────
    from agents.llm_knowledge import run_parallel_search_and_llm

    web_results, llm_facts = await run_parallel_search_and_llm(sub_query, trace)
    raw_results = list(web_results)

    # ── 2. Pin LLM knowledge as the top result ────────────────────────────────
    if llm_facts and settings.enable_llm_knowledge:
        facts_summary = "Parametric LLM Knowledge:\n" + "\n".join(
            f"- {f.get('fact')} (confidence: {f.get('confidence')})" for f in llm_facts
        )
        llm_result = SearchResult(
            title="LLM Parametric Knowledge",
            url="internal://llm-knowledge",
            snippet=facts_summary,
            scraped_text=facts_summary,
            source="local",
        )
        raw_results.insert(0, llm_result)

    # ── 3. Recall relevant semantic memory hits (500ms cap) ───────────────────
    semantic_results = []
    try:
        from rag.vectorstore import similarity_search

        # similarity_search has a hard 500ms timeout cap
        semantic_docs = await asyncio.wait_for(
            similarity_search("semantic_knowledge", sub_query, k=3),
            timeout=0.5,
        )
        for doc, score in semantic_docs:
            semantic_results.append(
                SearchResult(
                    title=f"Semantic Memory (score: {score:.2f})",
                    url="internal://semantic-memory",
                    snippet=doc.page_content,
                    scraped_text=doc.page_content,
                    source="local",
                )
            )
        logger.info("semantic_memory_recalled", count=len(semantic_results))
    except Exception as exc:
        logger.warning("semantic_memory_recall_failed", error=str(exc))

    # Append semantic hits to raw results
    raw_results.extend(semantic_results)

    # ── 4. Scrape URLs that don't already have content ────────────────────────
    urls_to_scrape = [
        r.url
        for r in raw_results
        if r.url
        and not r.url.startswith("internal://")
        and not (r.scraped_text and len(r.scraped_text) > 200)
    ]
    scraped = await scrape_urls(urls_to_scrape)

    enriched: list[SearchResult] = []
    for r in raw_results:
        if r.url and r.url.startswith("internal://"):
            enriched.append(r)
        elif r.scraped_text and len(r.scraped_text) > 200:
            # Already has content from Tavily / Firecrawl
            enriched.append(r)
        elif r.url in scraped:
            text, ts = scraped[r.url]
            enriched.append(r.model_copy(update={"scraped_text": text, "scraped_at": ts}))
        else:
            enriched.append(r)

    # ── 5. Cross-encoder reranking ────────────────────────────────────────────
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
