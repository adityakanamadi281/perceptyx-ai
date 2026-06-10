"""
rag/reranker.py
───────────────
Cross-encoder reranking layer.
  - rerank()              → for RAG RankedDoc objects (existing interface)
  - rerank_search_results() → for SearchResult objects (NEW — used by search agent)

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (faster) or BAAI/bge-reranker-base (better).
Falls back gracefully if sentence-transformers is unavailable.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import structlog

from config.settings import settings

log = structlog.get_logger()


@dataclass
class RankedDoc:
    content: str
    score: float
    metadata: dict


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder
        # Prefer the faster ms-marco model for web search reranking
        model_name = getattr(settings, "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        model = CrossEncoder(model_name, max_length=512)
        log.info("reranker_loaded", model=model_name)
        return model
    except Exception as exc:
        log.warning("reranker_unavailable", error=str(exc))
        return None


def rerank(query: str, docs: list[RankedDoc], top_k: int | None = None) -> list[RankedDoc]:
    """
    Rerank RAG documents using cross-encoder. Returns top_k docs sorted by score.
    Falls back to original order if reranker unavailable.
    """
    if not settings.enable_reranking or not docs:
        return docs[: top_k or settings.rerank_top_k]

    top_k = top_k or settings.rerank_top_k
    t0 = time.perf_counter()

    model = _get_reranker()
    if model is None:
        return docs[:top_k]

    try:
        pairs = [(query, doc.content[:512]) for doc in docs]
        scores = model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc.score = float(score)
        reranked = sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000
        log.info("rerank_done", docs_in=len(docs), docs_out=len(reranked), latency_ms=round(latency_ms, 1))
        return reranked
    except Exception as exc:
        log.warning("rerank_failed", error=str(exc))
        return docs[:top_k]


async def rerank_search_results(
    query: str,
    results: list,  # list[SearchResult]
    top_k: int = 5,
) -> list:
    """
    Rerank SearchResult objects using cross-encoder.
    Runs in a thread pool to avoid blocking the async event loop.
    Returns top_k results sorted by relevance score.
    """
    if not settings.enable_reranking or not results:
        return results[:top_k]

    def _sync_rerank():
        model = _get_reranker()
        if model is None:
            return results[:top_k]

        t0 = time.perf_counter()
        texts = [r.scraped_text or r.snippet or "" for r in results]
        pairs = [(query, t[:512]) for t in texts]

        try:
            scores = model.predict(pairs)
            ranked = sorted(
                zip(scores, results),
                key=lambda x: float(x[0]),
                reverse=True,
            )
            top = [r for _, r in ranked[:top_k]]
            latency_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "rerank_search_done",
                query=query[:60],
                in_count=len(results),
                out_count=len(top),
                latency_ms=round(latency_ms, 1),
            )
            return top
        except Exception as exc:
            log.warning("rerank_search_failed", error=str(exc))
            return results[:top_k]

    return await asyncio.to_thread(_sync_rerank)
