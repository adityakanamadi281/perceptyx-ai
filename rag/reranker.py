"""
rag/reranker.py
───────────────
Cross-encoder reranking layer (BGE Reranker).
Retrieve Top-50 → Rerank → Top-10.
Falls back to score-based sort if model unavailable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

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
        model = CrossEncoder(settings.rerank_model, max_length=512)
        log.info("reranker_loaded", model=settings.rerank_model)
        return model
    except Exception as exc:
        log.warning("reranker_unavailable", error=str(exc))
        return None


def rerank(query: str, docs: list[RankedDoc], top_k: int | None = None) -> list[RankedDoc]:
    """
    Rerank documents using cross-encoder. Returns top_k docs sorted by score.
    Falls back to original order if reranker unavailable.
    """
    if not settings.enable_reranking or not docs:
        return docs[:top_k or settings.rerank_top_k]

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
