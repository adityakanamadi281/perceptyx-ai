"""
rag/hybrid_search.py
────────────────────
Hybrid retrieval: Vector Search + BM25 fused via Reciprocal Rank Fusion (RRF).
Pipeline: Vector(top-50) + BM25(top-50) → RRF Fusion → Rerank → Top-10
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import structlog

from config.settings import settings
from rag.reranker import RankedDoc, rerank

log = structlog.get_logger()


def _rrf_fusion(
    vector_results: list[tuple[str, float, dict]],
    bm25_results: list[tuple[str, float, dict]],
    k: int = 60,
) -> list[RankedDoc]:
    """
    Reciprocal Rank Fusion.
    Each result is (content, score, metadata).
    Returns merged list sorted by RRF score.
    """
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, tuple[str, dict]] = {}

    for rank, (content, score, meta) in enumerate(vector_results):
        key = content[:100]
        scores[key] += 1.0 / (k + rank + 1)
        docs[key] = (content, meta)

    for rank, (content, score, meta) in enumerate(bm25_results):
        key = content[:100]
        scores[key] += 1.0 / (k + rank + 1)
        if key not in docs:
            docs[key] = (content, meta)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [RankedDoc(content=docs[k][0], score=v, metadata=docs[k][1]) for k, v in ranked]


def _vector_search(query: str, top_k: int) -> list[tuple[str, float, dict]]:
    try:
        from rag.vectorstore import get_vectorstore
        vs = get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            (doc.page_content, float(score), doc.metadata)
            for doc, score in results
            if float(score) >= settings.rag_score_threshold
        ]
    except Exception as exc:
        log.warning("vector_search_error", error=str(exc))
        return []


def _bm25_search(query: str, candidates: list[str], top_k: int) -> list[tuple[str, float, dict]]:
    """Simple BM25 over candidate corpus using rank_bm25 if available."""
    if not candidates:
        return []
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in candidates]
        bm25 = BM25Okapi(tokenized)
        q_tokens = query.lower().split()
        scores = bm25.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(candidates[i], float(scores[i]), {}) for i in top_indices]
    except ImportError:
        log.debug("rank_bm25_not_available_skipping_bm25")
        return []
    except Exception as exc:
        log.warning("bm25_error", error=str(exc))
        return []


async def hybrid_retrieve(query: str) -> list[RankedDoc]:
    """
    Full hybrid retrieval + reranking pipeline.
    Returns top-K reranked documents.
    """
    t0 = time.perf_counter()
    retrieve_k = settings.rag_retrieve_top_k

    if not settings.enable_hybrid_search:
        # Simple vector-only path
        vector_res = _vector_search(query, retrieve_k)
        docs = [RankedDoc(content=c, score=s, metadata=m) for c, s, m in vector_res]
        return rerank(query, docs)

    vector_res = _vector_search(query, retrieve_k)
    candidate_texts = [c for c, _, _ in vector_res]
    bm25_res = _bm25_search(query, candidate_texts, retrieve_k)

    fused = _rrf_fusion(vector_res, bm25_res)
    reranked = rerank(query, fused)

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "hybrid_retrieve_done",
        query=query[:60],
        vector=len(vector_res),
        bm25=len(bm25_res),
        fused=len(fused),
        final=len(reranked),
        latency_ms=round(latency_ms, 1),
    )
    return reranked
