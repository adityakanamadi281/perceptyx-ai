"""
core/context_manager.py
-----------------------
Merges, deduplicates, and ranks context chunks from all retrieval hops
before they are passed to the reasoning agent.
Uses cosine similarity on embeddings to deduplicate near-duplicate chunks.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from models.schemas import HopChainOutput

log = structlog.get_logger()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _deduplicate(snippets: list[str], threshold: float = 0.92) -> list[str]:
    """
    Remove near-duplicate snippets using character-level Jaccard similarity
    (fast, no embedding call needed here since we're working with text chunks
    that already went through the embedding pipeline).
    """
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    unique: list[str] = []
    for snippet in snippets:
        if not any(jaccard(snippet, u) >= threshold for u in unique):
            unique.append(snippet)
    return unique


def build_merged_context(
    hop_outputs: list[HopChainOutput],
    max_chars: int = 24_000,
    run_id: str = "",
) -> str:
    """
    Merge context from all hop chain outputs into a single ranked string.

    Ranking heuristic:
      - Local chunks first (more specific to user's domain)
      - Web chunks ordered by hop number (earlier = more relevant)
      - Deduplication removes near-identical passages
      - Hard truncation at max_chars
    """
    t0 = time.perf_counter()
    local_parts: list[str] = []
    web_parts: list[str] = []

    for hop_out in hop_outputs:
        for hop in hop_out.hops:
            for snippet in hop.content_snippets:
                if not snippet or not snippet.strip():
                    continue
                tagged = f"[{hop.source.upper()} | hop {hop.hop_number} | {hop.sub_query[:60]}]\n{snippet.strip()}"
                if hop.source == "local":
                    local_parts.append(tagged)
                else:
                    web_parts.append(tagged)

    local_unique = _deduplicate(local_parts)
    web_unique = _deduplicate(web_parts)

    merged = "\n\n---\n\n".join(local_unique + web_unique)
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n[...truncated]"

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "context_merged",
        run_id=run_id,
        local_chunks=len(local_unique),
        web_chunks=len(web_unique),
        total_chars=len(merged),
        latency_ms=round(latency_ms, 1),
    )
    return merged
