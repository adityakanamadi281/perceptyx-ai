"""
core/semantic_cache.py
──────────────────────
Semantic cache using embedding similarity.
"How does Python GIL work?" and "Explain the Python Global Interpreter Lock"
both hit the same cached answer.

Uses cosine similarity with a configurable threshold (default: 0.92).
Embeddings stored in Redis as binary blobs.

For production scale, replace the scan loop with Redis Stack's
FT.CREATE + VECTOR index for O(log n) ANN lookup.
"""
from __future__ import annotations

import asyncio
import json
import struct
import time
from functools import lru_cache
from typing import Any

import structlog

log = structlog.get_logger()

_SIMILARITY_THRESHOLD = 0.92
_MAX_SCAN_KEYS = 200  # Limit linear scan for small deployments


@lru_cache(maxsize=1)
def _get_embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("semantic_cache_embed_model_loaded")
        return model
    except Exception as exc:
        log.warning("semantic_cache_embed_unavailable", error=str(exc))
        return None


def _encode(text: str) -> list[float] | None:
    model = _get_embed_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as exc:
        log.warning("embed_failed", error=str(exc))
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for normalised vectors (= dot product)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _pack_vec(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vec(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


async def semantic_cache_get(query: str) -> Any | None:
    """
    Find a cached answer for a semantically similar query.
    Returns the cached answer dict or None.
    """
    try:
        from core.cache import get_redis
    except ImportError:
        return None

    query_vec = await asyncio.to_thread(_encode, query)
    if query_vec is None:
        return None

    try:
        r = get_redis()
        # Scan stored query embedding keys
        cursor = 0
        best_score = 0.0
        best_hash = None

        # Use SCAN instead of KEYS for production safety
        scan_count = 0
        async for key in r.scan_iter("semcache:emb:*", count=50):
            if scan_count >= _MAX_SCAN_KEYS:
                break
            scan_count += 1

            raw = await r.get(key)
            if not raw:
                continue

            stored_vec = _unpack_vec(raw)
            score = _cosine(query_vec, stored_vec)
            if score > best_score:
                best_score = score
                best_hash = key.replace("semcache:emb:", "")

        if best_score >= _SIMILARITY_THRESHOLD and best_hash:
            answer_raw = await r.get(f"semcache:ans:{best_hash}")
            if answer_raw:
                log.info(
                    "semantic_cache_hit",
                    query=query[:60],
                    score=round(best_score, 3),
                )
                return json.loads(answer_raw)
    except Exception as exc:
        log.warning("semantic_cache_get_error", error=str(exc))

    return None


async def semantic_cache_set(query: str, answer: dict, ttl: int = 7200) -> None:
    """Store a query embedding + answer in the semantic cache."""
    try:
        from core.cache import get_redis
        import hashlib
    except ImportError:
        return

    query_vec = await asyncio.to_thread(_encode, query)
    if query_vec is None:
        return

    try:
        r = get_redis()
        cache_hash = hashlib.sha256(query.encode()).hexdigest()[:32]

        # Store embedding as binary
        emb_bytes = _pack_vec(query_vec)
        await r.setex(f"semcache:emb:{cache_hash}", ttl, emb_bytes)

        # Store answer as JSON
        await r.setex(
            f"semcache:ans:{cache_hash}",
            ttl,
            json.dumps(answer, default=str),
        )

        log.debug("semantic_cache_set", query=query[:60], hash=cache_hash[:8])
    except Exception as exc:
        log.warning("semantic_cache_set_error", error=str(exc))
