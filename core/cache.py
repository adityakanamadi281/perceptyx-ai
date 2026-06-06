"""
core/cache.py
─────────────
Redis cache layer (L2) with TTL support.
Keys: cache:query:{hash}, cache:search:{hash}, cache:answer:{hash}
Session keys: session:{id}:context
Job keys: job:{id}:status, job:{id}:progress
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from redis.asyncio import ConnectionPool, Redis

from config.settings import settings

log = structlog.get_logger()

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_size,
            decode_responses=True,
        )
    return _pool


def get_redis() -> Redis:
    return Redis(connection_pool=_get_pool())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Generic helpers ───────────────────────────────────────────────────────────

async def cache_get(key: str) -> Any | None:
    try:
        r = get_redis()
        val = await r.get(key)
        if val is None:
            return None
        return json.loads(val)
    except Exception as exc:
        log.warning("cache_get_error", key=key, error=str(exc))
        return None


async def cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        r = get_redis()
        await r.setex(key, ttl, json.dumps(value, default=str))
    except Exception as exc:
        log.warning("cache_set_error", key=key, error=str(exc))


async def cache_delete(key: str) -> None:
    try:
        r = get_redis()
        await r.delete(key)
    except Exception as exc:
        log.warning("cache_delete_error", key=key, error=str(exc))


# ── Specific cache helpers ────────────────────────────────────────────────────

async def get_cached_query(query: str) -> dict | None:
    return await cache_get(f"cache:query:{_hash(query)}")


async def set_cached_query(query: str, result: dict) -> None:
    await cache_set(f"cache:query:{_hash(query)}", result, settings.cache_query_ttl)


async def get_cached_search(query: str) -> dict | None:
    return await cache_get(f"cache:search:{_hash(query)}")


async def set_cached_search(query: str, result: dict) -> None:
    await cache_set(f"cache:search:{_hash(query)}", result, settings.cache_search_ttl)


async def get_cached_answer(query: str) -> dict | None:
    return await cache_get(f"cache:answer:{_hash(query)}")


async def set_cached_answer(query: str, result: dict) -> None:
    await cache_set(f"cache:answer:{_hash(query)}", result, settings.cache_answer_ttl)


# ── Session context ───────────────────────────────────────────────────────────

async def get_session_context(session_id: str) -> str:
    result = await cache_get(f"session:{session_id}:context")
    return result or ""


async def set_session_context(session_id: str, context: str) -> None:
    await cache_set(f"session:{session_id}:context", context, settings.cache_session_ttl)


async def get_session_turns(session_id: str) -> list[dict]:
    result = await cache_get(f"session:{session_id}:turns")
    return result or []


async def append_session_turn(session_id: str, role: str, content: str, max_turns: int = 12) -> None:
    turns = await get_session_turns(session_id)
    turns.append({"role": role, "content": content})
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    await cache_set(f"session:{session_id}:turns", turns, settings.cache_session_ttl)


# ── Job state ─────────────────────────────────────────────────────────────────

async def set_job_status(job_id: str, status: str, progress: int = 0, result: Any = None) -> None:
    await cache_set(f"job:{job_id}:status", {
        "status": status, "progress": progress, "result": result
    }, ttl=3600)


async def get_job_status(job_id: str) -> dict | None:
    return await cache_get(f"job:{job_id}:status")


# ── Distributed locks ─────────────────────────────────────────────────────────

async def acquire_lock(key: str, ttl: int = 30) -> bool:
    """Returns True if lock acquired, False if already held."""
    r = get_redis()
    result = await r.set(f"lock:{key}", "1", nx=True, ex=ttl)
    return result is True


async def release_lock(key: str) -> None:
    await cache_delete(f"lock:{key}")


# ── Pub/Sub ───────────────────────────────────────────────────────────────────

async def publish_event(channel: str, payload: dict) -> None:
    try:
        r = get_redis()
        await r.publish(channel, json.dumps(payload, default=str))
    except Exception as exc:
        log.warning("publish_error", channel=channel, error=str(exc))
