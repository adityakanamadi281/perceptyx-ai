"""
memory/store.py - v2
────────────────────
Two-layer memory:
  L1 (hot):  Redis session cache   — sub-millisecond, TTL-bounded
  L2 (cold): PostgreSQL messages   — durable, queryable, paginated

On startup, Redis is always expected to be available (see docker-compose).
PostgreSQL is the only persistent store. If either is temporarily unavailable
the call logs a warning and returns gracefully — the pipeline continues.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog

from config.settings import settings

log = structlog.get_logger()


# ── L1: Redis ─────────────────────────────────────────────────────────────────

async def _redis_get_context(session_id: str) -> str | None:
    try:
        from core.cache import get_session_turns
        turns = await get_session_turns(session_id)
        if not turns:
            return None
        lines = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
        return "CONVERSATION HISTORY:\n" + "\n".join(lines)
    except Exception as exc:
        log.warning("redis_memory_miss", session_id=session_id, error=str(exc))
        return None


async def _redis_save_turn(session_id: str, role: str, content: str) -> None:
    try:
        from core.cache import append_session_turn
        await append_session_turn(session_id, role, content, settings.memory_max_turns * 2)
    except Exception as exc:
        log.warning("redis_memory_save_error", session_id=session_id, error=str(exc))


# ── L2: PostgreSQL ────────────────────────────────────────────────────────────

async def _pg_save_turn(session_id: str, role: str, content: str) -> None:
    try:
        from sqlalchemy import select
        from db.engine import get_session
        from db.models import Message, Session as DBSession

        async with get_session() as db:
            stmt = select(DBSession).where(DBSession.id == session_id)
            result = await db.execute(stmt)
            sess = result.scalar_one_or_none()
            if sess is None:
                sess = DBSession(id=session_id)
                db.add(sess)
                await db.flush()

            msg = Message(
                session_id=session_id,
                conversation_id=None,
                role=role,
                content=content[:2000],
            )
            db.add(msg)
    except Exception as exc:
        log.warning("pg_save_turn_error", session_id=session_id, error=str(exc))


async def _pg_get_context(session_id: str) -> str | None:
    try:
        from sqlalchemy import select
        from db.engine import get_session
        from db.models import Message

        cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.memory_session_ttl_hours)
        async with get_session() as db:
            stmt = (
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.created_at > cutoff,
                )
                .order_by(Message.created_at.desc())
                .limit(settings.memory_max_turns * 2)
            )
            result = await db.execute(stmt)
            rows = list(reversed(result.scalars().all()))

        if not rows:
            return None
        lines = [f"{r.role.capitalize()}: {r.content}" for r in rows]
        return "CONVERSATION HISTORY:\n" + "\n".join(lines)
    except Exception as exc:
        log.warning("pg_get_context_error", session_id=session_id, error=str(exc))
        return None


async def _pg_clear_session(session_id: str) -> None:
    try:
        from sqlalchemy import delete
        from db.engine import get_session
        from db.models import Message

        async with get_session() as db:
            await db.execute(delete(Message).where(Message.session_id == session_id))
    except Exception as exc:
        log.warning("pg_clear_session_error", session_id=session_id, error=str(exc))


# ── Public API ────────────────────────────────────────────────────────────────

async def save_turn(session_id: str, role: str, content: str) -> None:
    """
    Persist a conversation turn.
    Redis write is fire-and-forget (hot cache).
    PostgreSQL write is the durable record.
    Both run concurrently via asyncio.gather.
    """
    import asyncio
    await asyncio.gather(
        _redis_save_turn(session_id, role, content),
        _pg_save_turn(session_id, role, content),
        return_exceptions=True,
    )


async def get_context_string(session_id: str) -> str:
    """
    Retrieve conversation context.
    Tries Redis first (fast). Falls back to PostgreSQL if Redis is cold/empty.
    Returns empty string if neither has data.
    """
    ctx = await _redis_get_context(session_id)
    if ctx:
        return ctx
    ctx = await _pg_get_context(session_id)
    return ctx or ""


async def clear_session(session_id: str) -> None:
    """
    Remove all session data from Redis and PostgreSQL.
    """
    import asyncio
    from core.cache import get_redis

    async def _redis_clear():
        try:
            r = get_redis()
            await r.delete(
                f"session:{session_id}:context",
                f"session:{session_id}:turns",
            )
        except Exception as exc:
            log.warning("redis_clear_session_error", session_id=session_id, error=str(exc))

    await asyncio.gather(
        _redis_clear(),
        _pg_clear_session(session_id),
        return_exceptions=True,
    )
