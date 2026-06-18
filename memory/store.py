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
        from db.models import Conversation, Message, Session as DBSession

        async with get_session() as db:
            stmt = select(DBSession).where(DBSession.id == session_id)
            result = await db.execute(stmt)
            sess = result.scalar_one_or_none()
            if sess is None:
                sess = DBSession(id=session_id)
                db.add(sess)
                await db.flush()

            conversation = Conversation(session_id=session_id)
            db.add(conversation)
            await db.flush()

            msg = Message(
                session_id=session_id,
                conversation_id=conversation.id,
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


async def save_episodic_memory(query: str, answer: str) -> None:
    try:
        from rag.vectorstore import upsert_document
        import hashlib
        # Generate stable UUID from query and answer prefix
        raw = f"{query}:{answer[:100]}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        uid = f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        await upsert_document(
            collection_name="episodic_memory",
            text=f"Q: {query}\nA: {answer}",
            metadata={
                "query": query,
                "answer": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            id=uid,
        )
        log.info("episodic_memory_saved", query=query[:60])
    except Exception as e:
        log.warning("save_episodic_memory_failed", error=str(e))


async def get_context_string(session_id: str, query: str = "") -> str:
    """
    Retrieve conversation context from all four memory layers:
      1. Working memory (Redis hot turns or PG backup)
      2. Episodic memory (Qdrant search on episodic_memory)
      3. Semantic memory (Qdrant search on semantic_knowledge with 500ms timeout)
      4. Procedural memory (Qdrant search on procedural_memory)
    Loads them in parallel and merges into a single context block.
    """
    import asyncio

    tasks = []

    # 1. Working Memory
    async def get_working_mem():
        if not session_id:
            return ""
        ctx = await _redis_get_context(session_id)
        if not ctx:
            ctx = await _pg_get_context(session_id)
        return ctx or ""
    tasks.append(get_working_mem())

    # 2. Episodic Memory
    async def get_episodic_mem():
        if not query:
            return ""
        try:
            from rag.vectorstore import similarity_search
            results = await similarity_search("episodic_memory", query, k=3)
            if not results:
                return ""
            lines = []
            for doc, score in results:
                q = doc.metadata.get("query", "")
                a = doc.metadata.get("answer", doc.page_content)
                lines.append(f"- Query: {q}\n  Answer: {a}")
            return "RELEVANT PAST QUESTIONS & ANSWERS:\n" + "\n".join(lines)
        except Exception as e:
            log.warning("episodic_memory_retrieve_failed", error=str(e))
            return ""
    tasks.append(get_episodic_mem())

    # 3. Semantic Memory (distilled facts) with 500ms cap
    async def get_semantic_mem():
        if not query:
            return ""
        try:
            from rag.vectorstore import similarity_search
            results = await asyncio.wait_for(
                similarity_search("semantic_knowledge", query, k=5),
                timeout=0.5
            )
            if not results:
                return ""
            lines = []
            for doc, score in results:
                fact = doc.page_content
                conf = doc.metadata.get("confidence", 1.0)
                lines.append(f"- {fact} (confidence: {conf})")
            return "DISTILLED FACTS & KNOWLEDGE:\n" + "\n".join(lines)
        except Exception as e:
            log.warning("semantic_memory_retrieve_failed", error=str(e))
            return ""
    tasks.append(get_semantic_mem())

    # 4. Procedural Memory (strategy rules)
    async def get_procedural_mem():
        if not query:
            return ""
        try:
            from rag.vectorstore import similarity_search
            results = await similarity_search("procedural_memory", query, k=3)
            if not results:
                return ""
            lines = []
            for doc, score in results:
                rule = doc.page_content
                strategy = doc.metadata.get("strategy", "")
                lines.append(f"- Query type: {rule} -> Strategy: {strategy}")
            return "LEARNED STRATEGY RULES:\n" + "\n".join(lines)
        except Exception as e:
            log.warning("procedural_memory_retrieve_failed", error=str(e))
            return ""
    tasks.append(get_procedural_mem())

    # Load all 4 layers in parallel
    working_ctx, episodic_ctx, semantic_ctx, procedural_ctx = await asyncio.gather(*tasks)

    # Merge them into one context block
    parts = []
    if working_ctx:
        parts.append(working_ctx)
    if episodic_ctx:
        parts.append(episodic_ctx)
    if semantic_ctx:
        parts.append(semantic_ctx)
    if procedural_ctx:
        parts.append(procedural_ctx)

    return "\n\n=========================================\n\n".join(parts)


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
