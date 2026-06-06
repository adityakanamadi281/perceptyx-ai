"""
core/planner.py - v2
─────────────────────
Breaks a query into sub-queries. Uses llm_invoke (Gemini + Cloudflare fallback).
"""
from __future__ import annotations

import json

from config.settings import settings
from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace

_SYSTEM = """\
You are a research planner. Given a user question (and optional prior conversation
history), decompose it into {n} focused sub-queries for parallel research.

Rules:
- Each sub-query should target a distinct aspect.
- Keep sub-queries concise (max 15 words each).
- Use prior conversation context to avoid repeating already-answered questions.
- Return ONLY a JSON array of strings, nothing else.

{memory_section}
"""


async def plan_sub_queries(
    query: str,
    trace: PipelineTrace,
    n: int | None = None,
    memory_context: str = "",
) -> list[str]:
    n = n or settings.max_sub_queries
    logger = get_logger("planner", trace.run_id)
    callback = TelemetryCallback("planner", trace)

    memory_section = f"Prior conversation:\n{memory_context}" if memory_context else ""
    system = _SYSTEM.format(n=n, memory_section=memory_section)

    logger.info("planning", query=query[:60], has_memory=bool(memory_context))
    try:
        from providers.llm import llm_invoke
        raw = await llm_invoke(system, query, callback=callback)
        raw = raw.strip().lstrip("```json").rstrip("```")
        sub_queries = [q.strip() for q in json.loads(raw) if q.strip()][:n]
        logger.info("plan_done", sub_queries=sub_queries)
        return sub_queries or [query]
    except Exception as exc:
        logger.warning("plan_failed", error=str(exc))
        return [query]
