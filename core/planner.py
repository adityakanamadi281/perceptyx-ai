"""
core/planner.py
---------------
Breaks a high-level user query into focused sub-queries using Gemini.
Returns a list of strings, one per parallel search branch.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace
from providers.gemini import get_gemini_llm


_SYSTEM = """\
You are a research planner. Given a user question, decompose it into
{n} focused sub-queries that together will provide a comprehensive answer.
Rules:
- Each sub-query should target a distinct aspect.
- Keep sub-queries concise (max 15 words each).
- Return ONLY a JSON array of strings, nothing else.
Example output: ["sub-query 1", "sub-query 2", "sub-query 3"]
"""


async def plan_sub_queries(
    query: str,
    trace: PipelineTrace,
    n: int | None = None,
) -> list[str]:
    """
    Decompose *query* into at most *n* sub-queries.

    Args:
        query: Original user question.
        trace: Shared PipelineTrace for telemetry.
        n:     Override for max_sub_queries setting.

    Returns:
        List of sub-query strings (always ≥ 1, original query used as fallback).
    """
    n = n or settings.max_sub_queries
    logger = get_logger("planner", trace.run_id)
    callback = TelemetryCallback("planner", trace)
    llm = get_gemini_llm()

    messages = [
        SystemMessage(content=_SYSTEM.format(n=n)),
        HumanMessage(content=query),
    ]

    logger.info("planning", query=query, max_sub_queries=n)
    try:
        response = await llm.ainvoke(messages, config={"callbacks": [callback]})
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_queries: list[str] = json.loads(raw)
        sub_queries = [q.strip() for q in sub_queries if q.strip()][:n]
        logger.info("plan_done", sub_queries=sub_queries)
        return sub_queries or [query]
    except Exception as exc:
        logger.warning("plan_failed", error=str(exc), fallback="original query")
        return [query]
