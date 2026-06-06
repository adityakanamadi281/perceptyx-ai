"""
agents/reason.py - v2
─────────────────────
Reasoning agent. Only invoked for COMPLEX/RESEARCH queries.
Uses llm_invoke (Gemini + Cloudflare fallback).
"""
from __future__ import annotations

import json
import time

import structlog

from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace, ReasonOutput, ReasoningStep, SearchOutput

log = structlog.get_logger()

_SYSTEM = """\
You are a rigorous research analyst. Given a sub-query and web-scraped sources:
1. Think step-by-step (chain-of-thought).
2. Extract relevant facts, noting which source each comes from.
3. Cross-check facts; flag contradictions.
4. Produce a concise, factual summary.

Respond ONLY with JSON matching:
{"steps":[{"thought":"...","conclusion":"..."}],"summary":"...","supporting_urls":["url1"]}
"""


def _build_context(search_output: SearchOutput) -> str:
    parts = []
    for i, r in enumerate(search_output.results, 1):
        body = (r.scraped_text or r.snippet)[:600]
        parts.append(f"[{i}] TITLE: {r.title}\nURL: {r.url}\nCONTENT:\n{body}")
    return "\n---\n".join(parts)


async def run_reason_agent(search_output: SearchOutput, trace: PipelineTrace) -> ReasonOutput:
    logger = get_logger("reason_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("reason_agent", trace)

    context = _build_context(search_output)
    user_msg = f"SUB-QUERY: {search_output.sub_query}\n\nSOURCES:\n{context}"

    logger.info("reason_start", sub_query=search_output.sub_query[:60])

    try:
        from providers.llm import llm_invoke
        raw = await llm_invoke(_SYSTEM, user_msg, callback=callback)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        steps = [ReasoningStep(**s) for s in data.get("steps", [])]
        summary = data.get("summary", "")
        supporting_urls = data.get("supporting_urls", [])
    except Exception as exc:
        logger.warning("reason_failed", error=str(exc))
        steps = [ReasoningStep(thought="Parsing failed", conclusion=str(exc))]
        import re
        raw_snippet = search_output.results[0].snippet if search_output.results else ""
        summary = re.sub(r'\s+', ' ', re.sub(r'\[\s*\w+\s*\]', '', raw_snippet)).strip()
        supporting_urls = [r.url for r in search_output.results[:2]]

    tokens = trace.spans[-1].tokens_used if trace.spans else 0
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("reason_done", sub_query=search_output.sub_query[:60], latency_ms=round(latency_ms, 1))

    return ReasonOutput(
        sub_query=search_output.sub_query,
        steps=steps,
        summary=summary,
        supporting_urls=supporting_urls,
        tokens_used=tokens,
        latency_ms=latency_ms,
    )
