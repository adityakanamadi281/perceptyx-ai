"""
agents/reason.py
----------------
Reasoning agent: given a sub-query and its search results,
calls Gemini with chain-of-thought prompting to:
  - Identify relevant facts
  - Cross-check and fact-check across sources
  - Produce a concise summary with supporting URLs
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace, ReasonOutput, ReasoningStep, SearchOutput
from providers.gemini import get_gemini_llm


_SYSTEM = """\
You are a rigorous research analyst. You will be given a sub-query and
web-scraped content from multiple sources. Your task:

1. Think step-by-step (chain-of-thought).
2. Extract relevant facts, noting which source each comes from.
3. Cross-check facts across sources; flag any contradictions.
4. Produce a concise, factual summary that answers the sub-query.

Respond ONLY with a JSON object matching this schema (no extra text):
{
  "steps": [
    {"thought": "...", "conclusion": "..."}
  ],
  "summary": "...",
  "supporting_urls": ["url1", "url2"]
}
"""


def _build_context(search_output: SearchOutput) -> str:
    """Format scraped results into a numbered context block."""
    parts = []
    for i, r in enumerate(search_output.results, 1):
        body = r.scraped_text or r.snippet
        parts.append(f"[{i}] TITLE: {r.title}\nURL: {r.url}\nCONTENT:\n{body}\n")
    return "\n---\n".join(parts)


async def run_reason_agent(
    search_output: SearchOutput,
    trace: PipelineTrace,
) -> ReasonOutput:
    """
    Run chain-of-thought reasoning over SearchOutput.

    Args:
        search_output: Populated SearchOutput from the search agent.
        trace:         Shared PipelineTrace for telemetry.

    Returns:
        ReasonOutput with structured reasoning steps and summary.
    """
    logger = get_logger("reason_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("reason_agent", trace)
    llm = get_gemini_llm()

    context = _build_context(search_output)
    user_msg = f"SUB-QUERY: {search_output.sub_query}\n\nSOURCES:\n{context}"

    logger.info("reason_start", sub_query=search_output.sub_query)

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)],
            config={"callbacks": [callback]},
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)

        steps = [ReasoningStep(**s) for s in data.get("steps", [])]
        summary = data.get("summary", "")
        supporting_urls = data.get("supporting_urls", [])

    except Exception as exc:
        logger.warning("reason_failed", error=str(exc))
        steps = [ReasoningStep(thought="Parsing failed", conclusion=str(exc))]
        summary = search_output.results[0].snippet if search_output.results else ""
        supporting_urls = [r.url for r in search_output.results[:2]]

    # Collect tokens from the most recent span
    tokens = 0
    if trace.spans:
        tokens = trace.spans[-1].tokens_used

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "reason_done",
        sub_query=search_output.sub_query,
        latency_ms=round(latency_ms, 1),
        tokens=tokens,
    )

    return ReasonOutput(
        sub_query=search_output.sub_query,
        steps=steps,
        summary=summary,
        supporting_urls=supporting_urls,
        tokens_used=tokens,
        latency_ms=latency_ms,
    )
