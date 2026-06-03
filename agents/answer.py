"""
agents/answer.py
----------------
Answer agent: synthesises all ReasonOutputs into a single coherent answer
with numbered citations.
"""

from __future__ import annotations

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from core.observability import TelemetryCallback, get_logger
from models.schemas import (
    AnswerResponse,
    Citation,
    PipelineTrace,
    ReasonOutput,
    SearchOutput,
)
from providers.gemini import get_gemini_llm


_SYSTEM = """\
You are a world-class research synthesiser. You will receive multiple
reasoning summaries derived from web sources. Your task:

1. Merge the summaries into one coherent, factual answer.
2. Remove redundancy; resolve any contradictions (prefer majority view).
3. Write in clear prose (2-4 paragraphs).
4. At the end, include citations as a JSON block ONLY — no other JSON in the text.

Respond with this exact format:

<answer>
Your prose answer here.
</answer>

<citations>
[
  {"index": 1, "title": "...", "url": "...", "relevant_snippet": "..."},
  ...
]
</citations>
"""


def _build_summaries(
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput],
) -> str:
    """Format reasoning summaries with source metadata."""
    # Build URL → title map from search results
    url_title: dict[str, str] = {}
    for so in search_outputs:
        for r in so.results:
            url_title[r.url] = r.title

    parts = []
    for i, ro in enumerate(reason_outputs, 1):
        sources = "\n".join(f"  - [{url_title.get(u, u)}]({u})" for u in ro.supporting_urls)
        parts.append(f"TOPIC {i}: {ro.sub_query}\nSUMMARY: {ro.summary}\nSOURCES:\n{sources}")
    return "\n\n".join(parts)


def _parse_response(raw: str, url_title: dict[str, str]) -> tuple[str, list[Citation]]:
    """Extract the prose answer and citation list from the model's response."""
    answer = ""
    citations: list[Citation] = []

    if "<answer>" in raw and "</answer>" in raw:
        answer = raw.split("<answer>")[1].split("</answer>")[0].strip()

    if "<citations>" in raw and "</citations>" in raw:
        cit_raw = raw.split("<citations>")[1].split("</citations>")[0].strip()
        try:
            cit_data = json.loads(cit_raw)
            citations = [
                Citation(
                    index=c.get("index", i + 1),
                    title=c.get("title", url_title.get(c.get("url", ""), "")),
                    url=c.get("url", ""),
                    relevant_snippet=c.get("relevant_snippet", ""),
                )
                for i, c in enumerate(cit_data)
            ]
        except json.JSONDecodeError:
            pass

    if not answer:
        answer = raw  # fallback: return raw text

    return answer, citations


async def run_answer_agent(
    query: str,
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput],
    trace: PipelineTrace,
) -> AnswerResponse:
    """
    Synthesise all reasoning into a final answer with citations.

    Args:
        query:          Original user query.
        reason_outputs: All ReasonOutput objects from parallel reasoning.
        search_outputs: All SearchOutput objects (used for URL→title mapping).
        trace:          Shared PipelineTrace for telemetry.

    Returns:
        AnswerResponse with prose answer and Citation list.
    """
    logger = get_logger("answer_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("answer_agent", trace)
    llm = get_gemini_llm()

    url_title: dict[str, str] = {}
    for so in search_outputs:
        for r in so.results:
            url_title[r.url] = r.title

    summaries = _build_summaries(reason_outputs, search_outputs)
    user_msg = f"ORIGINAL QUESTION: {query}\n\nRESEARCH SUMMARIES:\n{summaries}"

    logger.info("answer_start", query=query, num_summaries=len(reason_outputs))

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)],
            config={"callbacks": [callback]},
        )
        raw = response.content.strip()
        answer_text, citations = _parse_response(raw, url_title)
    except Exception as exc:
        logger.error("answer_failed", error=str(exc))
        answer_text = " ".join(ro.summary for ro in reason_outputs)
        citations = []

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("answer_done", latency_ms=round(latency_ms, 1), citations=len(citations))

    return AnswerResponse(
        run_id=trace.run_id,
        query=query,
        answer=answer_text,
        citations=citations,
        total_tokens=trace.total_tokens,
        latency_ms=latency_ms,
    )


_DIRECT_SYSTEM = """\
You are a helpful, extremely knowledgeable, and accurate AI assistant.
Answer the user's question directly.
Keep your response concise, structured, and informative.
Use formatting (like lists, bolding) where appropriate.
Since you are answering directly without web search results, do NOT add any source citations.
"""


async def run_direct_answer(
    query: str,
    trace: PipelineTrace,
) -> AnswerResponse:
    """
    Directly answer the user query without web search or cross-checking.
    """
    logger = get_logger("direct_answer_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("direct_answer_agent", trace)
    llm = get_gemini_llm()

    logger.info("direct_answer_start", query=query)
    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_DIRECT_SYSTEM), HumanMessage(content=query)],
            config={"callbacks": [callback]},
        )
        answer_text = response.content.strip()
    except Exception as exc:
        logger.error("direct_answer_failed", error=str(exc))
        answer_text = f"An error occurred while generating direct answer: {str(exc)}"

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("direct_answer_done", latency_ms=round(latency_ms, 1))

    return AnswerResponse(
        run_id=trace.run_id,
        query=query,
        answer=answer_text,
        citations=[],
        total_tokens=trace.total_tokens,
        latency_ms=latency_ms,
    )
