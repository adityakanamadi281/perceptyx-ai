"""
agents/answer.py - v2
─────────────────────
Synthesises all ReasonOutputs into a final answer.
Uses llm_invoke (Gemini + Cloudflare fallback).
"""
from __future__ import annotations

import json
import re
import time

import structlog

from core.observability import TelemetryCallback, get_logger
from models.schemas import (
    AnswerResponse, Citation, HopChainOutput,
    ImageAnalysis, PipelineTrace, ReasonOutput, SearchOutput,
)

log = structlog.get_logger()

_SYSTEM = """\
You are a world-class research synthesiser. You receive reasoning summaries from
web, local knowledge base, news, and GitHub sources.

Tasks:
1. Merge into one coherent, factual answer (2-4 paragraphs).
2. Remove redundancy; resolve contradictions (prefer majority view).
3. Include chunk_id for LOCAL, published_at for NEWS, repo for GITHUB citations.

Respond with EXACTLY this format (no extra text):

<answer>
Your prose answer here.
</answer>

<citations>
[{"index":1,"title":"...","url":"...","relevant_snippet":"...","source_type":"web","chunk_id":null,"source_file":null,"published_at":null,"repo":null}]
</citations>
"""


def _build_summaries(
    reason_outputs: list[ReasonOutput],
    hop_outputs: list[HopChainOutput],
    search_outputs: list[SearchOutput] | None = None,
) -> str:
    parts = []
    for i, ro in enumerate(reason_outputs, 1):
        sources_str = "\n".join(f"  - {u}" for u in ro.supporting_urls)
        parts.append(f"TOPIC {i}: {ro.sub_query}\nSUMMARY: {ro.summary}\nSOURCES:\n{sources_str}")
    if not parts and hop_outputs:
        # No reason outputs — build from hops directly
        for ho in hop_outputs:
            snippets = []
            for hop in ho.hops:
                snippets.extend(hop.content_snippets[:2])
            parts.append(f"TOPIC: {ho.original_query}\nCONTENT:\n" + "\n".join(snippets[:4]))
    if not parts and search_outputs:
        # No reason or hop outputs — build from search results directly (fast path)
        for so in search_outputs:
            snippets = []
            for r in so.results:
                snippets.append(f"[{r.title} ({r.url})]: {r.scraped_text or r.snippet}")
            parts.append(f"TOPIC: {so.sub_query}\nCONTENT:\n" + "\n".join(snippets[:5]))
    return "\n\n".join(parts)


def _parse_response(raw: str) -> tuple[str, list[Citation]]:
    answer = ""
    citations: list[Citation] = []
    if "<answer>" in raw and "</answer>" in raw:
        answer = raw.split("<answer>")[1].split("</answer>")[0].strip()
    if "<citations>" in raw and "</citations>" in raw:
        cit_raw = raw.split("<citations>")[1].split("</citations>")[0].strip()
        try:
            for i, c in enumerate(json.loads(cit_raw)):
                citations.append(Citation(
                    index=c.get("index", i + 1),
                    title=c.get("title", ""),
                    url=c.get("url", ""),
                    relevant_snippet=c.get("relevant_snippet", ""),
                    source_type=c.get("source_type", "web"),
                    chunk_id=c.get("chunk_id") or None,
                    source_file=c.get("source_file") or None,
                    published_at=c.get("published_at") or None,
                    repo=c.get("repo") or None,
                ))
        except json.JSONDecodeError:
            pass
    return answer or raw, citations


async def run_answer_agent(
    query: str,
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput],
    trace: PipelineTrace,
    hop_outputs: list[HopChainOutput] | None = None,
    image_analysis: ImageAnalysis | None = None,
    audio_transcript: str | None = None,
) -> AnswerResponse:
    logger = get_logger("answer_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("answer_agent", trace)

    summaries = _build_summaries(reason_outputs, hop_outputs or [], search_outputs)

    enriched_query = query
    if image_analysis:
        enriched_query += f"\n\n[Image context: {image_analysis.description}"
        if image_analysis.extracted_text:
            enriched_query += f" | Text: {image_analysis.extracted_text}"
        enriched_query += "]"
    if audio_transcript:
        enriched_query += f"\n\n[Audio: {audio_transcript}]"

    user_msg = f"ORIGINAL QUESTION: {enriched_query}\n\nRESEARCH SUMMARIES:\n{summaries}"
    logger.info("answer_start", query=query[:80], summaries=len(reason_outputs))

    try:
        from providers.llm import llm_invoke
        raw = await llm_invoke(_SYSTEM, user_msg, callback=callback)
        answer_text, citations = _parse_response(raw.strip())
    except Exception as exc:
        logger.error("answer_failed", error=str(exc))
        raw_text = " ".join(ro.summary for ro in reason_outputs)
        answer_text = re.sub(r'\s+', ' ', re.sub(r'\[\s*\w+\s*\]', '', raw_text)).strip()
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
        image_analysis=image_analysis,
        audio_transcript=audio_transcript,
    )
