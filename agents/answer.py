"""
agents/answer.py
─────────────────
Synthesises all ReasonOutputs into a final streamed answer with:
  - Token-by-token streaming via llm_stream()
  - Strict inline citation format (enforced by system prompt)
  - Citation verification (strips hallucinated [n] references)
  - Follow-up question generation
  - OpenAI Agents SDK integration
"""
from __future__ import annotations

import json
import re
import time
from typing import AsyncIterator

import structlog

from core.observability import TelemetryCallback, get_logger
from models.schemas import (
    AnswerResponse, Citation, HopChainOutput,
    ImageAnalysis, PipelineTrace, ReasonOutput, SearchOutput,
)

log = structlog.get_logger()

# ── System prompts ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert research synthesiser. Synthesise the provided research summaries into a final direct answer.

CONCISE ANSWER RULE (Mandatory):
- Answer the user's question directly in exactly 2 to 3 sentences (lines).
- Be extremely concise and factual. Do not explain your reasoning process.
- Avoid preamble, drafts, or thinking steps. Output ONLY the final answer.

CITATION RULES:
- Every sentence that uses info from a source MUST end with [n] where n is the source index (e.g., [1], [2]).
- Cite inline only. Do not add a citation list or bibliography at the end.

RESPONSE FORMAT:
You MUST wrap your answer in XML tags exactly like this:
<answer>
Your 2-to-3 sentence answer here with inline [n] citations.
</answer>
"""

_FOLLOW_UP_SYSTEM = """\
You suggest follow-up research questions. Return ONLY a JSON array of exactly 3 strings.
Each question must be specific, concise (under 15 words), and explore a different angle
from the original. No preamble, no explanation — just the JSON array.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_summaries(
    reason_outputs: list[ReasonOutput],
    hop_outputs: list[HopChainOutput],
    search_outputs: list[SearchOutput] | None = None,
) -> str:
    parts = []
    for i, ro in enumerate(reason_outputs, 1):
        sources_str = "\n".join(f"  - {u}" for u in ro.supporting_urls)
        parts.append(
            f"TOPIC {i}: {ro.sub_query}\nSUMMARY: {ro.summary}\nSOURCES:\n{sources_str}"
        )
    if not parts and hop_outputs:
        for ho in hop_outputs:
            snippets = []
            for hop in ho.hops:
                snippets.extend(hop.content_snippets[:2])
            parts.append(
                f"TOPIC: {ho.original_query}\nCONTENT:\n" + "\n".join(snippets[:4])
            )
    if not parts and search_outputs:
        for so in search_outputs:
            snippets = []
            for r in so.results:
                snippets.append(f"[{r.title} ({r.url})]: {r.scraped_text or r.snippet}")
            parts.append(
                f"TOPIC: {so.sub_query}\nCONTENT:\n" + "\n".join(snippets[:5])
            )
    return "\n\n".join(parts)


def _parse_response(raw: str) -> tuple[str, list[Citation]]:
    answer = ""
    citations: list[Citation] = []
    
    import re
    match = re.search(r"<(?:a)?nswer>(.*?)(?:<\/?(?:a)?nswer>|$)", raw, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        if "Draft:" in raw:
            answer = raw.split("Draft:")[1].strip()
        else:
            answer = raw.strip()
    answer = re.sub(r"<\/?(?:a)?nswer>", "", answer, flags=re.IGNORECASE).strip()

    if "<citations>" in raw and "</citations>" in raw:
        cit_raw = raw.split("<citations>")[1].split("</citations>")[0].strip()
        try:
            parsed = json.loads(cit_raw)
            if isinstance(parsed, list):
                for i, c in enumerate(parsed):
                    if not isinstance(c, dict):
                        continue
                    try:
                        src_type = str(c.get("source_type") or "web").lower()
                        if src_type not in ("web", "local", "news", "github"):
                            src_type = "web"
                        citations.append(
                            Citation(
                                index=int(c.get("index") or (i + 1)),
                                title=str(c.get("title") or ""),
                                url=str(c.get("url") or ""),
                                relevant_snippet=str(c.get("relevant_snippet") or ""),
                                source_type=src_type,
                                chunk_id=c.get("chunk_id") or None,
                                source_file=c.get("source_file") or None,
                                published_at=c.get("published_at") or None,
                                repo=c.get("repo") or None,
                            )
                        )
                    except Exception as e:
                        log.warning("citation_parse_item_failed", error=str(e), item=c)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            log.warning("citation_parsing_failed", error=str(e))
    return answer, citations


def _fallback_citations(
    answer_text: str,
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput] | None,
) -> list[Citation]:
    candidates = []
    seen = set()
    
    if search_outputs:
        for so in search_outputs:
            if not so or not so.results:
                continue
            for r in so.results:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    candidates.append({
                        "url": r.url,
                        "title": r.title or "Source",
                        "snippet": r.scraped_text or r.snippet or "",
                        "source_type": "web"
                    })
                    
    for ro in reason_outputs:
        if not ro or not ro.supporting_urls:
            continue
        for url in ro.supporting_urls:
            if url and url not in seen:
                seen.add(url)
                candidates.append({
                    "url": url,
                    "title": "Source",
                    "snippet": "",
                    "source_type": "web"
                })
                
    import re
    indices = [int(m) for m in re.findall(r"\[(\d+)\]", answer_text)]
    unique_indices = sorted(list(set(indices)))
    
    citations = []
    for idx in unique_indices:
        if 1 <= idx <= len(candidates):
            c = candidates[idx - 1]
            citations.append(
                Citation(
                    index=idx,
                    title=c["title"],
                    url=c["url"],
                    relevant_snippet=c["snippet"],
                    source_type=c["source_type"],
                )
            )
        else:
            citations.append(
                Citation(
                    index=idx,
                    title=f"Source {idx}",
                    url="#",
                    relevant_snippet="No preview available",
                    source_type="web",
                )
            )
    return citations


def _verify_citations(answer: str, citations: list[Citation]) -> str:
    """Strip any [n] references that don't correspond to a real citation index."""
    valid_indices = {str(c.index) for c in citations}
    return re.sub(
        r"\[(\d+)\]",
        lambda m: f"[{m.group(1)}]" if m.group(1) in valid_indices else "",
        answer,
    )


# ── Streaming answer agent ─────────────────────────────────────────────────────

async def stream_answer_agent(
    query: str,
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput],
    trace: PipelineTrace,
    hop_outputs: list[HopChainOutput] | None = None,
    image_analysis: ImageAnalysis | None = None,
    audio_transcript: str | None = None,
) -> AsyncIterator[str]:
    """
    Token-by-token streaming answer generator.
    Yields raw text tokens as they are produced by the LLM.
    Caller collects them and then calls parse_streamed_answer() for citations.
    """
    from providers.llm import llm_stream

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

    logger = get_logger("answer_agent", trace.run_id)
    logger.info("stream_answer_start", query=query[:80])

    async for token in llm_stream(_SYSTEM, user_msg):
        yield token


def parse_streamed_answer(
    full_text: str,
    reason_outputs: list[ReasonOutput] | None = None,
    search_outputs: list[SearchOutput] | None = None,
) -> tuple[str, list[Citation]]:
    """Parse the complete streamed text into answer + citations."""
    answer_text, citations = _parse_response(full_text.strip())
    if not citations:
        citations = _fallback_citations(answer_text, reason_outputs or [], search_outputs or [])
    answer_text = _verify_citations(answer_text, citations)
    return answer_text, citations


# ── Blocking answer agent (for non-streaming path) ────────────────────────────

async def run_answer_agent(
    query: str,
    reason_outputs: list[ReasonOutput],
    search_outputs: list[SearchOutput],
    trace: PipelineTrace,
    hop_outputs: list[HopChainOutput] | None = None,
    image_analysis: ImageAnalysis | None = None,
    audio_transcript: str | None = None,
) -> AnswerResponse:
    """Blocking answer synthesis (collects streaming tokens internally)."""
    logger = get_logger("answer_agent", trace.run_id)
    t0 = time.perf_counter()

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
        # Collect streaming tokens into a single string
        from providers.llm import llm_stream
        tokens = []
        async for token in llm_stream(_SYSTEM, user_msg):
            tokens.append(token)
        raw = "".join(tokens)
        answer_text, citations = _parse_response(raw.strip())
        if not citations:
            citations = _fallback_citations(answer_text, reason_outputs, search_outputs)
        answer_text = _verify_citations(answer_text, citations)
    except Exception as exc:
        logger.error("answer_failed", error=str(exc))
        raw_text = " ".join(ro.summary for ro in reason_outputs)
        answer_text = re.sub(r"\s+", " ", re.sub(r"\[\s*\w+\s*\]", "", raw_text)).strip()
        citations = []

    # Generate follow-up questions
    follow_up_questions: list[str] = []
    try:
        follow_up_questions = await generate_follow_ups(query, answer_text, citations)
    except Exception as exc:
        logger.warning("follow_up_gen_failed", error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("answer_done", latency_ms=round(latency_ms, 1), citations=len(citations))

    return AnswerResponse(
        run_id=trace.run_id,
        query=query,
        answer=answer_text,
        citations=citations,
        follow_up_questions=follow_up_questions,
        total_tokens=trace.total_tokens,
        latency_ms=latency_ms,
        image_analysis=image_analysis,
        audio_transcript=audio_transcript,
    )


# ── Follow-up question generator ─────────────────────────────────────────────

async def generate_follow_ups(
    query: str,
    answer: str,
    citations: list[Citation],
) -> list[str]:
    """Generate 3 specific follow-up questions the user might want to ask."""
    from providers.llm import llm_invoke

    prompt = (
        f"Based on this Q&A, suggest 3 specific follow-up questions the user might have.\n\n"
        f"Q: {query}\n"
        f"A: {answer[:600]}...\n\n"
        f"Rules:\n"
        f"- Each question explores a DIFFERENT angle (scope, implementation, comparison, limitation)\n"
        f"- Keep each question concise (under 15 words)\n"
        f"- Make them genuinely useful, not generic filler\n"
        f"- Return ONLY a JSON array of 3 strings, no explanation\n\n"
        f'Example: ["What are the performance implications?", "How does this compare to X?", '
        f'"When would you NOT use this approach?"]'
    )

    try:
        raw = await llm_invoke(_FOLLOW_UP_SYSTEM, prompt, max_tokens=200)
        raw = raw.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        questions = json.loads(raw)
        if isinstance(questions, list):
            return [str(q) for q in questions[:3]]
    except Exception as exc:
        log.debug("follow_up_parse_failed", error=str(exc))

    return []
