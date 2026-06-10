"""
workers/research_worker.py
──────────────────────────
True deep research pipeline:
  Phase 1: Generate structured outline (sections + sub-questions)
  Phase 2: Research each section in parallel (semaphore-limited)
  Phase 3: Write each section with evidence
  Phase 4: Assemble + executive summary
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

import structlog

from config.settings import settings
from core.cache import set_job_status, publish_event

log = structlog.get_logger()


# ── Outline generation ────────────────────────────────────────────────────────

async def _generate_outline(query: str) -> dict:
    """
    Generate a research outline with sections and sub-questions per section.
    Returns: {"title": str, "sections": [{"title": str, "sub_questions": [str]}]}
    """
    from providers.llm import llm_invoke

    prompt = (
        f"Create a structured research outline for: {query}\n\n"
        "Return ONLY a JSON object with this structure:\n"
        '{"title": "...", "sections": [{"title": "...", "sub_questions": ["...", "...", "..."]}]}\n\n'
        "Rules:\n"
        "- 3-5 sections covering different angles\n"
        "- 2-3 targeted sub-questions per section\n"
        "- Sub-questions must be specific and searchable\n"
        "- No preamble — JSON only"
    )

    raw = await llm_invoke("You are a research planner. Return valid JSON only.", prompt, max_tokens=600)
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        outline = json.loads(raw)
        # Validate structure
        if "sections" not in outline:
            raise ValueError("Missing 'sections' key")
        return outline
    except Exception as exc:
        log.warning("outline_parse_failed", error=str(exc))
        # Fallback: single section with the original query
        return {
            "title": query,
            "sections": [
                {"title": "Overview", "sub_questions": [query, f"What are the key aspects of {query}?"]},
                {"title": "Details", "sub_questions": [f"How does {query} work?", f"What are the implications of {query}?"]},
            ],
        }


# ── Section research ──────────────────────────────────────────────────────────

async def _research_section(section: dict, trace, semaphore: asyncio.Semaphore) -> tuple[dict, list]:
    """Research a single outline section. Returns (section, list of reason outputs)."""
    from agents.search import run_search_agent
    from agents.reason import run_reason_agent
    from models.schemas import SearchOutput, SearchResult

    results = []
    async with semaphore:
        for sub_q in section.get("sub_questions", []):
            try:
                search_out = await run_search_agent(sub_q, trace)
                # Build a search proxy for the reason agent
                proxy = SearchOutput(
                    sub_query=sub_q,
                    results=search_out.results[:5],
                    latency_ms=search_out.latency_ms,
                )
                reason_out = await run_reason_agent(proxy, trace)
                results.append(reason_out)
            except Exception as exc:
                log.warning("section_research_error", sub_q=sub_q[:60], error=str(exc))
    return section, results


# ── Section writer ────────────────────────────────────────────────────────────

async def _write_section(section_title: str, reason_outputs: list, original_query: str) -> str:
    """Write a full section from evidence."""
    from providers.llm import llm_invoke

    if not reason_outputs:
        return f"*Insufficient information found for this section.*"

    evidence = "\n\n".join(
        f"**{ro.sub_query}**\n{ro.summary}" for ro in reason_outputs
    )

    prompt = (
        f"Write a well-structured section titled '{section_title}' for a research report on: {original_query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Rules:\n"
        "- 2-3 focused paragraphs\n"
        "- Use specific facts from evidence with inline [n] citations where applicable\n"
        "- Academic but readable tone\n"
        "- No section header in your output (it will be added)\n"
        "- Do not start with 'In this section' or similar filler phrases"
    )

    return await llm_invoke("You are a technical research writer.", prompt, max_tokens=800)


# ── Executive summary ─────────────────────────────────────────────────────────

async def _generate_executive_summary(query: str, full_report: str) -> str:
    from providers.llm import llm_invoke
    prompt = (
        f"Write a 3-4 sentence executive summary for this research report on: {query}\n\n"
        f"Report excerpt:\n{full_report[:2000]}...\n\n"
        "The summary should state the main finding and 2-3 key takeaways."
    )
    return await llm_invoke("You are a concise research summariser.", prompt, max_tokens=300)


# ── Main worker ───────────────────────────────────────────────────────────────

async def deep_research(ctx: dict, job_id: str, query: str, session_id: str | None = None) -> dict:
    """
    ARQ job: deep_research(job_id, query)

    True deep research pipeline:
      Phase 1: Outline generation
      Phase 2: Parallel section research (max 3 concurrent)
      Phase 3: Section writing
      Phase 4: Assembly + executive summary
    """
    from models.schemas import PipelineTrace

    log.info("research_start", job_id=job_id, query=query[:80])
    run_id = str(uuid.uuid4())
    trace = PipelineTrace(run_id=run_id, query=query)

    await set_job_status(job_id, "running", progress=0)
    await publish_event("research_events", {"job_id": job_id, "status": "running"})

    try:
        # ── Phase 1: Generate outline ─────────────────────────────────────────
        await set_job_status(job_id, "running", progress=5, result={"step": "Generating outline..."})
        await publish_event("research_events", {"job_id": job_id, "progress": 5, "step": "outline"})

        outline = await _generate_outline(query)
        sections = outline.get("sections", [])
        log.info("research_outline_ready", job_id=job_id, sections=len(sections))

        # ── Phase 2: Research sections in parallel ────────────────────────────
        await set_job_status(job_id, "running", progress=15, result={"step": "Researching sections..."})
        await publish_event("research_events", {"job_id": job_id, "progress": 15, "step": "searching"})

        semaphore = asyncio.Semaphore(3)  # Max 3 parallel section searches
        section_tasks = [
            _research_section(section, trace, semaphore)
            for section in sections
        ]
        section_results = await asyncio.gather(*section_tasks, return_exceptions=True)

        # ── Phase 3: Write each section ───────────────────────────────────────
        section_texts: list[str] = []
        for i, result in enumerate(section_results):
            progress = 60 + int(30 * i / max(len(section_results), 1))
            step = f"Writing section {i+1}/{len(section_results)}..."
            await set_job_status(job_id, "running", progress=progress, result={"step": step})

            if isinstance(result, Exception):
                log.warning("section_failed", error=str(result))
                continue

            section, reason_outputs = result
            section_title = section.get("title", f"Section {i+1}")

            text = await _write_section(section_title, reason_outputs, query)
            section_texts.append(f"## {section_title}\n\n{text}")

        # ── Phase 4: Assemble report ──────────────────────────────────────────
        await set_job_status(job_id, "running", progress=90, result={"step": "Assembling report..."})

        full_report = "\n\n".join(section_texts)
        exec_summary = await _generate_executive_summary(query, full_report)

        final_report = (
            f"# {outline.get('title', query)}\n\n"
            f"**Executive Summary**\n\n{exec_summary}\n\n"
            f"---\n\n"
            f"{full_report}"
        )

        # Build a mock AnswerResponse for compatibility
        from models.schemas import AnswerResponse
        answer = AnswerResponse(
            run_id=run_id,
            query=query,
            answer=final_report,
            citations=[],
            follow_up_questions=[],
            total_tokens=trace.total_tokens,
            latency_ms=0.0,
            complexity="RESEARCH",
        )

        await set_job_status(job_id, "done", progress=100, result=answer.model_dump())
        await publish_event("research_events", {"job_id": job_id, "status": "done", "progress": 100})
        log.info("research_done", job_id=job_id, sections=len(section_texts))
        return answer.model_dump()

    except Exception as exc:
        log.error("research_failed", job_id=job_id, error=str(exc))
        await set_job_status(job_id, "failed", progress=0)
        await publish_event("research_events", {"job_id": job_id, "status": "failed", "error": str(exc)})
        raise
