"""
evaluation/evaluator.py - v2
─────────────────────────────
LLM-as-judge. Only runs for RESEARCH complexity or low-confidence answers.
Uses Cloudflare AI fallback when Gemini quota is exhausted.
"""
from __future__ import annotations

import json
import time

import structlog

from config.settings import settings
from models.schemas import AnswerResponse, EvalResult, PipelineTrace

log = structlog.get_logger()

_EVAL_SYSTEM = """\
You are a strict QA evaluator. Score the generated answer on:
- faithfulness (0-1): all claims grounded in retrieved context
- relevance (0-1): answer addresses the question
- citation_coverage (0-1): key claims are cited

Return ONLY JSON (no markdown):
{"faithfulness":0.0,"relevance":0.0,"citation_coverage":0.0,"notes":"one sentence"}
"""


async def evaluate_answer(
    answer: AnswerResponse,
    retrieved_context: str,
    trace: PipelineTrace,
) -> EvalResult:
    if not settings.enable_eval:
        return EvalResult(
            run_id=trace.run_id, faithfulness=1.0, relevance=1.0,
            citation_coverage=1.0, passed=True, notes="Evaluation disabled",
        )

    t0 = time.perf_counter()
    prompt = (
        f"QUESTION: {answer.query}\n\n"
        f"RETRIEVED CONTEXT (first 4000 chars):\n{retrieved_context[:4000]}\n\n"
        f"GENERATED ANSWER:\n{answer.answer}\n\n"
        f"CITATIONS: {json.dumps([c.model_dump() for c in answer.citations])}"
    )

    try:
        from providers.llm import llm_invoke
        raw = await llm_invoke(_EVAL_SYSTEM, prompt)
        raw = raw.strip().lstrip("```json").rstrip("```")
        data = json.loads(raw)
        faithfulness = float(data.get("faithfulness", 0.5))
        relevance = float(data.get("relevance", 0.5))
        citation_coverage = float(data.get("citation_coverage", 0.5))
        notes = data.get("notes", "")
    except Exception as exc:
        log.warning("eval_failed", run_id=trace.run_id, error=str(exc))
        faithfulness = relevance = citation_coverage = 0.5
        notes = f"Eval error: {exc}"

    passed = faithfulness >= settings.eval_faithfulness_threshold
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("eval_done", run_id=trace.run_id, faithfulness=round(faithfulness, 3),
             passed=passed, latency_ms=round(latency_ms, 1))

    return EvalResult(
        run_id=trace.run_id,
        faithfulness=faithfulness,
        relevance=relevance,
        citation_coverage=citation_coverage,
        passed=passed,
        notes=notes,
    )
