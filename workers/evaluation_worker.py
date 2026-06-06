"""
workers/evaluation_worker.py
─────────────────────────────
ARQ worker: run LLM-as-judge evaluation off the request path.
Only triggered for: deep research, research reports, low-confidence answers.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger()


async def evaluate_async(ctx: dict, run_id: str, answer_dict: dict, context: str) -> dict:
    """ARQ job: evaluate_async(run_id, answer_dict, context) → EvalResult dict"""
    from evaluation.evaluator import evaluate_answer
    from models.schemas import AnswerResponse, PipelineTrace

    answer = AnswerResponse(**answer_dict)
    trace = PipelineTrace(run_id=run_id, query=answer.query)
    result = await evaluate_answer(answer, context, trace)
    log.info("async_eval_done", run_id=run_id, passed=result.passed)
    return result.model_dump()
