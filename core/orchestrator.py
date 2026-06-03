"""
core/orchestrator.py
--------------------
LangGraph state-machine that wires the full pipeline:

  plan → [search × N || reason × N] → answer

Parallel execution uses asyncio.gather inside the LangGraph nodes.
Telemetry is recorded on the shared PipelineTrace.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from agents.answer import run_answer_agent, run_direct_answer
from agents.reason import run_reason_agent
from agents.search import run_search_agent
from config.settings import settings
from core.observability import get_logger
from core.planner import plan_sub_queries
from core.router import classify_query
from models.schemas import (
    AnswerResponse,
    PipelineState,
    PipelineTrace,
    QueryRequest,
)


# ── Node implementations ──────────────────────────────────────────────────────


async def _node_plan(state: dict[str, Any]) -> dict[str, Any]:
    ps: PipelineState = PipelineState(**state)
    requires_search = await classify_query(ps.query, ps.trace)  # type: ignore[arg-type]
    ps.requires_search = requires_search

    if requires_search:
        sub_queries = await plan_sub_queries(ps.query, ps.trace)  # type: ignore[arg-type]
        ps.sub_queries = sub_queries
    else:
        ps.sub_queries = []
    return ps.model_dump()


async def _node_search_and_reason(state: dict[str, Any]) -> dict[str, Any]:
    """
    Fan out: run (search → reason) pairs for each sub-query concurrently.
    """
    ps = PipelineState(**state)
    if not ps.requires_search:
        return ps.model_dump()

    logger = get_logger("orchestrator", ps.run_id)

    async def _pipeline_one(sub_query: str):
        search_out = await run_search_agent(sub_query, ps.trace)  # type: ignore[arg-type]
        reason_out = await run_reason_agent(search_out, ps.trace)  # type: ignore[arg-type]
        return search_out, reason_out

    logger.info("parallel_start", n=len(ps.sub_queries))
    pairs = await asyncio.gather(*[_pipeline_one(q) for q in ps.sub_queries])

    ps.search_outputs = [p[0] for p in pairs]
    ps.reason_outputs = [p[1] for p in pairs]
    return ps.model_dump()


async def _node_answer(state: dict[str, Any]) -> dict[str, Any]:
    ps = PipelineState(**state)
    if not ps.requires_search:
        answer = await run_direct_answer(
            query=ps.query,
            trace=ps.trace,  # type: ignore[arg-type]
        )
    else:
        answer = await run_answer_agent(
            query=ps.query,
            reason_outputs=ps.reason_outputs,
            search_outputs=ps.search_outputs,
            trace=ps.trace,  # type: ignore[arg-type]
        )
    ps.answer = answer
    return ps.model_dump()


# ── Graph construction ────────────────────────────────────────────────────────


def _build_graph() -> Any:
    g = StateGraph(dict)
    g.add_node("plan", _node_plan)
    g.add_node("search_and_reason", _node_search_and_reason)
    g.add_node("answer", _node_answer)

    g.set_entry_point("plan")
    g.add_edge("plan", "search_and_reason")
    g.add_edge("search_and_reason", "answer")
    g.add_edge("answer", END)
    return g.compile()


_graph = _build_graph()


# ── Public entry-point ────────────────────────────────────────────────────────


async def run_pipeline(request: QueryRequest) -> AnswerResponse:
    """
    Execute the full query → answer pipeline.

    Args:
        request: Validated QueryRequest from the API layer.

    Returns:
        AnswerResponse with answer text, citations, and telemetry metadata.

    Raises:
        asyncio.TimeoutError: if the pipeline exceeds settings.pipeline_timeout_s.
        RuntimeError: on any unhandled internal error (error stored in state).
    """
    run_id = str(uuid.uuid4())
    trace = PipelineTrace(run_id=run_id, query=request.query)
    logger = get_logger("orchestrator", run_id)

    initial_state = PipelineState(
        run_id=run_id,
        query=request.query,
        trace=trace,
    ).model_dump()

    t0 = time.perf_counter()
    logger.info("pipeline_start", query=request.query)

    try:
        final_state_dict = await asyncio.wait_for(
            _graph.ainvoke(initial_state),
            timeout=settings.pipeline_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error("pipeline_timeout", timeout_s=settings.pipeline_timeout_s)
        raise

    final_state = PipelineState(**final_state_dict)

    if final_state.error:
        raise RuntimeError(final_state.error)

    total_ms = (time.perf_counter() - t0) * 1000
    trace.total_latency_ms = total_ms

    logger.info(
        "pipeline_done",
        total_latency_ms=round(total_ms, 1),
        total_tokens=trace.total_tokens,
        num_citations=len(final_state.answer.citations) if final_state.answer else 0,
    )

    # Emit a final trace summary log
    logger.info(
        "pipeline_trace",
        spans=[
            {"agent": s.agent, "latency_ms": round(s.latency_ms, 1), "tokens": s.tokens_used}
            for s in trace.spans
        ],
        total_tokens=trace.total_tokens,
        total_latency_ms=round(total_ms, 1),
    )

    return final_state.answer  # type: ignore[return-value]
