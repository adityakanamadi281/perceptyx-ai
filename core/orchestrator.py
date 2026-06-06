"""
core/orchestrator.py - v2
─────────────────────────
Supervisor Architecture with:
  - Query complexity classification
  - Fast path (SIMPLE < 3s)
  - Parallel agent execution
  - Redis answer cache
  - Conditional reasoning / evaluation
  - Streaming progress events
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

import structlog

from config.settings import settings
from core.cache import (
    get_cached_answer, set_cached_answer,
    get_cached_query, set_cached_query,
    publish_event,
)
from core.complexity import classify_query, needs_planner, needs_reason_agent, needs_evaluation
from core.observability import get_logger, record_latency, record_cache_hit, record_cache_miss
from memory.store import get_context_string, save_turn
from models.schemas import (
    AnswerResponse, Citation, HopChainOutput, PipelineState, PipelineTrace,
    QueryComplexity, QueryRequest, SSEEvent, SSEEventType,
)

log = structlog.get_logger()


def _emit(state: PipelineState, event_type: SSEEventType, data: dict, **kw) -> None:
    state.sse_queue.append(SSEEvent(event=event_type, run_id=state.run_id, data=data, **kw))


# ── Fast path: SIMPLE queries ─────────────────────────────────────────────────

async def _fast_path(query: str, run_id: str, trace: PipelineTrace) -> AnswerResponse:
    """
    SIMPLE query bypass: Search → Answer only. Target < 3s.
    Skips: planner, router, reason agent, evaluator, multi-hop.
    """
    from agents.search import run_search_agent
    from agents.answer import run_answer_agent
    from models.schemas import RouteDecision, RouteMode

    logger = get_logger("fast_path", run_id)
    logger.info("fast_path_start", query=query[:60])

    search_out = await run_search_agent(query, trace)
    answer = await run_answer_agent(
        query=query,
        reason_outputs=[],
        search_outputs=[search_out],
        trace=trace,
    )
    answer.complexity = QueryComplexity.SIMPLE
    return answer


# ── Parallel supervisor agent execution ───────────────────────────────────────

async def _run_parallel_retrieval(
    sub_queries: list[str],
    route_decisions: list,
    trace: PipelineTrace,
    sse_queue: list,
    run_id: str,
) -> list[HopChainOutput]:
    from core.hop_chain import run_hop_chain

    async def _one(route):
        hop_out = await run_hop_chain(route.sub_query, route, trace)
        sse_queue.append(SSEEvent(
            event=SSEEventType.HOP_RESULT,
            run_id=run_id,
            data={
                "sub_query": route.sub_query,
                "mode": route.mode,
                "hops": len(hop_out.hops),
                "context_chars": len(hop_out.merged_context),
            },
        ))
        return hop_out

    return list(await asyncio.gather(*[_one(r) for r in route_decisions]))


# ── Node functions ────────────────────────────────────────────────────────────

async def _node_load_memory(state: dict) -> dict:
    ps = PipelineState(**state)
    if ps.session_id:
        ps.memory_context = await get_context_string(ps.session_id)
    return ps.model_dump()


async def _node_classify(state: dict) -> dict:
    ps = PipelineState(**state)
    complexity = classify_query(ps.query)
    ps.complexity = complexity
    _emit(ps, SSEEventType.PROGRESS, {"step": "classified", "complexity": complexity})
    return ps.model_dump()


async def _node_plan(state: dict) -> dict:
    ps = PipelineState(**state)
    if needs_planner(ps.complexity):
        from core.planner import plan_sub_queries
        sub_queries = await plan_sub_queries(ps.query, ps.trace, memory_context=ps.memory_context)
    else:
        sub_queries = [ps.query]
    ps.sub_queries = sub_queries
    _emit(ps, SSEEventType.PLAN_DONE, {"sub_queries": sub_queries})
    return ps.model_dump()


async def _node_route(state: dict) -> dict:
    ps = PipelineState(**state)
    from agents.router import route_all
    decisions = await route_all(ps.sub_queries, ps.trace)
    ps.route_decisions = decisions
    for d in decisions:
        _emit(ps, SSEEventType.ROUTE_DECIDED, {
            "sub_query": d.sub_query, "mode": d.mode, "reasoning": d.reasoning,
        })
    return ps.model_dump()


async def _node_retrieve(state: dict) -> dict:
    ps = PipelineState(**state)
    _emit(ps, SSEEventType.PROGRESS, {"step": "Fetching Sources..."})
    hop_outputs = await _run_parallel_retrieval(
        ps.sub_queries, ps.route_decisions, ps.trace, ps.sse_queue, ps.run_id
    )
    ps.hop_outputs = hop_outputs
    _emit(ps, SSEEventType.PROGRESS, {"step": "Reranking Evidence..."})
    return ps.model_dump()


async def _node_build_context(state: dict) -> dict:
    ps = PipelineState(**state)
    from core.context_manager import build_merged_context
    merged = build_merged_context(ps.hop_outputs, run_id=ps.run_id)
    d = ps.model_dump()
    d["_merged_context"] = merged
    return d


async def _node_reason(state: dict) -> dict:
    ps = PipelineState(**state)
    merged_context = state.get("_merged_context", "")

    if not needs_reason_agent(ps.complexity):
        # Skip reasoning for simple/medium queries
        d = ps.model_dump()
        d["_merged_context"] = merged_context
        return d

    from agents.reason import run_reason_agent
    from models.schemas import SearchOutput, SearchResult

    search_proxies = []
    for hop_out in ps.hop_outputs:
        results = [
            SearchResult(
                title=f"[{hop.source.upper()}] {hop.sub_query[:60]}",
                url=f"internal://hop/{hop.hop_number}",
                snippet=hop.content_snippets[0][:400] if hop.content_snippets else "",
                scraped_text=hop.content_snippets[0] if hop.content_snippets else "",
                source="serper",
            )
            for hop in hop_out.hops
        ]
        search_proxies.append(SearchOutput(
            sub_query=hop_out.original_query,
            results=results,
            latency_ms=hop_out.total_latency_ms,
        ))

    reason_outs = []
    for so in search_proxies:
        reason_outs.append(await run_reason_agent(so, ps.trace))
        await asyncio.sleep(0.3)
    ps.reason_outputs = reason_outs

    for ro in ps.reason_outputs:
        _emit(ps, SSEEventType.REASON_CHUNK, {
            "sub_query": ro.sub_query,
            "summary": ro.summary[:200],
            "tokens": ro.tokens_used,
        }, token_delta=ro.tokens_used, latency_ms=ro.latency_ms)

    d = ps.model_dump()
    d["_merged_context"] = merged_context
    return d


async def _node_answer(state: dict) -> dict:
    ps = PipelineState(**state)
    merged_context = state.get("_merged_context", "")
    _emit(ps, SSEEventType.PROGRESS, {"step": "Generating Answer..."})

    from agents.answer import run_answer_agent
    from models.schemas import SearchOutput, SearchResult

    search_proxies = [
        SearchOutput(
            sub_query=ho.original_query,
            results=[
                SearchResult(
                    title=f"[{hop.source.upper()}] hop {hop.hop_number}",
                    url=f"internal://hop/{hop.hop_number}",
                    snippet=hop.content_snippets[0][:300] if hop.content_snippets else "",
                    source="serper",
                )
                for hop in ho.hops
            ],
            latency_ms=ho.total_latency_ms,
        )
        for ho in ps.hop_outputs
    ]

    answer = await run_answer_agent(
        query=ps.query,
        reason_outputs=ps.reason_outputs,
        search_outputs=search_proxies,
        trace=ps.trace,
    )
    answer.complexity = ps.complexity
    ps.answer = answer

    _emit(ps, SSEEventType.ANSWER_CHUNK, {
        "answer": answer.answer,
        "citations": [c.model_dump() for c in answer.citations],
    }, latency_ms=answer.latency_ms)

    d = ps.model_dump()
    d["_merged_context"] = merged_context
    return d


async def _node_evaluate(state: dict) -> dict:
    ps = PipelineState(**state)
    merged_context = state.get("_merged_context", "")

    if ps.answer and needs_evaluation(ps.complexity):
        from evaluation.evaluator import evaluate_answer
        eval_result = await evaluate_answer(ps.answer, merged_context, ps.trace)
        ps.eval_result = eval_result
        _emit(ps, SSEEventType.EVAL_DONE, eval_result.model_dump())

    d = ps.model_dump()
    d["_merged_context"] = merged_context
    return d


async def _node_save_memory(state: dict) -> dict:
    ps = PipelineState(**state)
    if ps.session_id and ps.answer:
        await save_turn(ps.session_id, "user", ps.query)
        await save_turn(ps.session_id, "assistant", ps.answer.answer[:800])
    return ps.model_dump()


async def _node_persist_metrics(state: dict) -> dict:
    """Persist query metrics to PostgreSQL asynchronously."""
    ps = PipelineState(**state)
    try:
        from db.engine import get_session
        from db.models import QueryMetric
        latency_ms = ps.trace.total_latency_ms if ps.trace else 0.0
        async with get_session() as db:
            metric = QueryMetric(
                run_id=ps.run_id,
                complexity=ps.complexity,
                latency_ms=latency_ms,
                tokens=ps.trace.total_tokens if ps.trace else 0,
                cache_hit=ps.cache_hit,
            )
            db.add(metric)
    except Exception as exc:
        log.debug("metrics_persist_error", error=str(exc))
    return ps.model_dump()


# ── Node execution pipeline ───────────────────────────────────────────────────

_FULL_NODES = [
    _node_load_memory,
    _node_classify,
    _node_plan,
    _node_route,
    _node_retrieve,
    _node_build_context,
    _node_reason,
    _node_answer,
    _node_evaluate,
    _node_save_memory,
    _node_persist_metrics,
]


async def _execute_nodes(initial_state: dict, nodes) -> dict:
    state = dict(initial_state)
    for fn in nodes:
        state = await fn(state)
    return state


# ── Public API ────────────────────────────────────────────────────────────────

async def run_pipeline(request: QueryRequest) -> AnswerResponse:
    """Blocking pipeline with cache check and complexity routing."""
    run_id = str(uuid.uuid4())
    logger = get_logger("orchestrator", run_id)
    t0 = time.perf_counter()

    # L2 cache check
    cached = await get_cached_answer(request.query)
    if cached:
        record_cache_hit("answer")
        logger.info("cache_hit_answer", query=request.query[:60])
        return AnswerResponse(**cached)
    record_cache_miss("answer")

    complexity = classify_query(request.query)
    if request.force_research:
        complexity = QueryComplexity.RESEARCH

    trace = PipelineTrace(run_id=run_id, query=request.query)
    logger.info("pipeline_start", query=request.query[:60], complexity=complexity)

    # Fast path for SIMPLE queries
    if complexity == QueryComplexity.SIMPLE:
        answer = await asyncio.wait_for(_fast_path(request.query, run_id, trace), timeout=60)
        answer.run_id = run_id
        await set_cached_answer(request.query, answer.model_dump())
        record_latency(complexity, (time.perf_counter() - t0))
        return answer

    initial = PipelineState(
        run_id=run_id,
        session_id=request.session_id,
        query=request.query,
        complexity=complexity,
        trace=trace,
    ).model_dump()

    final = await asyncio.wait_for(
        _execute_nodes(initial, _FULL_NODES),
        timeout=settings.pipeline_timeout_s,
    )
    ps = PipelineState(**{k: v for k, v in final.items() if not k.startswith("_")})
    trace.total_latency_ms = (time.perf_counter() - t0) * 1000
    record_latency(complexity, time.perf_counter() - t0)

    if ps.answer:
        await set_cached_answer(request.query, ps.answer.model_dump())

    logger.info("pipeline_done", latency_ms=round(trace.total_latency_ms, 1), tokens=trace.total_tokens)
    return ps.answer


async def stream_pipeline(request: QueryRequest) -> AsyncIterator[SSEEvent]:
    """SSE streaming with progress events and cache."""
    run_id = str(uuid.uuid4())
    logger = get_logger("stream", run_id)
    event_q: asyncio.Queue[SSEEvent | None] = asyncio.Queue()

    # Cache check
    cached = await get_cached_answer(request.query)
    if cached:
        record_cache_hit("answer")
        yield SSEEvent(event=SSEEventType.CACHE_HIT, run_id=run_id, data=cached)
        yield SSEEvent(event=SSEEventType.DONE, run_id=run_id, data={})
        return
    record_cache_miss("answer")

    complexity = classify_query(request.query)
    if request.force_research:
        complexity = QueryComplexity.RESEARCH

    trace = PipelineTrace(run_id=run_id, query=request.query)
    initial = PipelineState(
        run_id=run_id,
        session_id=request.session_id,
        query=request.query,
        complexity=complexity,
        trace=trace,
    ).model_dump()

    seen = 0

    async def _run():
        nonlocal seen
        try:
            state = dict(initial)
            nodes = _FULL_NODES if complexity != QueryComplexity.SIMPLE else [
                _node_load_memory, _node_classify, _node_answer, _node_save_memory
            ]
            for fn in nodes:
                state = await fn(state)
                ps = PipelineState(**{k: v for k, v in state.items() if not k.startswith("_")})
                while seen < len(ps.sse_queue):
                    event_q.put_nowait(ps.sse_queue[seen])
                    seen += 1
        except Exception as exc:
            event_q.put_nowait(SSEEvent(event=SSEEventType.ERROR, run_id=run_id, data={"error": str(exc)}))
        finally:
            ps_final = PipelineState(**{k: v for k, v in state.items() if not k.startswith("_")})
            event_q.put_nowait(SSEEvent(
                event=SSEEventType.TRACE_SUMMARY, run_id=run_id,
                data={
                    "total_tokens": trace.total_tokens,
                    "total_latency_ms": trace.total_latency_ms,
                    "complexity": complexity,
                    "eval": ps_final.eval_result.model_dump() if ps_final.eval_result else None,
                },
            ))
            event_q.put_nowait(SSEEvent(event=SSEEventType.DONE, run_id=run_id, data={}))
            event_q.put_nowait(None)

    asyncio.create_task(_run())

    while True:
        event = await event_q.get()
        if event is None:
            break
        yield event
