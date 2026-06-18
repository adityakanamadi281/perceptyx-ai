"""
core/orchestrator.py
─────────────────────
Supervisor Architecture with:
  - Query complexity classification
  - Fast path (SIMPLE < 3s)
  - Parallel agent execution
  - Redis answer cache
  - Conditional reasoning / evaluation
  - Real-time token streaming (SSE answer_token events)

CHANGES:
  - _node_reason: parallelised (removed asyncio.sleep(0.3))
  - _node_answer: streams tokens via SSEEventType.ANSWER_TOKEN
  - stream_pipeline: emits answer_token events in real-time
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

import structlog

from config.settings import settings
from core.cache import (
    get_cached_answer,
    set_cached_answer,
)
from core.complexity import classify_query, needs_evaluation, needs_planner, needs_reason_agent
from core.observability import get_logger, record_cache_hit, record_cache_miss, record_latency
from memory.store import get_context_string, save_turn
from models.schemas import (
    AnswerResponse,
    HopChainOutput,
    PipelineState,
    PipelineTrace,
    QueryComplexity,
    QueryRequest,
    SSEEvent,
    SSEEventType,
)

log = structlog.get_logger()


def _emit(state: PipelineState, event_type: SSEEventType, data: dict, **kw) -> None:
    state.sse_queue.append(SSEEvent(event=event_type, run_id=state.run_id, data=data, **kw))


# ── Fast path ─────────────────────────────────────────────────────────────────

async def _fast_path(query: str, run_id: str, trace: PipelineTrace) -> AnswerResponse:
    from agents.answer import run_answer_agent
    from agents.search import run_search_agent
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


# ── Parallel retrieval ────────────────────────────────────────────────────────

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
    if not ps.memory_context and ps.session_id:
        ps.memory_context = await get_context_string(ps.session_id, ps.query)
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
        assert ps.trace is not None
        sub_queries = await plan_sub_queries(ps.query, ps.trace, memory_context=ps.memory_context)
    else:
        sub_queries = [ps.query]
    ps.sub_queries = sub_queries
    _emit(ps, SSEEventType.PLAN_DONE, {"sub_queries": sub_queries})
    return ps.model_dump()


async def _node_route(state: dict) -> dict:
    ps = PipelineState(**state)
    from agents.router import route_all
    assert ps.trace is not None
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
    assert ps.trace is not None
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

    # FIXED: run reason agents in PARALLEL (removed sequential sleep(0.3))
    assert ps.trace is not None
    reason_outs = list(
        await asyncio.gather(*[run_reason_agent(so, ps.trace) for so in search_proxies])
    )
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
    """
    Blocking answer node — collects all streaming tokens internally.
    The streaming is handled separately in stream_pipeline via _node_answer_stream.
    """
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

    assert ps.trace is not None
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
        "follow_up_questions": answer.follow_up_questions,
    }, latency_ms=answer.latency_ms)

    d = ps.model_dump()
    d["_merged_context"] = merged_context
    return d


async def _node_evaluate(state: dict) -> dict:
    ps = PipelineState(**state)
    merged_context = state.get("_merged_context", "")

    if ps.answer and needs_evaluation(ps.complexity):
        from evaluation.evaluator import evaluate_answer
        assert ps.trace is not None
        eval_result = await evaluate_answer(ps.answer, merged_context, ps.trace)
        ps.eval_result = eval_result
        _emit(ps, SSEEventType.EVAL_DONE, eval_result.model_dump())

    d = ps.model_dump()
    d["_merged_context"] = merged_context
    return d


async def _node_save_memory(state: dict) -> dict:
    ps = PipelineState(**state)
    if ps.session_id and ps.answer:
        async def _save():
            await save_turn(ps.session_id, "user", ps.query)
            await save_turn(ps.session_id, "assistant", ps.answer.answer[:800])
            try:
                from memory.store import save_episodic_memory
                await save_episodic_memory(ps.query, ps.answer.answer)
            except Exception as e:
                log.warning("bg_save_episodic_failed", error=str(e))
        asyncio.create_task(_save())
    return ps.model_dump()


async def _node_persist_metrics(state: dict) -> dict:
    ps = PipelineState(**state)
    async def _persist():
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
    asyncio.create_task(_persist())
    return ps.model_dump()


# ── Pipeline nodes list ───────────────────────────────────────────────────────

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

    cached = await get_cached_answer(request.query)
    if cached:
        record_cache_hit("answer")
        logger.info("cache_hit_answer", query=request.query[:60])
        return AnswerResponse(**cached)
    record_cache_miss("answer")

    # Concurrently load memory and run Strategy Advisor
    from core.rlhf import get_strategy_advisor_recommendation
    memory_context, advisor_recommendation = await asyncio.gather(
        get_context_string(request.session_id or "", request.query),
        get_strategy_advisor_recommendation(request.query),
        return_exceptions=True
    )
    if isinstance(memory_context, BaseException):
        memory_context = ""
    if isinstance(advisor_recommendation, BaseException):
        advisor_recommendation = None

    complexity = classify_query(request.query)
    if request.force_research:
        complexity = QueryComplexity.RESEARCH

    # Upgrade complexity based on strategy advisor recommendation (never downgrade)
    if advisor_recommendation in ("SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH"):
        rec_complexity = QueryComplexity(advisor_recommendation)
        order = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 2,
            QueryComplexity.COMPLEX: 3,
            QueryComplexity.RESEARCH: 4
        }
        if order.get(rec_complexity, 0) > order.get(complexity, 0):
            logger.info("complexity_upgraded_by_rlhf_advisor", old=complexity, new=rec_complexity)
            complexity = rec_complexity

    if complexity == QueryComplexity.SIMPLE:
        from agents.router import _corpus_match_score
        corpus_score = await _corpus_match_score(request.query)
        if corpus_score >= settings.rag_score_threshold:
            complexity = QueryComplexity.MEDIUM
            logger.info(
                "upgraded_complexity_for_kb_match",
                query=request.query[:60],
                corpus_score=corpus_score,
            )

    trace = PipelineTrace(run_id=run_id, query=request.query)
    logger.info("pipeline_start", query=request.query[:60], complexity=complexity)

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
        memory_context=memory_context,
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
    assert ps.answer is not None
    return ps.answer


async def stream_pipeline(request: QueryRequest) -> AsyncIterator[SSEEvent]:
    """
    SSE streaming pipeline.
    Emits progress events AND real-time answer tokens (SSEEventType.ANSWER_TOKEN).
    """
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

    # Concurrently load memory and run Strategy Advisor
    from core.rlhf import get_strategy_advisor_recommendation
    memory_context, advisor_recommendation = await asyncio.gather(
        get_context_string(request.session_id or "", request.query),
        get_strategy_advisor_recommendation(request.query),
        return_exceptions=True
    )
    if isinstance(memory_context, BaseException):
        memory_context = ""
    if isinstance(advisor_recommendation, BaseException):
        advisor_recommendation = None

    complexity = classify_query(request.query)
    if request.force_research:
        complexity = QueryComplexity.RESEARCH

    # Upgrade complexity based on strategy advisor recommendation (never downgrade)
    if advisor_recommendation in ("SIMPLE", "MEDIUM", "COMPLEX", "RESEARCH"):
        rec_complexity = QueryComplexity(advisor_recommendation)
        order = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 2,
            QueryComplexity.COMPLEX: 3,
            QueryComplexity.RESEARCH: 4
        }
        if order.get(rec_complexity, 0) > order.get(complexity, 0):
            logger.info("complexity_upgraded_by_rlhf_advisor", old=complexity, new=rec_complexity)
            complexity = rec_complexity

    if complexity == QueryComplexity.SIMPLE:
        from agents.router import _corpus_match_score
        corpus_score = await _corpus_match_score(request.query)
        if corpus_score >= settings.rag_score_threshold:
            complexity = QueryComplexity.MEDIUM
            logger.info(
                "upgraded_complexity_for_kb_match",
                query=request.query[:60],
                corpus_score=corpus_score,
            )

    trace = PipelineTrace(run_id=run_id, query=request.query)
    initial = PipelineState(
        run_id=run_id,
        session_id=request.session_id,
        query=request.query,
        complexity=complexity,
        trace=trace,
        memory_context=memory_context,
    ).model_dump()

    seen = 0

    async def _run():
        nonlocal seen
        state = dict(initial)
        try:
            # Run all nodes EXCEPT the answer node
            pre_answer_nodes = [
                _node_load_memory,
                _node_classify,
                _node_plan,
                _node_route,
                _node_retrieve,
                _node_build_context,
                _node_reason,
            ]

            if complexity == QueryComplexity.SIMPLE:
                # Fast path — just search
                from agents.search import run_search_agent
                _search_trace = PipelineTrace(run_id=run_id, query=request.query)
                event_q.put_nowait(SSEEvent(
                    event=SSEEventType.PROGRESS, run_id=run_id,
                    data={"step": "Searching..."},
                ))
                search_out = await run_search_agent(request.query, _search_trace)
                event_q.put_nowait(SSEEvent(
                    event=SSEEventType.PROGRESS, run_id=run_id,
                    data={"step": "Generating Answer..."},
                ))

                from agents.answer import parse_streamed_answer, stream_answer_agent
                tokens_collected: list[str] = []
                async for token in stream_answer_agent(
                    query=request.query,
                    reason_outputs=[],
                    search_outputs=[search_out],
                    trace=_search_trace,
                ):
                    event_q.put_nowait(SSEEvent(
                        event=SSEEventType.ANSWER_TOKEN,
                        run_id=run_id,
                        data={"token": token},
                    ))
                    tokens_collected.append(token)

                full_text = "".join(tokens_collected)
                answer_text, citations = parse_streamed_answer(full_text, [], [search_out])
                # Generate follow-ups
                from agents.answer import generate_follow_ups
                follow_ups = await generate_follow_ups(request.query, answer_text, citations)

                event_q.put_nowait(SSEEvent(
                    event=SSEEventType.ANSWER_CHUNK,
                    run_id=run_id,
                    data={
                        "answer": answer_text,
                        "citations": [c.model_dump() for c in citations],
                        "follow_up_questions": follow_ups,
                    },
                ))
                return

            # Full path: run pre-answer nodes
            for fn in pre_answer_nodes:
                state = await fn(state)
                ps = PipelineState(**{k: v for k, v in state.items() if not k.startswith("_")})
                while seen < len(ps.sse_queue):
                    event_q.put_nowait(ps.sse_queue[seen])
                    seen += 1

            ps = PipelineState(**{k: v for k, v in state.items() if not k.startswith("_")})
            merged_context = state.get("_merged_context", "")

            # Build search proxies for answer agent
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

            # Stream answer tokens in real-time
            event_q.put_nowait(SSEEvent(
                event=SSEEventType.PROGRESS, run_id=run_id,
                data={"step": "Generating Answer..."},
            ))

            from agents.answer import (
                generate_follow_ups,
                parse_streamed_answer,
                stream_answer_agent,
            )
            tokens_collected = []
            assert ps.trace is not None
            async for token in stream_answer_agent(
                query=request.query,
                reason_outputs=ps.reason_outputs,
                search_outputs=search_proxies,
                trace=ps.trace,
            ):
                event_q.put_nowait(SSEEvent(
                    event=SSEEventType.ANSWER_TOKEN,
                    run_id=run_id,
                    data={"token": token},
                ))
                tokens_collected.append(token)

            full_text = "".join(tokens_collected)
            answer_text, citations = parse_streamed_answer(full_text, ps.reason_outputs, search_proxies)
            follow_ups = await generate_follow_ups(request.query, answer_text, citations)

            event_q.put_nowait(SSEEvent(
                event=SSEEventType.ANSWER_CHUNK,
                run_id=run_id,
                data={
                    "answer": answer_text,
                    "citations": [c.model_dump() for c in citations],
                    "follow_up_questions": follow_ups,
                },
            ))

            # Post-answer: evaluate + save memory
            post_nodes = [_node_evaluate, _node_save_memory, _node_persist_metrics]
            # Inject answer into state for post-processing
            from models.schemas import AnswerResponse
            ps.answer = AnswerResponse(
                run_id=run_id,
                query=request.query,
                answer=answer_text,
                citations=citations,
                follow_up_questions=follow_ups,
                total_tokens=trace.total_tokens,
                latency_ms=0.0,
                complexity=complexity,
            )
            state = ps.model_dump()
            state["_merged_context"] = merged_context

            for fn in post_nodes:
                state = await fn(state)
                ps2 = PipelineState(**{k: v for k, v in state.items() if not k.startswith("_")})
                while seen < len(ps2.sse_queue):
                    event_q.put_nowait(ps2.sse_queue[seen])
                    seen += 1

        except Exception as exc:
            log.error("stream_pipeline_error", error=str(exc))
            event_q.put_nowait(
                SSEEvent(event=SSEEventType.ERROR, run_id=run_id, data={"error": str(exc)})
            )
        finally:
            event_q.put_nowait(SSEEvent(
                event=SSEEventType.TRACE_SUMMARY, run_id=run_id,
                data={
                    "total_tokens": trace.total_tokens,
                    "total_latency_ms": trace.total_latency_ms,
                    "complexity": complexity,
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
