"""
core/hop_chain.py  (v3)
-----------------------
Multi-hop retrieval: routes each sub-query to the correct source(s):
  web_only  → Serper → DDG fallback
  local_only → Qdrant RAG
  hybrid    → local first, gap-detect, then web
  news      → NewsAPI → GNews fallback
  github    → GitHub REST API
"""

from __future__ import annotations

import asyncio
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from agents.rag_agent import run_rag_agent
from agents.search import run_search_agent
from config.settings import settings
from core.observability import TelemetryCallback, get_logger
from models.schemas import (
    HopChainOutput,
    HopResult,
    PipelineTrace,
    RAGOutput,
    RouteDecision,
    RouteMode,
    SearchOutput,
)
from providers.llm import coerce_content, get_gemini_llm
from tools.github_tool import extract_repo_slug, fetch_github_data
from tools.news import fetch_news

_GAP_SYSTEM = """\
You are a knowledge gap detector. Given a query and retrieved content,
identify facts NOT covered that are needed to fully answer the query.
Return ONLY a JSON array of follow-up search queries (max 3, empty if none):
["follow-up 1", "follow-up 2"]
"""


async def _detect_gaps(query: str, context: str, trace: PipelineTrace) -> list[str]:
    callback = TelemetryCallback("hop_chain", trace)
    llm = get_gemini_llm()
    prompt = f"QUERY: {query}\n\nRETRIEVED CONTENT:\n{context[:3000]}\n\nWhat gaps remain?"
    try:
        resp = await asyncio.wait_for(
            llm.ainvoke(
                [SystemMessage(content=_GAP_SYSTEM), HumanMessage(content=prompt)],
                config={"callbacks": [callback]},
            ),
            timeout=10.0,
        )
        raw = coerce_content(resp.content).strip().lstrip("```json").rstrip("```")
        gaps = json.loads(raw)
        return [g for g in gaps if isinstance(g, str) and g.strip()][:3]
    except Exception:
        return []


def _search_to_context(so: SearchOutput) -> str:
    return "\n\n".join(
        f"[WEB:{r.url}] {(r.scraped_text or r.snippet)[:600]}" for r in so.results
    )


def _rag_to_context(ro: RAGOutput) -> str:
    return "\n\n".join(
        f"[LOCAL:{c.source_file}|chunk:{c.chunk_id}] {c.content}" for c in ro.chunks
    )


def _news_to_context(no) -> str:
    return "\n\n".join(
        f"[NEWS:{a.source_name}|{a.published_at}|{a.url}] {a.title}. {a.description or ''}"
        for a in no.articles
    )


def _github_to_context(go) -> str:
    parts = []
    if go.readme_excerpt:
        parts.append(f"[GITHUB:{go.repo}/README]\n{go.readme_excerpt[:800]}")
    for c in go.commits[:5]:
        parts.append(f"[GITHUB:{go.repo}/commit/{c.sha}] {c.author} ({c.date}): {c.message}")
    for p in go.pull_requests[:5]:
        parts.append(f"[GITHUB:{go.repo}/PR#{p.number}] [{p.state}] {p.title}\n{p.body or ''}")
    return "\n\n".join(parts)


async def run_hop_chain(
    sub_query: str,
    route: RouteDecision,
    trace: PipelineTrace,
) -> HopChainOutput:
    logger = get_logger("hop_chain", trace.run_id)
    t0 = time.perf_counter()
    hops: list[HopResult] = []
    ctx_parts: list[str] = []

    # ── NEWS ──────────────────────────────────────────────────────────────────
    if route.mode == RouteMode.NEWS:
        news_out = await fetch_news(sub_query)
        ctx = _news_to_context(news_out)
        ctx_parts.append(ctx)
        hops.append(HopResult(
            hop_number=1, source="news", sub_query=sub_query,
            content_snippets=[f"{a.title} — {a.description or ''}" for a in news_out.articles],
            latency_ms=news_out.latency_ms,
        ))

    # ── GITHUB ────────────────────────────────────────────────────────────────
    elif route.mode == RouteMode.GITHUB:
        repo = extract_repo_slug(sub_query)
        if repo:
            gh_out = await fetch_github_data(repo, sub_query)
            ctx = _github_to_context(gh_out)
            ctx_parts.append(ctx)
            snippets = []
            for c in gh_out.commits[:3]:
                snippets.append(f"commit {c.sha}: {c.message}")
            for p in gh_out.pull_requests[:3]:
                snippets.append(f"PR #{p.number}: {p.title}")
            hops.append(HopResult(
                hop_number=1, source="github", sub_query=sub_query,
                content_snippets=snippets, latency_ms=gh_out.latency_ms,
            ))
        else:
            # No repo found — fall back to web search
            logger.warning("github_no_repo_found_fallback_web", query=sub_query)
            search_out = await run_search_agent(sub_query, trace)
            ctx_parts.append(_search_to_context(search_out))
            hops.append(HopResult(
                hop_number=1, source="web", sub_query=sub_query,
                content_snippets=[r.scraped_text or r.snippet for r in search_out.results],
                latency_ms=search_out.latency_ms,
            ))

    # ── LOCAL ONLY ────────────────────────────────────────────────────────────
    elif route.mode == RouteMode.LOCAL_ONLY:
        rag_out = await run_rag_agent(sub_query, trace)
        ctx = _rag_to_context(rag_out)
        ctx_parts.append(ctx)
        hops.append(HopResult(
            hop_number=1, source="local", sub_query=sub_query,
            content_snippets=[c.content[:400] for c in rag_out.chunks],
            latency_ms=rag_out.latency_ms,
        ))

    # ── WEB ONLY ──────────────────────────────────────────────────────────────
    elif route.mode == RouteMode.WEB_ONLY:
        search_out = await run_search_agent(sub_query, trace)
        ctx = _search_to_context(search_out)
        ctx_parts.append(ctx)
        hops.append(HopResult(
            hop_number=1, source="web", sub_query=sub_query,
            content_snippets=[r.scraped_text or r.snippet for r in search_out.results],
            latency_ms=search_out.latency_ms,
        ))

    # ── HYBRID (multi-hop) ────────────────────────────────────────────────────
    else:
        # Hop 1: local
        rag_out = await run_rag_agent(sub_query, trace)
        ctx = _rag_to_context(rag_out)
        ctx_parts.append(ctx)
        gaps = await _detect_gaps(sub_query, ctx, trace) if rag_out.chunks else []
        hops.append(HopResult(
            hop_number=1, source="local", sub_query=sub_query,
            content_snippets=[c.content[:400] for c in rag_out.chunks],
            gap_queries=gaps, latency_ms=rag_out.latency_ms,
        ))
        logger.info("hop1_local", chunks=len(rag_out.chunks), gaps=len(gaps))

        # Hop 2+: web for gaps
        for hop_n, gap_q in enumerate(gaps[:settings.max_hops - 1], start=2):
            gap_q_str = str(gap_q)
            search_out = await run_search_agent(gap_q_str, trace)
            ctx_parts.append(_search_to_context(search_out))
            hops.append(HopResult(
                hop_number=hop_n, source="web", sub_query=gap_q_str,
                content_snippets=[r.scraped_text or r.snippet for r in search_out.results],
                latency_ms=search_out.latency_ms,
            ))

        # Fallback: no local chunks → web for original
        if not rag_out.chunks:
            search_out = await run_search_agent(sub_query, trace)
            ctx_parts.append(_search_to_context(search_out))
            hops.append(HopResult(
                hop_number=len(hops) + 1, source="web", sub_query=sub_query,
                content_snippets=[r.scraped_text or r.snippet for r in search_out.results],
                latency_ms=search_out.latency_ms,
            ))

    merged = "\n\n---\n\n".join(ctx_parts)
    total_ms = (time.perf_counter() - t0) * 1000
    logger.info("hop_chain_done", mode=route.mode, hops=len(hops),
                ctx_chars=len(merged), latency_ms=round(total_ms, 1))

    return HopChainOutput(
        original_query=sub_query, hops=hops,
        merged_context=merged, total_latency_ms=total_ms,
    )
