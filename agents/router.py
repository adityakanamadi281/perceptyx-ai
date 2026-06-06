"""
agents/router.py 
----------------------
Router agent: classifies each sub-query into retrieval modes:
  web_only | local_only | hybrid | news | github

Decision logic:
  1. Fast heuristics (keywords, corpus probe)
  2. LLM for ambiguous cases
"""

from __future__ import annotations

import json
import re
import time
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace, RouteDecision, RouteMode
from providers.gemini import get_gemini_llm
from rag.vectorstore import get_vectorstore

_GITHUB_RE = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+")
_NEWS_KEYWORDS = {"news", "breaking", "headline", "article", "reported", "announced",
                   "published", "press", "media", "journalist"}
_GITHUB_KEYWORDS = {"commit", "pull request", "pr", "merge", "repo", "repository",
                     "branch", "issue", "code", "github", "release", "tag", "diff"}

_SYSTEM = """\
You are a retrieval router. Classify the query into ONE of:
- "web_only": needs current/live web info
- "local_only": answerable from the user's local knowledge base
- "hybrid": benefits from both local KB and web
- "news": specifically about news articles or current events
- "github": about a GitHub repository (commits, PRs, code)

Return ONLY JSON:
{"mode":"...", "reasoning":"one sentence", "recency_score":0.0-1.0, "corpus_match_score":0.0-1.0}
"""


def _heuristic_recency(query: str) -> float:
    q = query.lower()
    hits = sum(1 for kw in settings.router_recency_keywords if kw in q)
    return min(hits / 2.0, 1.0)


def _corpus_match_score(query: str) -> float:
    try:
        vs = get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=1)
        if results:
            return round(results[0][1], 3)
    except Exception:
        pass
    return 0.0


def _fast_route(query: str, recency: float, corpus: float) -> RouteMode | None:
    """Return a RouteMode if heuristics are decisive, else None."""
    q_lower = query.lower()
    words = set(q_lower.split())

    # GitHub: has owner/repo pattern or explicit keywords
    if _GITHUB_RE.search(query) or words & _GITHUB_KEYWORDS:
        return RouteMode.GITHUB

    # News: explicit news keywords
    if words & _NEWS_KEYWORDS:
        return RouteMode.NEWS

    # Strong web signal
    if recency >= 0.8 and corpus < 0.4:
        return RouteMode.WEB_ONLY

    # Strong local signal
    if corpus >= 0.65 and recency < 0.3:
        return RouteMode.LOCAL_ONLY

    return None  # ambiguous


async def route_sub_query(sub_query: str, trace: PipelineTrace) -> RouteDecision:
    logger = get_logger("router", trace.run_id)
    t0 = time.perf_counter()

    recency = _heuristic_recency(sub_query)
    corpus = _corpus_match_score(sub_query)

    fast = _fast_route(sub_query, recency, corpus)
    if fast is not None:
        logger.info("route_heuristic", sub_query=sub_query[:60], mode=fast)
        return RouteDecision(
            sub_query=sub_query, mode=fast,
            reasoning="heuristic", recency_score=recency, corpus_match_score=corpus,
        )

    # LLM fallback
    callback = TelemetryCallback("router", trace)
    llm = get_gemini_llm()
    prompt = (
        f"Query: {sub_query}\n"
        f"Heuristic recency score: {recency:.2f}\n"
        f"KB corpus match score: {corpus:.2f}\n"
        "Decide the retrieval mode."
    )
    try:
        resp = await llm.ainvoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
            config={"callbacks": [callback]},
        )
        raw = resp.content.strip().lstrip("```json").rstrip("```")
        data = json.loads(raw)
        mode = RouteMode(data["mode"])
        reasoning = data.get("reasoning", "")
        recency = float(data.get("recency_score", recency))
        corpus = float(data.get("corpus_match_score", corpus))
    except Exception as exc:
        if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
            raise exc
        logger.warning("router_llm_failed", error=str(exc))
        mode = RouteMode.HYBRID
        reasoning = "LLM routing failed; defaulting to hybrid"

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("route_decided", sub_query=sub_query[:60], mode=mode,
                latency_ms=round(latency_ms, 1))

    return RouteDecision(
        sub_query=sub_query, mode=mode, reasoning=reasoning,
        recency_score=recency, corpus_match_score=corpus,
    )


async def route_all(sub_queries: list[str], trace: PipelineTrace) -> list[RouteDecision]:
    decisions = []
    for q in sub_queries:
        decisions.append(await route_sub_query(q, trace))
        await asyncio.sleep(0.5)
    return decisions
