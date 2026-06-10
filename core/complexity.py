"""
core/complexity.py
──────────────────
Query complexity classifier.
  - Fast path: regex heuristics (no LLM call, ~0ms)
  - LLM fallback: for ambiguous queries only

SIMPLE   → definition, fact, single-hop  (< 3s fast path)
MEDIUM   → comparison, explanation       (< 6s normal path)
COMPLEX  → multi-hop analysis            (< 10s full path)
RESEARCH → deep research report          (< 30s async job)
"""
from __future__ import annotations

import re
from enum import Enum

from config.settings import settings


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    RESEARCH = "RESEARCH"


_RESEARCH_RE = re.compile(
    r"\b(research|survey|comprehensive|in-depth|full report|write a report|"
    r"literature review|white paper|detailed analysis)\b",
    re.IGNORECASE,
)

# Tightened: only fire on truly complex patterns
_COMPLEX_RE = re.compile(
    r"\b(compare|contrast|versus|vs\.?|pros and cons|"
    r"tradeoffs?|analyze|analyse|architecture of|"
    r"step by step implementation|build a|design a)\b",
    re.IGNORECASE,
)

_SIMPLE_RE = re.compile(
    r"^(what is|who is|when (is|was|did)|where is|define |meaning of|"
    r"how (many|much|old|tall|long)|what (year|date|time))\b",
    re.IGNORECASE,
)

# Queries that look complex due to phrasing but are actually MEDIUM
_MEDIUM_OVERRIDE_RE = re.compile(
    r"\b(difference between|explain how|how does|why does|what are the)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryComplexity:
    """Classify query complexity from heuristics — no LLM call needed."""
    q = query.strip()

    if _RESEARCH_RE.search(q):
        return QueryComplexity.RESEARCH

    word_count = len(q.split())

    # Medium-override: "difference between TCP and UDP" should be MEDIUM not COMPLEX
    if _MEDIUM_OVERRIDE_RE.search(q) and word_count <= 15:
        return QueryComplexity.MEDIUM

    if _COMPLEX_RE.search(q) or word_count > 25:
        return QueryComplexity.COMPLEX

    if _SIMPLE_RE.match(q) and word_count <= 10:
        return QueryComplexity.SIMPLE

    return QueryComplexity.MEDIUM


async def classify_query_llm(query: str) -> QueryComplexity:
    """
    LLM-based classifier for ambiguous queries.
    Use this when heuristics return MEDIUM and you want higher confidence.
    """
    from providers.llm import llm_invoke

    PROMPT = (
        "Classify this search query into ONE category:\n"
        "SIMPLE: factual lookups, definitions, 'what is X', 'who is X' (1-2 sentence answer)\n"
        "MEDIUM: explanations, how-something-works, simple comparisons (1-2 paragraphs)\n"
        "COMPLEX: multi-part analysis, detailed comparisons, multi-step questions\n"
        "RESEARCH: 'write a report', 'comprehensive overview', 'literature review'\n\n"
        f"Query: {query}\n\n"
        "Reply with ONLY one word: SIMPLE, MEDIUM, COMPLEX, or RESEARCH"
    )

    try:
        result = await llm_invoke("", PROMPT, max_tokens=5)
        result = result.strip().upper().split()[0]  # take first word only
        return QueryComplexity(result)
    except (ValueError, Exception):
        return QueryComplexity.MEDIUM  # safe default


def needs_planner(complexity: QueryComplexity) -> bool:
    return complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH)


def needs_reason_agent(complexity: QueryComplexity) -> bool:
    """Only invoke Reason Agent for non-trivial queries."""
    return complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH)


def needs_evaluation(complexity: QueryComplexity) -> bool:
    """Evaluation only runs for research / low-confidence answers."""
    return complexity == QueryComplexity.RESEARCH
