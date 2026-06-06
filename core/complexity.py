"""
core/complexity.py
──────────────────
Query complexity classifier.
SIMPLE  → definition, fact, single-hop  (< 3s fast path)
MEDIUM  → comparison, explanation       (< 6s normal path)
COMPLEX → multi-hop analysis            (< 10s full path)
RESEARCH → deep research report         (< 30s async job)
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

_COMPLEX_RE = re.compile(
    r"\b(compare|contrast|versus|vs\.?|difference between|pros and cons|"
    r"tradeoffs?|analyze|analyse|explain how|how does|why does|"
    r"what are the|step by step|architecture of)\b",
    re.IGNORECASE,
)

_SIMPLE_RE = re.compile(
    r"^(what is|who is|when (is|was|did)|where is|define |meaning of|"
    r"how (many|much|old|tall|long)|what (year|date|time))\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryComplexity:
    """Classify query complexity from heuristics — no LLM call needed."""
    q = query.strip()

    if _RESEARCH_RE.search(q):
        return QueryComplexity.RESEARCH

    word_count = len(q.split())

    if _COMPLEX_RE.search(q) or word_count > 20:
        return QueryComplexity.COMPLEX

    if _SIMPLE_RE.match(q) and word_count <= 10:
        return QueryComplexity.SIMPLE

    return QueryComplexity.MEDIUM


def needs_planner(complexity: QueryComplexity) -> bool:
    return complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH)


def needs_reason_agent(complexity: QueryComplexity) -> bool:
    """Only invoke Reason Agent for non-trivial queries."""
    return complexity in (QueryComplexity.COMPLEX, QueryComplexity.RESEARCH)


def needs_evaluation(complexity: QueryComplexity) -> bool:
    """Evaluation only runs for research / low-confidence answers."""
    return complexity == QueryComplexity.RESEARCH
