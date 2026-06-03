"""
models/schemas.py
-----------------
Pydantic v2 models for every data structure in the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ── Inbound ───────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    max_sources: int = Field(5, ge=1, le=20)
    locale: str = Field("en", pattern=r"^[a-z]{2}$")


# ── Search layer ──────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    scraped_text: str | None = None
    scraped_at: datetime | None = None


class SearchOutput(BaseModel):
    sub_query: str
    results: list[SearchResult]
    latency_ms: float


# ── Reasoning layer ───────────────────────────────────────────────────────────


class ReasoningStep(BaseModel):
    thought: str
    conclusion: str


class ReasonOutput(BaseModel):
    sub_query: str
    steps: list[ReasoningStep]
    summary: str
    supporting_urls: list[str]
    tokens_used: int
    latency_ms: float


# ── Answer layer ──────────────────────────────────────────────────────────────


class Citation(BaseModel):
    index: int
    title: str
    url: str
    relevant_snippet: str


class AnswerResponse(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    citations: list[Citation]
    total_tokens: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Observability ─────────────────────────────────────────────────────────────


class AgentSpan(BaseModel):
    agent: str
    run_id: str
    latency_ms: float
    tokens_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    run_id: str
    query: str
    spans: list[AgentSpan] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── LangGraph state ───────────────────────────────────────────────────────────


class PipelineState(BaseModel):
    """Mutable state threaded through the LangGraph nodes."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    search_outputs: list[SearchOutput] = Field(default_factory=list)
    reason_outputs: list[ReasonOutput] = Field(default_factory=list)
    requires_search: bool = True
    answer: AnswerResponse | None = None
    trace: PipelineTrace | None = None
    error: str | None = None


# ── Error response ─────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    run_id: str
    detail: str
    code: str
