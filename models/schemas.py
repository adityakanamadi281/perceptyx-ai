"""
models/schemas.py - v2
──────────────────────
All Pydantic models. Extended with QueryComplexity, ResearchRequest,
StreamProgress, and updated SSEEventType.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    max_sources: int = Field(5, ge=1, le=20)
    locale: str = Field("en", pattern=r"^[a-z]{2}$")
    session_id: str | None = Field(None)
    image_context: str | None = Field(None)
    audio_transcript: str | None = Field(None)
    force_research: bool = Field(False, description="Force RESEARCH complexity path")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    session_id: str | None = Field(None)
    depth: Literal["quick", "standard", "deep"] = Field("standard")


# ── Multimodal ────────────────────────────────────────────────────────────────

class ImageAnalysis(BaseModel):
    description: str
    extracted_text: str | None = None
    detected_entities: list[str] = Field(default_factory=list)
    model_used: str = "gemini-2.5-flash"


class AudioTranscription(BaseModel):
    transcript: str
    language: str | None = None
    duration_s: float | None = None
    model_used: str = "faster-whisper"


# ── Routing ───────────────────────────────────────────────────────────────────

class RouteMode(str, Enum):
    WEB_ONLY = "web_only"
    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"
    NEWS = "news"
    GITHUB = "github"


class RouteDecision(BaseModel):
    sub_query: str
    mode: RouteMode
    reasoning: str
    recency_score: float = Field(ge=0.0, le=1.0)
    corpus_match_score: float = Field(ge=0.0, le=1.0)


# ── Query complexity ──────────────────────────────────────────────────────────

class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    RESEARCH = "RESEARCH"


# ── Search layer ──────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    scraped_text: str | None = None
    scraped_at: datetime | None = None
    source: Literal["serper", "duckduckgo", "web", "local", "news", "github"] = "serper"


class SearchOutput(BaseModel):
    sub_query: str
    results: list[SearchResult]
    latency_ms: float
    provider_used: Literal["serper", "duckduckgo"] = "serper"
    cache_hit: bool = False


# ── News layer ────────────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    title: str
    url: str
    source_name: str
    published_at: str | None = None
    description: str | None = None
    content: str | None = None
    provider: Literal["newsapi", "gnews"] = "newsapi"


class NewsOutput(BaseModel):
    sub_query: str
    articles: list[NewsArticle]
    latency_ms: float


# ── GitHub layer ──────────────────────────────────────────────────────────────

class GitHubCommit(BaseModel):
    sha: str
    message: str
    author: str
    date: str
    url: str


class GitHubPR(BaseModel):
    number: int
    title: str
    state: str
    body: str | None = None
    author: str
    created_at: str
    url: str


class GitHubOutput(BaseModel):
    repo: str
    commits: list[GitHubCommit] = Field(default_factory=list)
    pull_requests: list[GitHubPR] = Field(default_factory=list)
    readme_excerpt: str | None = None
    latency_ms: float


# ── RAG layer ─────────────────────────────────────────────────────────────────

class RAGChunk(BaseModel):
    content: str
    source_file: str
    page: int | None = None
    score: float
    chunk_id: str


class RAGOutput(BaseModel):
    sub_query: str
    chunks: list[RAGChunk]
    latency_ms: float
    reranked: bool = False


# ── Multi-hop ─────────────────────────────────────────────────────────────────

class HopResult(BaseModel):
    hop_number: int
    source: Literal["web", "local", "news", "github"]
    sub_query: str
    content_snippets: list[str]
    gap_queries: list[str] = Field(default_factory=list)
    latency_ms: float


class HopChainOutput(BaseModel):
    original_query: str
    hops: list[HopResult]
    merged_context: str
    total_latency_ms: float


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
    source_type: Literal["web", "local", "news", "github"] = "web"
    chunk_id: str | None = None
    source_file: str | None = None
    published_at: str | None = None
    repo: str | None = None


class AnswerResponse(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    citations: list[Citation]
    total_tokens: int
    latency_ms: float
    complexity: str = "MEDIUM"
    cache_hit: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    image_analysis: ImageAnalysis | None = None
    audio_transcript: str | None = None


# ── Research ──────────────────────────────────────────────────────────────────

class ResearchJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    progress: int = Field(0, ge=0, le=100)
    result: AnswerResponse | None = None
    error: str | None = None


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    run_id: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    passed: bool
    notes: str = ""


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── SSE Events ────────────────────────────────────────────────────────────────

class SSEEventType(str, Enum):
    PROGRESS = "progress"
    PLAN_DONE = "plan_done"
    ROUTE_DECIDED = "route_decided"
    HOP_RESULT = "hop_result"
    REASON_CHUNK = "reason_chunk"
    ANSWER_CHUNK = "answer_chunk"
    EVAL_DONE = "eval_done"
    TRACE_SUMMARY = "trace_summary"
    CACHE_HIT = "cache_hit"
    ERROR = "error"
    DONE = "done"


class SSEEvent(BaseModel):
    event: SSEEventType
    run_id: str
    data: dict[str, Any]
    token_delta: int = 0
    latency_ms: float = 0.0


# ── LangGraph state ───────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    query: str = ""
    complexity: QueryComplexity = QueryComplexity.MEDIUM
    memory_context: str = ""
    image_analysis: ImageAnalysis | None = None
    audio_transcript: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    route_decisions: list[RouteDecision] = Field(default_factory=list)
    search_outputs: list[SearchOutput] = Field(default_factory=list)
    news_outputs: list[NewsOutput] = Field(default_factory=list)
    github_outputs: list[GitHubOutput] = Field(default_factory=list)
    rag_outputs: list[RAGOutput] = Field(default_factory=list)
    hop_outputs: list[HopChainOutput] = Field(default_factory=list)
    reason_outputs: list[ReasonOutput] = Field(default_factory=list)
    answer: AnswerResponse | None = None
    eval_result: EvalResult | None = None
    trace: PipelineTrace | None = None
    sse_queue: list[SSEEvent] = Field(default_factory=list)
    error: str | None = None
    cache_hit: bool = False


class MemoryTurn(BaseModel):
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    run_id: str
    detail: str
    code: str
