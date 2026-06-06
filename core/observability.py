"""
core/observability.py - v2
──────────────────────────
Structured logging + OpenTelemetry + Prometheus metrics.
"""
from __future__ import annotations

import os
import time
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from config.settings import settings
from models.schemas import AgentSpan, PipelineTrace

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    REQUEST_LATENCY = Histogram(
        "perplexity_request_latency_seconds",
        "End-to-end request latency",
        ["complexity"],
        buckets=[0.5, 1, 2, 3, 5, 8, 15, 30, 60],
    )
    CACHE_HITS = Counter("perplexity_cache_hits_total", "Cache hits", ["cache_type"])
    CACHE_MISSES = Counter("perplexity_cache_misses_total", "Cache misses", ["cache_type"])
    LLM_TOKENS = Counter("perplexity_llm_tokens_total", "LLM tokens used", ["provider", "agent"])
    LLM_LATENCY = Histogram(
        "perplexity_llm_latency_seconds", "LLM call latency", ["agent"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
    )
    RETRIEVAL_LATENCY = Histogram(
        "perplexity_retrieval_latency_seconds", "Retrieval latency", ["source"],
    )
    RERANK_LATENCY = Histogram("perplexity_rerank_latency_seconds", "Reranking latency")
    WORKER_QUEUE = Gauge("perplexity_worker_queue_length", "ARQ worker queue length")
    WORKER_FAILURES = Counter("perplexity_worker_failures_total", "ARQ worker failures", ["job_type"])
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def init_otel() -> None:
    if settings.enable_otel:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            resource = Resource({"service.name": "perplexity-agent-v2"})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
            )
            trace.set_tracer_provider(provider)
        except Exception as e:
            get_logger("otel").warning("otel_init_failed", error=str(e))

    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), settings.log_level, 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if PROMETHEUS_AVAILABLE and settings.enable_prometheus:
        try:
            start_http_server(settings.prometheus_port)
        except Exception:
            pass  # already running or port taken


def get_logger(name: str, run_id: str = "") -> Any:
    logger = structlog.get_logger(name)
    if run_id:
        logger = logger.bind(run_id=run_id)
    return logger


def record_latency(complexity: str, latency_s: float) -> None:
    if PROMETHEUS_AVAILABLE:
        REQUEST_LATENCY.labels(complexity=complexity).observe(latency_s)


def record_cache_hit(cache_type: str) -> None:
    if PROMETHEUS_AVAILABLE:
        CACHE_HITS.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    if PROMETHEUS_AVAILABLE:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_llm_tokens(provider: str, agent: str, tokens: int) -> None:
    if PROMETHEUS_AVAILABLE:
        LLM_TOKENS.labels(provider=provider, agent=agent).inc(tokens)


class TelemetryCallback(BaseCallbackHandler):
    def __init__(self, agent_name: str, trace: PipelineTrace):
        super().__init__()
        self.agent_name = agent_name
        self.trace = trace
        self._t0 = time.perf_counter()

    def on_llm_start(self, *args, **kwargs) -> None:
        self._t0 = time.perf_counter()

    def on_llm_end(self, response, **kwargs) -> None:
        latency_ms = (time.perf_counter() - self._t0) * 1000
        tokens = 0
        try:
            usage = response.llm_output.get("usage_metadata", {}) if response.llm_output else {}
            tokens = usage.get("total_token_count", 0) or usage.get("total_tokens", 0)
        except Exception:
            pass

        self.trace.total_tokens += tokens
        self.trace.spans.append(AgentSpan(
            agent=self.agent_name,
            run_id=self.trace.run_id,
            latency_ms=latency_ms,
            tokens_used=tokens,
        ))
        if PROMETHEUS_AVAILABLE:
            LLM_LATENCY.labels(agent=self.agent_name).observe(latency_ms / 1000)
            LLM_TOKENS.labels(provider="gemini", agent=self.agent_name).inc(tokens)

    def on_llm_error(self, error, **kwargs) -> None:
        get_logger(self.agent_name).error("llm_error", error=str(error))
