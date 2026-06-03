"""
core/observability.py
---------------------
Centralised telemetry:
  - structlog JSON logger (run_id bound on every call)
  - Token-usage aggregator (plugs into LangChain callbacks)
  - Latency span context-manager
  - Optional OpenTelemetry export
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from config.settings import settings
from models.schemas import AgentSpan, PipelineTrace

# ── Logger setup ──────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


def get_logger(agent: str, run_id: str) -> Any:
    """Return a logger pre-bound with agent name and run_id."""
    return log.bind(agent=agent, run_id=run_id)


# ── Latency helpers ───────────────────────────────────────────────────────────


@contextmanager
def timed(label: str = ""):
    """Sync context-manager that yields elapsed ms after the block."""
    start = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000


@asynccontextmanager
async def async_timed(label: str = ""):
    """Async context-manager for async blocks."""
    start = time.perf_counter()
    result: dict[str, float] = {}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000


# ── LangChain callback for token + latency capture ────────────────────────────


class TelemetryCallback(AsyncCallbackHandler):
    """
    Plugs into every LangChain LLM call.
    Accumulates token usage per agent and records a span in the PipelineTrace.
    """

    def __init__(self, agent_name: str, trace: PipelineTrace) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.trace = trace
        self._t0: float = 0.0
        self._span_id: str = str(uuid.uuid4())

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._t0 = time.perf_counter()
        log.info(
            "llm_start",
            agent=self.agent_name,
            run_id=self.trace.run_id,
            model=serialized.get("id", ["unknown"])[-1],
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency_ms = (time.perf_counter() - self._t0) * 1000
        tokens = 0
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens = usage.get("total_tokens", 0)

        span = AgentSpan(
            agent=self.agent_name,
            run_id=self.trace.run_id,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
        self.trace.spans.append(span)
        self.trace.total_tokens += tokens
        self.trace.total_latency_ms += latency_ms

        log.info(
            "llm_end",
            agent=self.agent_name,
            run_id=self.trace.run_id,
            latency_ms=round(latency_ms, 1),
            tokens=tokens,
        )

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        log.error(
            "llm_error",
            agent=self.agent_name,
            run_id=self.trace.run_id,
            error=str(error),
        )


# ── OTEL bootstrap (optional) ─────────────────────────────────────────────────


def init_otel() -> None:
    """Initialise OpenTelemetry export if enabled in settings."""
    if not settings.enable_otel:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "perplexity-agent"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        log.info("otel_initialised", endpoint=settings.otel_endpoint)
    except ImportError:
        log.warning("otel_skipped", reason="opentelemetry packages not installed")
