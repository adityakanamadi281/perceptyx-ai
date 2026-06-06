"""
workers/research_worker.py
──────────────────────────
ARQ worker: deep research via multi-hop retrieval, gap detection,
iterative search, synthesis, and evaluation.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import structlog

from config.settings import settings
from core.cache import set_job_status, publish_event

log = structlog.get_logger()


async def deep_research(ctx: dict, job_id: str, query: str, session_id: str | None = None) -> dict:
    """
    ARQ job: deep_research(job_id, query)
    Multi-phase:
      1. Generate sub-queries
      2. Parallel search
      3. Crawl top URLs
      4. Gap detection + additional searches
      5. Synthesis + Evaluation
    """
    from core.orchestrator import run_pipeline
    from models.schemas import QueryRequest, QueryComplexity

    log.info("research_start", job_id=job_id, query=query[:80])
    await set_job_status(job_id, "running", progress=0)
    await publish_event("research_events", {"job_id": job_id, "status": "running"})

    try:
        # Phase 1: Initial planning
        await set_job_status(job_id, "running", progress=10)

        request = QueryRequest(
            query=query,
            session_id=session_id,
            force_research=True,
        )

        # Phase 2-4: Run full pipeline with research complexity
        await set_job_status(job_id, "running", progress=30)
        await publish_event("research_events", {"job_id": job_id, "progress": 30, "step": "searching"})

        answer = await asyncio.wait_for(
            run_pipeline(request),
            timeout=settings.research_timeout_s,
        )

        await set_job_status(job_id, "done", progress=100, result=answer.model_dump())
        await publish_event("research_events", {"job_id": job_id, "status": "done", "progress": 100})
        log.info("research_done", job_id=job_id, tokens=answer.total_tokens)
        return answer.model_dump()

    except Exception as exc:
        log.error("research_failed", job_id=job_id, error=str(exc))
        await set_job_status(job_id, "failed", progress=0)
        await publish_event("research_events", {"job_id": job_id, "status": "failed", "error": str(exc)})
        raise
