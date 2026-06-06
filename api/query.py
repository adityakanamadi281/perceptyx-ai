"""
api/query.py - v2
──────────────────
Endpoints:
  POST /query           — blocking JSON
  POST /query/stream    — SSE stream with progress events
  POST /research        — async deep research (ARQ job)
  GET  /research/{id}   — poll research job status
  POST /query/multimodal — multipart form with image/audio
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from core.orchestrator import run_pipeline, stream_pipeline
from models.schemas import (
    AnswerResponse, ErrorResponse, QueryRequest,
    ResearchJobStatus, ResearchRequest, SSEEvent,
)

router = APIRouter()
log = structlog.get_logger()

_AUDIO_MIME = {"audio/wav","audio/mpeg","audio/mp4","audio/ogg","audio/webm","audio/flac","application/octet-stream"}
_IMAGE_MIME = {"image/jpeg","image/png","image/webp","image/gif"}


def _rid() -> str:
    return str(uuid.uuid4())


# ── Blocking query ─────────────────────────────────────────────────────────────

@router.post("/query", response_model=AnswerResponse, tags=["pipeline"])
async def query_endpoint(request: Request, body: QueryRequest) -> AnswerResponse:
    rid = _rid()
    log.info("api_request", run_id=rid, query=body.query[:80])
    try:
        return await run_pipeline(body)
    except TimeoutError as exc:
        raise HTTPException(504, ErrorResponse(run_id=rid, detail="Timeout", code="TIMEOUT").model_dump()) from exc
    except Exception as exc:
        log.error("api_error", run_id=rid, error=str(exc))
        raise HTTPException(500, ErrorResponse(run_id=rid, detail=str(exc), code="PIPELINE_ERROR").model_dump()) from exc


# ── SSE streaming ─────────────────────────────────────────────────────────────

@router.post("/query/stream", tags=["pipeline"], response_class=EventSourceResponse)
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """
    SSE stream. Emits progress steps immediately:
    classified → plan_done → route_decided → hop_result → reason_chunk
    → answer_chunk → eval_done → trace_summary → done
    """
    log.info("sse_request", query=body.query[:80])

    async def _generate():
        async for event in stream_pipeline(body):
            payload = json.dumps({
                "run_id": event.run_id,
                "token_delta": event.token_delta,
                "latency_ms": event.latency_ms,
                **event.data,
            })
            yield {"event": event.event.value, "data": payload}

    return EventSourceResponse(_generate())


# ── Research (async ARQ job) ──────────────────────────────────────────────────

@router.post("/research", response_model=ResearchJobStatus, tags=["research"])
async def start_research(body: ResearchRequest):
    """
    Start a deep research job. Returns job_id immediately.
    Poll GET /research/{job_id} for status.
    """
    job_id = _rid()
    log.info("research_start", job_id=job_id, query=body.query[:80])

    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        from config.settings import settings
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        await pool.enqueue_job(
            "workers.research_worker.deep_research",
            job_id=job_id,
            query=body.query,
            session_id=body.session_id,
            _job_id=job_id,
        )
        from core.cache import set_job_status
        await set_job_status(job_id, "pending", progress=0)
        return ResearchJobStatus(job_id=job_id, status="pending")
    except Exception as exc:
        # Fallback: run inline if ARQ/Redis unavailable
        log.warning("arq_unavailable_running_inline", error=str(exc))
        import asyncio
        asyncio.create_task(_inline_research(job_id, body.query, body.session_id))
        return ResearchJobStatus(job_id=job_id, status="pending")


async def _inline_research(job_id: str, query: str, session_id: str | None):
    """Fallback: run research synchronously in background task."""
    from workers.research_worker import deep_research
    try:
        await deep_research({}, job_id=job_id, query=query, session_id=session_id)
    except Exception as exc:
        log.error("inline_research_failed", job_id=job_id, error=str(exc))


@router.get("/research/{job_id}", response_model=ResearchJobStatus, tags=["research"])
async def poll_research(job_id: str):
    from core.cache import get_job_status
    status = await get_job_status(job_id)
    if status is None:
        raise HTTPException(404, f"Research job {job_id} not found")
    result = None
    if status.get("result"):
        try:
            result = AnswerResponse(**status["result"])
        except Exception:
            pass
    return ResearchJobStatus(
        job_id=job_id,
        status=status.get("status", "unknown"),
        progress=status.get("progress", 0),
        result=result,
        error=status.get("error"),
    )


# ── Multimodal ────────────────────────────────────────────────────────────────

@router.post("/query/multimodal", response_model=AnswerResponse, tags=["pipeline"])
async def query_multimodal_endpoint(
    request: Request,
    query: str = Form(..., min_length=1),
    session_id: Optional[str] = Form(None),
    max_sources: int = Form(5),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    from tools.multimodal import analyse_image, transcribe_audio
    rid = _rid()
    log.info("multimodal_request", run_id=rid, has_image=image is not None, has_audio=audio is not None)

    image_analysis = None
    audio_transcript = None

    if image and image.filename:
        if (image.content_type or "") not in _IMAGE_MIME:
            raise HTTPException(400, f"Unsupported image type: {image.content_type}")
        image_analysis = await analyse_image(await image.read(), image.content_type or "image/jpeg")

    if audio and audio.filename:
        transcription = await transcribe_audio(await audio.read(), audio.filename)
        audio_transcript = transcription.transcript

    enriched = query
    if image_analysis:
        enriched += f". Image shows: {image_analysis.description}"
        if image_analysis.extracted_text:
            enriched += f". Text in image: {image_analysis.extracted_text}"
    if audio_transcript:
        enriched += f". Audio says: {audio_transcript}"

    body = QueryRequest(
        query=enriched, max_sources=max_sources, session_id=session_id,
        image_context=image_analysis.description if image_analysis else None,
        audio_transcript=audio_transcript,
    )
    try:
        answer = await run_pipeline(body)
        answer.image_analysis = image_analysis
        answer.audio_transcript = audio_transcript
        return answer
    except Exception as exc:
        log.error("multimodal_error", run_id=rid, error=str(exc))
        raise HTTPException(500, ErrorResponse(run_id=rid, detail=str(exc), code="PIPELINE_ERROR").model_dump()) from exc
