"""
main.py
-------
Root-level FastAPI app entrypoint.
Mounts feedback router, exposes health and query endpoints.
"""

from __future__ import annotations

import asyncio

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import uuid

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from core.feedback_routes import router as feedback_router
from core.observability import init_otel
from core.orchestrator import run_pipeline, stream_pipeline
from models.schemas import AnswerResponse, ErrorResponse, QueryRequest

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    init_otel()

    # DB initialisation
    try:
        from db.engine import init_db

        await init_db()
        log.info("db_ready")
    except Exception as exc:
        log.warning("db_init_failed", error=str(exc))

    # Redis ping
    try:
        from core.cache import get_redis

        r = get_redis()
        await r.ping()
        log.info("redis_ready")
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))

    # Qdrant verify/init collections
    try:
        from rag.vectorstore import ensure_collections

        await ensure_collections()
        log.info("qdrant_collections_verified")
    except Exception as exc:
        log.warning("qdrant_collections_init_failed", error=str(exc))

    Path("./data").mkdir(exist_ok=True)
    yield
    log.info("app_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PerceptyxAI",
        description="Perplexity-class AI search engine with Qdrant and RLHF self-learning.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Feedback & RLHF router ───────────────────────────────────────────────
    app.include_router(feedback_router)
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Health Endpoints ──────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse("static/index.html")

    @app.get("/healthz", tags=["infra"])
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", tags=["infra"])
    async def readyz():
        # Quick checks on redis, pg, and qdrant
        status = {"redis": "unknown", "postgres": "unknown", "qdrant": "unknown"}
        try:
            from core.cache import get_redis

            await get_redis().ping()
            status["redis"] = "ok"
        except Exception as exc:
            status["redis"] = f"error: {exc}"

        try:
            from db.engine import engine

            async with engine.connect() as conn:
                from sqlalchemy import text

                await conn.execute(text("SELECT 1"))
            status["postgres"] = "ok"
        except Exception as exc:
            status["postgres"] = f"error: {exc}"

        try:
            from rag.vectorstore import get_qdrant_client

            await get_qdrant_client().get_collections()
            status["qdrant"] = "ok"
        except Exception as exc:
            status["qdrant"] = f"error: {exc}"

        all_ok = all(v == "ok" for v in status.values())
        return JSONResponse(
            content={"status": "ok" if all_ok else "degraded", **status},
            status_code=200 if all_ok else 503,
        )

    # ── Query Pipeline Endpoints ──────────────────────────────────────────────

    @app.post("/api/query", response_model=AnswerResponse, tags=["pipeline"])
    async def query_endpoint(body: QueryRequest) -> AnswerResponse:
        run_id = str(uuid.uuid4())
        log.info("api_request", run_id=run_id, query=body.query[:80])
        try:
            return await run_pipeline(body)
        except TimeoutError as exc:
            raise HTTPException(
                504, ErrorResponse(run_id=run_id, detail="Timeout", code="TIMEOUT").model_dump()
            ) from exc
        except Exception as exc:
            log.error("api_error", run_id=run_id, error=str(exc))
            raise HTTPException(
                500,
                ErrorResponse(run_id=run_id, detail=str(exc), code="PIPELINE_ERROR").model_dump(),
            ) from exc

    @app.post("/api/query/multimodal", response_model=AnswerResponse, tags=["pipeline"])
    async def query_multimodal_endpoint(
        query: str = Form(...),
        max_sources: int = Form(5),
        locale: str = Form("en"),
        session_id: str | None = Form(None),
        force_research: bool = Form(False),
        image: UploadFile | None = File(None),
        audio: UploadFile | None = File(None),
    ) -> AnswerResponse:
        image_analysis = None
        audio_transcript = None
        enriched_query = query

        if image and image.filename:
            from tools.multimodal import analyse_image

            image_analysis = await analyse_image(
                await image.read(),
                image.content_type or "image/jpeg",
            )
            enriched_query += f"\n\nImage context: {image_analysis.description}"
            if image_analysis.extracted_text:
                enriched_query += f"\nVisible text: {image_analysis.extracted_text}"

        if audio and audio.filename:
            from tools.multimodal import transcribe_audio

            transcription = await transcribe_audio(await audio.read(), audio.filename)
            audio_transcript = transcription.transcript
            if audio_transcript:
                enriched_query += f"\n\nAudio transcript: {audio_transcript}"

        response = await run_pipeline(
            QueryRequest(
                query=enriched_query,
                max_sources=max_sources,
                locale=locale,
                session_id=session_id,
                force_research=force_research,
            )
        )
        response.query = query
        response.image_analysis = image_analysis
        response.audio_transcript = audio_transcript
        return response

    @app.post("/api/query/stream", tags=["pipeline"], response_class=EventSourceResponse)
    async def query_stream_endpoint(body: QueryRequest):
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

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=settings.app_workers,
        log_config=None,
    )
