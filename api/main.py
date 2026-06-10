"""
api/main.py - v2
────────────────
FastAPI app factory with:
  - Lifespan startup (DB init, Redis ping, OTel)
  - /api/v1 routes
  - /health with dependency checks
  - Prometheus /metrics (optional)
  - Static SPA
"""
from __future__ import annotations

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.query import router as query_router
from config.settings import settings
from core.observability import init_otel, get_logger

log = structlog.get_logger()
STATIC_DIR = Path(__file__).parent.parent / "static"


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

    Path("./data").mkdir(exist_ok=True)

    # Pre-warm reranker model
    if settings.enable_reranking:
        try:
            from rag.reranker import _get_reranker
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _get_reranker)
            log.info("reranker_prewarm_scheduled")
        except Exception as exc:
            log.warning("reranker_prewarm_failed", error=str(exc))

    log.info("app_startup", host=settings.app_host, port=settings.app_port)
    yield
    log.info("app_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PerceptyxAI v2",
        description=(
            "Perplexity-class AI search engine: parallel agents, hybrid retrieval, "
            "cross-encoder reranking, Redis cache, ARQ workers, Cloudflare AI fallback."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes ────────────────────────────────────────────────────────────
    app.include_router(query_router, prefix="/api/v1")

    # ── Ingest endpoint ───────────────────────────────────────────────────────
    @app.post("/api/v1/ingest", tags=["knowledge base"])
    async def ingest_document(
        file: __import__("fastapi").UploadFile = __import__("fastapi").File(...),
        background: bool = False,
    ):
        """Upload a PDF, Markdown, or text file to the local knowledge base."""
        from pathlib import Path as P
        import tempfile
        suffix = P(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".md", ".txt"}:
            from fastapi import HTTPException
            raise HTTPException(400, f"Unsupported file type: {suffix}")
        tmp = P(tempfile.mktemp(suffix=suffix))
        tmp.write_bytes(await file.read())
        if background:
            # Offload to ARQ worker
            try:
                from arq import create_pool
                from core.cache import get_arq_redis_settings
                pool = await create_pool(get_arq_redis_settings())
                await pool.enqueue_job("workers.embedding_worker.embed_documents", file_path=str(tmp))
                return {"file": file.filename, "status": "queued"}
            except Exception:
                pass
        try:
            from rag.ingester import ingest_file
            n_chunks = await ingest_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return {"file": file.filename, "chunks_indexed": n_chunks}

    # ── Memory endpoints ──────────────────────────────────────────────────────
    @app.delete("/api/v1/memory/{session_id}", tags=["memory"])
    async def clear_memory(session_id: str):
        from memory.store import clear_session
        await clear_session(session_id)
        return {"session_id": session_id, "cleared": True}

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["infra"])
    async def health():
        status = {"version": "2.0.0", "redis": "unknown", "postgres": "unknown"}
        try:
            from core.cache import get_redis
            await get_redis().ping()
            status["redis"] = "ok"
        except Exception as exc:
            status["redis"] = f"error: {exc}"
        try:
            from db.engine import engine
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy", fromlist=["text"]).text("SELECT 1"))
            status["postgres"] = "ok"
        except Exception as exc:
            status["postgres"] = f"error: {exc}"

        all_ok = all(v == "ok" for v in [status["redis"], status["postgres"]])
        return JSONResponse(
            content={"status": "ok" if all_ok else "degraded", **status},
            status_code=200,
        )

    # ── Prometheus metrics passthrough ────────────────────────────────────────
    @app.get("/metrics", tags=["infra"], include_in_schema=False)
    async def metrics():
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from fastapi.responses import Response
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            return JSONResponse({"error": "prometheus_client not installed"})

    # ── Static SPA ────────────────────────────────────────────────────────────
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_root():
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=settings.app_workers,
        log_config=None,
    )
