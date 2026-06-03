"""
api/main.py
-----------
FastAPI application factory with:
  - CORS middleware
  - Structured logging on startup
  - OpenTelemetry bootstrap
  - Health-check endpoint
  - Pipeline router
"""

from __future__ import annotations

import os
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.query import router as query_router
from config.settings import settings
from core.observability import init_otel

log = structlog.get_logger()


def create_app() -> FastAPI:
    # Bootstrap observability before the app starts handling requests
    init_otel()

    app = FastAPI(
        title="Perplexity Agent",
        description=(
            "Agentic web search, reasoning, and fact-checking system. "
            "Powered by LangGraph + Gemini."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(query_router, prefix="/api/v1")

    # ── UI Route ──────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse, tags=["ui"])
    async def serve_ui() -> HTMLResponse:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
        if not os.path.exists(file_path):
            file_path = "static/index.html"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content)
        except Exception as exc:
            return HTMLResponse(
                content=f"<h1>UI file not found</h1><p>{str(exc)}</p>", status_code=404
            )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["infra"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def on_startup() -> None:
        log.info(
            "app_startup",
            host=settings.app_host,
            port=settings.app_port,
            gemini_model=settings.gemini_model,
            log_level=settings.log_level,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=settings.app_workers,
        log_config=None,  # let structlog handle all logging
    )
