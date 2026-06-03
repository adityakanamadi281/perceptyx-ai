"""
api/query.py
------------
FastAPI router for POST /query.
Handles request validation, pipeline invocation, and error mapping.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from core.orchestrator import run_pipeline
from models.schemas import AnswerResponse, ErrorResponse, QueryRequest

router = APIRouter()
log = structlog.get_logger()


@router.post(
    "/query",
    response_model=AnswerResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
        504: {"model": ErrorResponse, "description": "Pipeline timeout"},
    },
    summary="Run the agentic search + reasoning pipeline",
    tags=["pipeline"],
)
async def query_endpoint(request: Request, body: QueryRequest) -> AnswerResponse:
    """
    Execute the full query → sub-queries → search → reason → answer pipeline.

    - Decomposes the query into parallel sub-queries
    - Searches the web via Serper and scrapes results
    - Reasons over scraped content with Gemini chain-of-thought
    - Synthesises a cited final answer

    **Typical latency**: 15–45 s depending on network and model response time.
    """
    run_id = str(uuid.uuid4())
    log.info("api_request", run_id=run_id, query=body.query, client=request.client)

    try:
        answer = await run_pipeline(body)
        return answer
    except TimeoutError as exc:
        log.error("api_timeout", run_id=run_id)
        raise HTTPException(
            status_code=504,
            detail=ErrorResponse(
                run_id=run_id, detail="Pipeline timed out", code="TIMEOUT"
            ).model_dump(),
        ) from exc
    except Exception as exc:
        log.error("api_error", run_id=run_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                run_id=run_id, detail=str(exc), code="PIPELINE_ERROR"
            ).model_dump(),
        ) from exc
