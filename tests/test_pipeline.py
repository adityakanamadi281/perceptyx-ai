"""
tests/test_pipeline.py
-----------------------
Integration tests for the search → reason → answer pipeline.
Uses pytest-httpx and pytest-mock to stub external calls so tests
run without real API keys.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api.main import app
from models.schemas import (
    AnswerResponse,
    Citation,
    PipelineTrace,
    QueryRequest,
    ReasonOutput,
    ReasoningStep,
    SearchOutput,
    SearchResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_query() -> QueryRequest:
    return QueryRequest(query="What is retrieval-augmented generation?")


@pytest.fixture
def sample_search_output() -> SearchOutput:
    return SearchOutput(
        sub_query="retrieval-augmented generation overview",
        results=[
            SearchResult(
                title="RAG Explained",
                url="https://example.com/rag",
                snippet="RAG combines retrieval with generation.",
                scraped_text="RAG is a technique that enhances LLMs with external knowledge.",
            )
        ],
        latency_ms=120.0,
    )


@pytest.fixture
def sample_reason_output() -> ReasonOutput:
    return ReasonOutput(
        sub_query="retrieval-augmented generation overview",
        steps=[
            ReasoningStep(
                thought="The source describes RAG as combining retrieval with generation.",
                conclusion="RAG retrieves relevant documents before generating an answer.",
            )
        ],
        summary="RAG enhances LLMs by retrieving relevant documents at inference time.",
        supporting_urls=["https://example.com/rag"],
        tokens_used=350,
        latency_ms=900.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_serper_search_parses_response():
    """Serper tool correctly parses the organic results array."""
    mock_response = {
        "organic": [
            {"title": "Test", "link": "https://example.com", "snippet": "Test snippet"},
        ]
    }

    with patch("tools.serper.httpx.AsyncClient") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=MagicMock(post=AsyncMock(return_value=mock_resp))
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from tools.serper import serper_search

        results = await serper_search("test query", num=1)

    assert len(results) == 1
    assert results[0].title == "Test"
    assert results[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_planner_returns_sub_queries():
    """Planner splits a query into multiple sub-queries via the LLM."""
    mock_llm_response = MagicMock()
    mock_llm_response.content = json.dumps(
        ["What is RAG?", "How does RAG retrieval work?", "RAG use cases"]
    )

    trace = PipelineTrace(run_id="test-run", query="What is RAG?")

    with patch("core.planner.get_gemini_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
        mock_get_llm.return_value = mock_llm

        from core.planner import plan_sub_queries

        sub_queries = await plan_sub_queries("What is RAG?", trace, n=3)

    assert isinstance(sub_queries, list)
    assert len(sub_queries) == 3
    assert all(isinstance(q, str) for q in sub_queries)


@pytest.mark.asyncio
async def test_search_agent_enriches_results(sample_search_output):
    """Search agent attaches scraped text to search results."""
    from datetime import datetime, timezone

    with (
        patch("agents.search.serper_search", return_value=sample_search_output.results),
        patch(
            "agents.search.scrape_urls",
            return_value={
                "https://example.com/rag": (
                    "Full article text about RAG.",
                    datetime.now(timezone.utc),
                )
            },
        ),
    ):
        trace = PipelineTrace(run_id="test-run", query="RAG overview")
        from agents.search import run_search_agent

        output = await run_search_agent("retrieval-augmented generation overview", trace)

    assert output.sub_query == "retrieval-augmented generation overview"
    assert len(output.results) == 1
    assert output.results[0].scraped_text == "Full article text about RAG."
    assert output.latency_ms > 0


@pytest.mark.asyncio
async def test_reason_agent_returns_structured_output(sample_search_output):
    """Reason agent parses chain-of-thought JSON from the LLM."""
    mock_cot = {
        "steps": [
            {"thought": "Source confirms RAG uses retrieval.", "conclusion": "RAG retrieves docs."}
        ],
        "summary": "RAG is an LLM enhancement technique.",
        "supporting_urls": ["https://example.com/rag"],
    }
    mock_response = MagicMock()
    mock_response.content = json.dumps(mock_cot)

    trace = PipelineTrace(run_id="test-run", query="RAG overview")

    with patch("agents.reason.get_gemini_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        from agents.reason import run_reason_agent

        output = await run_reason_agent(sample_search_output, trace)

    assert output.sub_query == sample_search_output.sub_query
    assert "RAG" in output.summary
    assert len(output.steps) == 1


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(sample_query):
    """Happy-path: mocked Serper + Gemini, real graph execution."""
    from datetime import datetime, timezone

    fake_search_results = [
        SearchResult(
            title="RAG Guide",
            url="https://example.com/rag-guide",
            snippet="RAG combines retrieval and generation.",
            scraped_text="RAG is a method for grounding LLM outputs in retrieved documents.",
        )
    ]

    fake_sub_queries_response = MagicMock()
    fake_sub_queries_response.content = json.dumps(["What is RAG?"])

    fake_cot_response = MagicMock()
    fake_cot_response.content = json.dumps(
        {
            "steps": [{"thought": "Source confirms RAG.", "conclusion": "RAG retrieves docs."}],
            "summary": "RAG retrieves relevant documents to ground LLM responses.",
            "supporting_urls": ["https://example.com/rag-guide"],
        }
    )

    fake_answer_response = MagicMock()
    fake_answer_response.content = (
        "<answer>RAG is a technique that grounds language model responses in retrieved documents.</answer>\n"
        '<citations>[{"index": 1, "title": "RAG Guide", "url": "https://example.com/rag-guide", '
        '"relevant_snippet": "RAG combines retrieval and generation."}]</citations>'
    )

    call_count = 0

    async def _mock_invoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return fake_sub_queries_response
        elif call_count == 2:
            return fake_cot_response
        else:
            return fake_answer_response

    with (
        patch("core.router.classify_query", return_value=True),
        patch("tools.serper.httpx.AsyncClient") as mock_httpx,
        patch(
            "tools.scraper._fetch_with_httpx", return_value="<html><body>RAG article</body></html>"
        ),
        patch("core.planner.get_gemini_llm") as mock_planner_llm,
        patch("agents.reason.get_gemini_llm") as mock_reason_llm,
        patch("agents.answer.get_gemini_llm") as mock_answer_llm,
    ):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "organic": [
                {
                    "title": "RAG Guide",
                    "link": "https://example.com/rag-guide",
                    "snippet": "RAG combines retrieval and generation.",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.return_value.__aenter__ = AsyncMock(
            return_value=MagicMock(post=AsyncMock(return_value=mock_resp))
        )
        mock_httpx.return_value.__aexit__ = AsyncMock(return_value=False)

        for mock_llm_getter in [mock_planner_llm, mock_reason_llm, mock_answer_llm]:
            mock_llm_inst = MagicMock()
            mock_llm_inst.ainvoke = _mock_invoke
            mock_llm_getter.return_value = mock_llm_inst

        from core.orchestrator import run_pipeline

        answer = await run_pipeline(sample_query)

    assert isinstance(answer, AnswerResponse)
    assert answer.query == sample_query.query
    assert len(answer.answer) > 10
    assert isinstance(answer.citations, list)


@pytest.mark.asyncio
async def test_query_router_classification():
    """Router correctly classifies query based on LLM JSON response."""
    mock_response_true = MagicMock()
    mock_response_true.content = json.dumps(
        {"requires_search": True, "reason": "Needs real-time fact checking"}
    )

    mock_response_false = MagicMock()
    mock_response_false.content = json.dumps(
        {"requires_search": False, "reason": "General programming question"}
    )

    trace = PipelineTrace(run_id="test-run", query="What is python?")

    with patch("core.router.get_gemini_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[mock_response_true, mock_response_false])
        mock_get_llm.return_value = mock_llm

        from core.router import classify_query

        res_true = await classify_query("Latest news about space", trace)
        res_false = await classify_query("Write a quicksort", trace)

    assert res_true is True
    assert res_false is False


@pytest.mark.asyncio
async def test_pipeline_direct_answer_routing():
    """If search is not required, graph runs direct answer and skips search/reason agents."""
    from models.schemas import QueryRequest

    query_req = QueryRequest(query="Write a quicksort algorithm in python")

    mock_router_response = MagicMock()
    mock_router_response.content = json.dumps(
        {"requires_search": False, "reason": "Standard coding query"}
    )

    mock_answer_response = MagicMock()
    mock_answer_response.content = "Here is the quicksort algorithm..."

    async def _mock_invoke(messages, **kwargs):
        system_content = messages[0].content
        if "query router" in system_content:
            return mock_router_response
        else:
            return mock_answer_response

    with (
        patch("core.router.get_gemini_llm") as mock_router_llm,
        patch("agents.answer.get_gemini_llm") as mock_answer_llm,
    ):
        mock_llm_inst = MagicMock()
        mock_llm_inst.ainvoke = _mock_invoke
        mock_router_llm.return_value = mock_llm_inst
        mock_answer_llm.return_value = mock_llm_inst

        from core.orchestrator import run_pipeline

        answer = await run_pipeline(query_req)

    assert isinstance(answer, AnswerResponse)
    assert answer.answer == "Here is the quicksort algorithm..."
    assert len(answer.citations) == 0
