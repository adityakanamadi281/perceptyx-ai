"""
agents/rag_agent.py
-------------------
Local RAG retrieval agent using LangChain's RetrievalQA chain.
Retrieves relevant chunks from Qdrant and returns structured RAGOutput.
"""

from __future__ import annotations

import time

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from config.settings import settings
from core.observability import TelemetryCallback, get_logger
from models.schemas import PipelineTrace, RAGChunk, RAGOutput
from providers.gemini import get_gemini_llm
from rag.vectorstore import similarity_search


async def run_rag_agent(
    sub_query: str,
    trace: PipelineTrace,
) -> RAGOutput:
    """
    Retrieve relevant chunks from the local knowledge base.
    Uses MMR (Maximum Marginal Relevance) for diverse retrieval.
    """
    logger = get_logger("rag_agent", trace.run_id)
    t0 = time.perf_counter()
    callback = TelemetryCallback("rag_agent", trace)

    logger.info("rag_retrieve_start", sub_query=sub_query)

    # Get docs with scores for ranking
    try:
        docs_scores: list[tuple[Document, float]] = await similarity_search(
            "main_knowledge", sub_query, k=settings.rag_top_k
        )
    except Exception as exc:
        logger.warning("rag_similarity_failed", error=str(exc))
        docs_scores = []

    chunks: list[RAGChunk] = []
    for doc, score in docs_scores:
        if score >= settings.rag_score_threshold:
            chunks.append(
                RAGChunk(
                    content=doc.page_content,
                    source_file=doc.metadata.get("source_file", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4),
                    chunk_id=doc.metadata.get("chunk_id", ""),
                )
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("rag_done", chunks=len(chunks), latency_ms=round(latency_ms, 1))

    return RAGOutput(
        sub_query=sub_query,
        chunks=chunks,
        latency_ms=latency_ms,
    )
