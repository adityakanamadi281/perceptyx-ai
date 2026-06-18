"""
rag/vectorstore.py
------------------
Singleton Qdrant vector store using HuggingFace embeddings.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qd_models

from config.settings import settings

log = structlog.get_logger()

COLLECTIONS = [
    "main_knowledge",
    "feedback_memory",
    "episodic_memory",
    "procedural_memory",
    "semantic_knowledge",
]

_client: AsyncQdrantClient | None = None


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        api_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key and settings.qdrant_api_key.get_secret_value()
            else None
        )
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=api_key,
        )
    return _client


async def get_embedding_dim() -> int:
    embeddings = get_embeddings()
    dummy = await embeddings.aembed_query("dummy")
    return len(dummy)


async def ensure_collections() -> None:
    client = get_qdrant_client()
    dim = await get_embedding_dim()
    
    existing_names = set()
    max_retries = 5
    retry_delay = 2.0
    
    for attempt in range(1, max_retries + 1):
        try:
            existing = await client.get_collections()
            existing_names = {c.name for c in existing.collections}
            break
        except Exception as e:
            if attempt == max_retries:
                log.error("qdrant_connection_failed_permanently", error=str(e))
                raise e
            log.warning(
                "qdrant_connection_attempt_failed",
                attempt=attempt,
                max_retries=max_retries,
                retry_delay=retry_delay,
                error=str(e)
            )
            await asyncio.sleep(retry_delay)
            retry_delay *= 1.5

    for name in COLLECTIONS:
        if name not in existing_names:
            try:
                await client.create_collection(
                    collection_name=name,
                    vectors_config=qd_models.VectorParams(
                        size=dim,
                        distance=qd_models.Distance.COSINE,
                    ),
                )
                log.info("qdrant_collection_created", name=name)
            except Exception as e:
                log.warning("qdrant_collection_creation_failed", name=name, error=str(e))


async def embed_text_async(text: str) -> list[float]:
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


async def upsert_document(
    collection_name: str,
    text: str,
    metadata: dict[str, Any],
    id: str,
) -> None:
    client = get_qdrant_client()
    vector = await embed_text_async(text)
    
    # Store the actual text in the payload as 'text'
    payload = {"text": text, **metadata}
    
    await client.upsert(
        collection_name=collection_name,
        points=[
            qd_models.PointStruct(
                id=id,
                vector=vector,
                payload=payload,
            )
        ],
    )


async def similarity_search(
    collection_name: str,
    query: str,
    k: int = 5,
    filter: Any = None,
) -> list[tuple[Document, float]]:
    client = get_qdrant_client()
    query_vector = await embed_text_async(query)

    try:
        response = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            query_filter=filter,
        )
        results = response.points
    except Exception as e:
        log.warning("similarity_search_failed", collection=collection_name, error=str(e))
        return []

    docs_with_scores = []
    for res in results:
        payload = dict(res.payload) if res.payload else {}
        text = payload.pop("text", "")
        doc = Document(page_content=text, metadata=payload)
        docs_with_scores.append((doc, res.score))

    return docs_with_scores


async def delete_by_filter(collection_name: str, filter: Any) -> None:
    client = get_qdrant_client()
    await client.delete(
        collection_name=collection_name,
        points_selector=filter,
    )
