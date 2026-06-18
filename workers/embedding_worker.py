"""
workers/embedding_worker.py
───────────────────────────
ARQ worker: chunk documents, generate embeddings, update Qdrant.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger()


async def embed_documents(ctx: dict, file_path: str, collection: str | None = None) -> dict:
    """ARQ job: embed_documents(file_path) → {chunks_indexed}"""
    from pathlib import Path
    from rag.ingester import ingest_file
    path = Path(file_path)
    n = await ingest_file(path)
    log.info("embed_done", file=file_path, chunks=n)
    return {"file": file_path, "chunks_indexed": n}
