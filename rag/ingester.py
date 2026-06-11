"""
rag/ingester.py
---------------
Ingest local documents (PDF, Markdown, plain text) into ChromaDB
via LangChain's document loaders and text splitter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_vectorstore

log = structlog.get_logger()

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
}


def _chunk_id(doc: Document) -> str:
    """Stable ID from source + page + content hash."""
    raw = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', 0)}:{doc.page_content[:120]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def ingest_file(path: str | Path) -> int:
    """
    Ingest a single file into the vector store.
    Returns the number of chunks added.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    if loader_cls is TextLoader:
        loader = TextLoader(str(p), encoding="utf-8")
    else:
        loader = loader_cls(str(p))
    raw_docs: list[Document] = loader.load()
    chunks = _SPLITTER.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = _chunk_id(chunk)
        chunk.metadata["source_file"] = p.name

    vs = get_vectorstore()
    vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    log.info("ingested", file=p.name, chunks=len(chunks))
    return len(chunks)


async def ingest_directory(directory: str | Path, glob: str = "**/*") -> dict[str, int]:
    """
    Ingest all supported files from a directory.
    Returns a dict of {filename: chunks_added}.
    """
    p = Path(directory)
    results: dict[str, int] = {}
    for ext in _LOADER_MAP:
        for fp in p.glob(f"**/*{ext}"):
            try:
                n = await ingest_file(fp)
                results[fp.name] = n
            except Exception as exc:
                log.warning("ingest_skip", file=str(fp), error=str(exc))
    return results
