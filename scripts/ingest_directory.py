#!/usr/bin/env python
"""
scripts/ingest_directory.py
--------------------------
Ingests all supported documents (PDF, Markdown, plain text) from a local folder 
into the Qdrant vector store.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to python path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from rag.ingester import ingest_directory
from config.settings import settings
from rag.vectorstore import ensure_collections


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest local documents (PDF, MD, TXT) into Qdrant."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/raw_docs",
        help="Path to the directory containing documents (default: ./data/raw_docs)",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir.resolve()}")
        print("Please create the folder or provide a valid path via --dir option.")
        sys.exit(1)

    print(f"Starting ingestion from directory: {data_dir.resolve()}")
    print(f"Target Qdrant URL: {settings.qdrant_url}")
    print(f"Qdrant Collection: main_knowledge")
    print(f"Embedding Model: {settings.embedding_model}")
    print("Ingesting... (this may take a moment to download/load the embedding model)")

    try:
        await ensure_collections()
        results = await ingest_directory(data_dir)
        if not results:
            print("No supported files (.pdf, .md, .txt) were found or ingested.")
        else:
            print("\nIngestion completed successfully:")
            for filename, chunks in results.items():
                print(f" - {filename}: {chunks} chunks indexed")
            print(f"\nTotal files indexed: {len(results)}")
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
