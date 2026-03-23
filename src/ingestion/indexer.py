from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.ingestion.index_db import IngestionIndex
from src.ingestion.loaders import load_file_as_documents


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tag_docs(docs: List[Document], file_id: str, file_hash: str, collection: str) -> List[Document]:
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["file_id"] = file_id
        d.metadata["file_hash"] = file_hash
        d.metadata["collection"] = collection
    return docs


def ingest_files_to_chroma(
    file_paths: List[str],
    *,
    collection: str,
    chunker_fn,
    embeddings,
    chroma_persist_dir: str = "./chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    index_db_path: str = "data/ingestion.db",
) -> Dict[str, str]:
    """
    Industry-standard ingestion flow:
    - hash file
    - skip if already indexed for collection
    - load -> chunk -> embed -> upsert
    Returns per-file status: indexed | skipped | error:...
    """
    from langchain_community.vectorstores import Chroma

    idx = IngestionIndex(index_db_path)

    # Open (or create) a Chroma collection for incremental adds.
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=chroma_persist_dir,
    )

    statuses: Dict[str, str] = {}

    for path in file_paths:
        try:
            file_hash = sha256_file(path)
            existing = idx.lookup_by_hash(file_hash, collection)
            if existing:
                statuses[path] = "skipped (already indexed)"
                continue

            file_id = str(uuid.uuid4())
            docs = load_file_as_documents(path)
            docs = _tag_docs(docs, file_id=file_id, file_hash=file_hash, collection=collection)

            chunks = chunker_fn(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Ensure each chunk has stable IDs so duplicates can be updated if needed.
            ids: List[str] = []
            for i, c in enumerate(chunks):
                content_hash = hashlib.sha256(c.page_content.encode("utf-8")).hexdigest()[:16]
                ids.append(f"{file_id}:{i}:{content_hash}")
                c.metadata = c.metadata or {}
                c.metadata["chunk_index"] = i
                c.metadata["chunk_id"] = ids[-1]

            store.add_documents(chunks, ids=ids)
            store.persist()

            idx.upsert_file(file_id=file_id, file_path=path, file_hash=file_hash, collection=collection)
            statuses[path] = f"indexed ({len(chunks)} chunks)"
        except Exception as e:
            statuses[path] = f"error: {e}"

    return statuses

