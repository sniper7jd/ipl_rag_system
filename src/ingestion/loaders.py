from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    CSVLoader,
    TextLoader,
)


def load_file_as_documents(file_path: str) -> List[Document]:
    """
    Load a local file into LangChain Documents with best-effort loader selection.
    Supported: pdf, docx, html/htm, csv, txt/md.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))

    if ext == ".pdf":
        docs = PyMuPDFLoader(str(path)).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext in (".html", ".htm"):
        docs = BSHTMLLoader(str(path)).load()
    elif ext == ".csv":
        docs = CSVLoader(str(path)).load()
    elif ext in (".txt", ".md"):
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext} ({mime})")

    # Add basic metadata
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", str(path))
        d.metadata.setdefault("filename", path.name)
        d.metadata.setdefault("file_ext", ext)
        if mime:
            d.metadata.setdefault("mime_type", mime)
    return docs

