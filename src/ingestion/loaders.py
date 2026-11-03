from __future__ import annotations

import csv
import mimetypes
from pathlib import Path
from typing import List

from langchain_core.documents import Document

MAX_CSV_ROWS = 5000


def _pdf(path: Path) -> List[Document]:
    import pymupdf

    with pymupdf.open(path) as pdf:
        return [Document(page_content=page.get_text(), metadata={"page": i}) for i, page in enumerate(pdf)
                if page.get_text().strip()]


def _docx(path: Path) -> List[Document]:
    import docx2txt

    return [Document(page_content=docx2txt.process(str(path)))]


def _html(path: Path) -> List[Document]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for el in soup(["script", "style", "nav", "footer"]):
        el.decompose()
    title = soup.title.get_text(strip=True) if soup.title else path.name
    return [Document(page_content=soup.get_text("\n", strip=True), metadata={"title": title})]


def _csv(path: Path) -> List[Document]:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = [" | ".join(f"{k}: {v}" for k, v in row.items() if v) for _, row in zip(range(MAX_CSV_ROWS), reader)]
    return [Document(page_content="\n".join(rows))]


def _text(path: Path) -> List[Document]:
    return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"))]


LOADERS = {".pdf": _pdf, ".docx": _docx, ".html": _html, ".htm": _html, ".csv": _csv, ".txt": _text, ".md": _text}


def load_file_as_documents(file_path: str) -> List[Document]:
    """Load a local file into LangChain Documents. Supported: pdf, docx, html/htm, csv, txt/md."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = [d for d in LOADERS[ext](path) if d.page_content.strip()]
    mime, _ = mimetypes.guess_type(str(path))
    for d in docs:
        d.metadata.setdefault("source", str(path))
        d.metadata.setdefault("filename", path.name)
        d.metadata.setdefault("file_ext", ext)
        if mime:
            d.metadata.setdefault("mime_type", mime)
    return docs
