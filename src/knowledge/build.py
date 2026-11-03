"""Build, persist and load the IPL knowledge snapshot."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.chunking import get_chunker
from src.chunking.structured import chunk_pages
from src.embeddings import get_embeddings
from src.embeddings.huggingface_eff import DEFAULT_MODEL as EMBEDDING_MODEL
from src.knowledge.base import KnowledgeBase
from src.retrieval.query import TEAMS

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "knowledge"
PAGES_FILE = "pages.jsonl"
MANIFEST_FILE = "manifest.json"

LEAGUE_PAGES = [
    "2026 Indian Premier League",
    "Indian Premier League",
    "List of Indian Premier League seasons and results",
]


def season_pages(season: int) -> List[str]:
    return [f"{season} {team} season" for team in TEAMS]


def page_titles(season: int = 2026) -> List[str]:
    return LEAGUE_PAGES + season_pages(season) + TEAMS


PAGES = page_titles()


def content_hash(pages: List[Dict]) -> str:
    h = hashlib.sha256()
    for p in sorted(pages, key=lambda p: p["title"]):
        h.update(json.dumps(p["sections"], sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def build_from_pages(pages: List[Dict], embeddings=None, chunking: str = "structured",
                     chunk_tokens: int = 400, overlap_tokens: int = 50, name: str = "ipl") -> KnowledgeBase:
    embeddings = embeddings or get_embeddings("huggingface")
    if chunking == "structured":
        docs = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    else:
        from src.chunking.structured import _make_doc, _page_season
        from src.loaders.wikipedia import page_to_text

        raw = [Document(page_content=page_to_text(p), metadata={"title": p["title"], "url": p["url"]}) for p in pages]
        kwargs = {"chunk_size": chunk_tokens * 4, "chunk_overlap": overlap_tokens * 4}
        if chunking == "semantic":
            kwargs["embeddings_model"] = embeddings
        pieces = get_chunker(chunking)(raw, **kwargs)
        docs = []
        for i, d in enumerate(pieces):
            meta = {"title": d.metadata["title"], "url": d.metadata["url"], "page_url": d.metadata["url"],
                    "section": "", "page_team": None, "page_season": _page_season(d.metadata["title"])}
            doc = _make_doc(d.page_content, "", meta, chunking)
            doc.metadata["chunk_id"] = i
            docs.append(doc)
    return KnowledgeBase.from_documents(docs, embeddings, name=name)


def save_pages(pages: List[Dict], directory: Path = KNOWLEDGE_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / PAGES_FILE, "w", encoding="utf-8") as f:
        for p in sorted(pages, key=lambda p: p["title"]):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_pages(directory: Path = KNOWLEDGE_DIR) -> List[Dict]:
    with open(directory / PAGES_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_manifest(pages: List[Dict], kb: KnowledgeBase, latest_season: int, directory: Path = KNOWLEDGE_DIR) -> Dict:
    manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content_hash": content_hash(pages),
        "latest_season": latest_season,
        "embedding_model": EMBEDDING_MODEL,
        "chunking": "structured",
        "num_pages": len(pages),
        "num_chunks": len(kb),
        "pages": [
            {"title": p["title"], "url": p["url"], "revision_timestamp": p["revision_timestamp"]}
            for p in sorted(pages, key=lambda p: p["title"])
        ],
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(directory: Path = KNOWLEDGE_DIR) -> Optional[Dict]:
    path = directory / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def load_default(embeddings=None, directory: Path = KNOWLEDGE_DIR) -> KnowledgeBase:
    return KnowledgeBase.load(directory, embeddings or get_embeddings("huggingface"), name="ipl")
