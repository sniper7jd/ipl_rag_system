"""
KnowledgeBase: one set of chunks indexed two ways.

- Dense: FAISS IndexFlatIP over L2-normalised bge-small embeddings (exact cosine search;
  at a few thousand chunks an approximate index would only add error).
- Sparse: BM25Okapi over lowercase word tokens, for exact names and numbers.

Both searches accept an optional allow-list of chunk ids, which is how metadata filters
(team / season) are applied. Persisted as faiss.index + chunks.jsonl (no pickle).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = set("""a an and are as at be by did do does for from had has have how i in is it its
of on or that the their them they this to was were what when where which who whom whose why
will with you your about tell me give list""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors


class KnowledgeBase:
    def __init__(self, docs: List[Document], vectors: np.ndarray, embeddings, name: str = "knowledge"):
        self.docs = docs
        self.embeddings = embeddings
        self.name = name
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(_normalize(vectors))
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
        self.by_team: Dict[str, Set[int]] = defaultdict(set)
        self.by_season: Dict[int, Set[int]] = defaultdict(set)
        for i, d in enumerate(docs):
            for t in d.metadata.get("teams") or []:
                self.by_team[t].add(i)
            for s in d.metadata.get("seasons") or []:
                self.by_season[int(s)].add(i)

    def __len__(self):
        return len(self.docs)

    @classmethod
    def from_documents(cls, docs: List[Document], embeddings, name: str = "knowledge"):
        vectors = np.array(embeddings.embed_documents([d.page_content for d in docs]), dtype="float32")
        return cls(docs, vectors, embeddings, name=name)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path, embeddings, name: str = "knowledge"):
        directory = Path(directory)
        index = faiss.read_index(str(directory / "faiss.index"))
        vectors = index.reconstruct_n(0, index.ntotal)
        docs = []
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(page_content=row["text"], metadata=row["metadata"]))
        return cls(docs, vectors, embeddings, name=name)

    def allowed_ids(self, teams: Iterable[str] = (), season: Optional[int] = None) -> Optional[Set[int]]:
        teams = list(teams)
        if not teams and season is None:
            return None
        ids: Optional[Set[int]] = None
        if teams:
            ids = set().union(*(self.by_team.get(t, set()) for t in teams))
        if season is not None:
            season_ids = self.by_season.get(int(season), set())
            ids = season_ids if ids is None else ids & season_ids
        return ids

    def embed_query(self, query: str) -> np.ndarray:
        return _normalize(np.array([self.embeddings.embed_query(query)], dtype="float32"))

    def dense_search(self, query: str, k: int, allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        qv = self.embed_query(query)
        params = None
        if allowed is not None:
            if not allowed:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(sorted(allowed), dtype="int64")))
            k = min(k, len(allowed))
        scores, ids = self.index.search(qv, min(k, len(self.docs)), params=params)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, k: int, allowed: Optional[Set[int]] = None) -> List[Tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if allowed is not None:
            mask = np.full(len(scores), -np.inf)
            idx = np.fromiter(allowed, dtype=int)
            mask[idx] = scores[idx]
            scores = mask
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]
