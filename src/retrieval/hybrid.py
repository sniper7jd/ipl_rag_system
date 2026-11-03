"""
Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion,
optional metadata filters, cross-encoder rerank, dedupe and a context token budget.

Every stage is recorded in a trace dict so the UI can show exactly how the context for
an answer was assembled.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.retrieval.query import expand_query
from src.retrieval.rerank import apply_token_budget, rerank_scores

_POOL = ThreadPoolExecutor(max_workers=4)


@dataclass
class SearchSettings:
    mode: str = "hybrid"          # dense | bm25 | hybrid
    top_k: int = 6
    fetch_k: int = 30
    rrf_k: int = 60
    use_reranker: bool = True
    use_filters: bool = True
    expand_query: bool = True     # append canonical team names for aliases ("RCB")
    token_budget: int = 3000
    min_filtered: int = 8         # fall back to unfiltered search below this many candidates


@dataclass
class Hit:
    doc: Document
    chunk_index: int
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf: float = 0.0
    rerank: Optional[float] = None
    scores: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        md = self.doc.metadata
        return {
            "chunk": self.chunk_index,
            "title": md.get("title"),
            "section": md.get("section"),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf": round(self.rrf, 4),
            "rerank": None if self.rerank is None else round(self.rerank, 3),
        }


def rrf_fuse(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def search(kb, query: str, teams: Sequence[str] = (), season: Optional[int] = None,
           settings: SearchSettings = SearchSettings(), rerank_query: Optional[str] = None):
    """Run the full retrieval pipeline against one KnowledgeBase. Returns (hits, trace)."""
    trace: Dict = {"query": query, "teams": list(teams), "season": season, "mode": settings.mode,
                   "timings_ms": {}, "filter": {"applied": False}}
    expanded = expand_query(query) if settings.expand_query else query
    if expanded != query:
        trace["expanded_query"] = expanded
        rerank_query = rerank_query or expanded
        query = expanded

    allowed = kb.allowed_ids(teams, season) if settings.use_filters else None
    if allowed is not None:
        if len(allowed) >= settings.min_filtered:
            trace["filter"] = {"applied": True, "candidates": len(allowed), "of": len(kb)}
        else:
            trace["filter"] = {"applied": False, "candidates": len(allowed), "of": len(kb),
                               "fallback": f"only {len(allowed)} chunks matched the filter - searched everything"}
            allowed = None

    dense: List[Tuple[int, float]] = []
    sparse: List[Tuple[int, float]] = []

    def run_dense():
        t = time.perf_counter()
        res = kb.dense_search(query, settings.fetch_k, allowed)
        trace["timings_ms"]["dense"] = _ms(t)
        return res

    def run_sparse():
        t = time.perf_counter()
        res = kb.bm25_search(query, settings.fetch_k, allowed)
        trace["timings_ms"]["bm25"] = _ms(t)
        return res

    if settings.mode == "hybrid":
        f_dense, f_sparse = _POOL.submit(run_dense), _POOL.submit(run_sparse)
        dense, sparse = f_dense.result(), f_sparse.result()
    elif settings.mode == "dense":
        dense = run_dense()
    elif settings.mode == "bm25":
        sparse = run_sparse()
    else:
        raise ValueError(f"Unknown retrieval mode: {settings.mode}")

    hits: Dict[int, Hit] = {}
    for rank, (i, s) in enumerate(dense, start=1):
        hits.setdefault(i, Hit(kb.docs[i], i)).__dict__.update(dense_rank=rank, dense_score=s)
    for rank, (i, s) in enumerate(sparse, start=1):
        hits.setdefault(i, Hit(kb.docs[i], i)).__dict__.update(bm25_rank=rank, bm25_score=s)

    fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]], k=settings.rrf_k)
    for i, score in fused:
        hits[i].rrf = score
    # Near-duplicate suppression: season pages share boilerplate paragraphs that differ only
    # in their breadcrumb header, so compare bodies (header stripped) by their first 200 chars.
    seen, ranked, dropped = set(), [], 0
    for i, _ in fused:
        body = hits[i].doc.page_content.split("\n", 1)[-1]
        key = " ".join(body.split()).lower()[:200]
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        ranked.append(hits[i])
        if len(ranked) >= settings.fetch_k:
            break
    trace["deduped"] = dropped

    if settings.use_reranker and len(ranked) > 1:
        t = time.perf_counter()
        scores = rerank_scores(rerank_query or query, [h.doc.page_content for h in ranked])
        for h, s in zip(ranked, scores):
            h.rerank = s
        ranked.sort(key=lambda h: h.rerank, reverse=True)
        trace["timings_ms"]["rerank"] = _ms(t)

    top = ranked[: settings.top_k]
    kept, used = apply_token_budget([(h.doc, h) for h in top], settings.token_budget)
    final = [h for _, h in kept]

    trace["candidates"] = [h.summary() for h in ranked]
    trace["final"] = [h.chunk_index for h in final]
    trace["context_tokens"] = used
    trace["dropped_by_budget"] = len(top) - len(final)
    return final, trace


def initialize(vectorstore, documents, top_k=5, weights=None, **_kwargs):
    """Legacy LangChain strategy (config-driven pipeline): weighted RRF via EnsembleRetriever."""
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    if documents is None:
        raise ValueError("Hybrid retriever requires documents for BM25 indexing.")
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = top_k
    return EnsembleRetriever(retrievers=[semantic_retriever, keyword_retriever], weights=weights or [0.5, 0.5])
