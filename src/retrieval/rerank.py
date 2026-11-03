"""
Deduplication, cross-encoder reranking (FastEmbed / ONNX) and context token budgeting.
"""
import os
from functools import lru_cache
from typing import List, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker(model_name: str = None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(
        model_name=model_name or os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL),
        cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
    )


def rerank_scores(query: str, texts: Sequence[str]) -> List[float]:
    if not texts:
        return []
    batch = int(os.getenv("RERANK_BATCH_SIZE", "8"))
    return [float(s) for s in get_reranker().rerank(query, list(texts), batch_size=batch)]


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = " ".join(d.page_content.split()).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def apply_token_budget(items: List[Tuple[Document, dict]], max_tokens: int, min_items: int = 1):
    """Keep the highest-ranked chunks until the context token budget is spent."""
    kept, used = [], 0
    for doc, info in items:
        tokens = doc.metadata.get("tokens") or max(1, len(doc.page_content) // 4)
        if kept and len(kept) >= min_items and used + tokens > max_tokens:
            break
        kept.append((doc, info))
        used += tokens
    return kept, used


def _rerank_docs(query: str, docs: List[Document]) -> List[Document]:
    scores = rerank_scores(query, [d.page_content for d in docs])
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]


class DedupeRerankRetriever(BaseRetriever):
    """Wraps any LangChain retriever: dedupe by content, then cross-encoder rerank."""

    base_retriever: BaseRetriever
    use_reranker: bool = True
    top_n: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        unique = dedupe_docs(self.base_retriever.invoke(query))
        if not self.use_reranker or len(unique) <= 1:
            return unique[: self.top_n]
        return _rerank_docs(query, unique)[: self.top_n]
