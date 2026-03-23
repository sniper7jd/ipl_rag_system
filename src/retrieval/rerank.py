"""
Lightweight deduplication + cross-encoder reranking for retrieved chunks.
"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

# Lazy-load cross-encoder to avoid import cost when not used
_RERANKER_MODEL = None


def _get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        from sentence_transformers import CrossEncoder

        _RERANKER_MODEL = CrossEncoder(model_name)
    return _RERANKER_MODEL


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _rerank_docs(query: str, docs: List[Document], model_name: str) -> List[Document]:
    if not docs:
        return docs
    model = _get_reranker(model_name)
    pairs = [(query, d.page_content) for d in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked]


class DedupeRerankRetriever(BaseRetriever):
    """Wraps a retriever: deduplicates by content, then reranks with cross-encoder."""

    base_retriever: BaseRetriever
    use_reranker: bool = True
    top_n: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        unique = _dedupe_docs(docs)
        if not self.use_reranker or len(unique) <= 1:
            return unique[: self.top_n]
        reranked = _rerank_docs(query, unique, self.rerank_model)
        return reranked[: self.top_n]
