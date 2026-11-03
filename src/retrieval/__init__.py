from .semantic import initialize as semantic_initialize
from .bm25 import initialize as bm25_initialize
from .hybrid import Hit, SearchSettings, initialize as hybrid_initialize, rrf_fuse, search
from .rerank import DedupeRerankRetriever

STRATEGIES = {
    "semantic": semantic_initialize,
    "bm25": bm25_initialize,
    "hybrid": hybrid_initialize,
}

# Config names -> modes of the KnowledgeBase pipeline in hybrid.search
MODE_ALIASES = {"semantic": "dense", "dense": "dense", "bm25": "bm25", "hybrid": "hybrid"}


def get_retriever(
    strategy: str,
    vectorstore,
    documents=None,
    top_k=5,
    use_reranker=False,
    **kwargs,
):
    """Legacy LangChain retriever factory used by custom config-driven pipelines."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Options: {list(STRATEGIES.keys())}")
    fetch_k = top_k * 3 if use_reranker else top_k
    base = STRATEGIES[strategy](vectorstore=vectorstore, documents=documents, top_k=fetch_k, **kwargs)
    return DedupeRerankRetriever(base_retriever=base, use_reranker=use_reranker, top_n=top_k)
