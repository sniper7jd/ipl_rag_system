from .semantic import initialize as semantic_initialize
from .bm25 import initialize as bm25_initialize
from .hybrid import initialize as hybrid_initialize
from .rerank import DedupeRerankRetriever

STRATEGIES = {
    "semantic": semantic_initialize,
    "bm25": bm25_initialize,
    "hybrid": hybrid_initialize,
}


def get_retriever(
    strategy: str,
    vectorstore,
    documents=None,
    top_k=5,
    use_reranker=False,
    **kwargs,
):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Options: {list(STRATEGIES.keys())}")
    fetch_k = top_k * 3 if use_reranker else top_k  # more candidates for reranking
    base = STRATEGIES[strategy](
        vectorstore=vectorstore,
        documents=documents,
        top_k=fetch_k,
        **kwargs,
    )
    return DedupeRerankRetriever(
        base_retriever=base,
        use_reranker=use_reranker,
        top_n=top_k,
    )
