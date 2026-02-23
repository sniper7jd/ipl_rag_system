from .semantic import initialize as semantic_initialize
from .bm25 import initialize as bm25_initialize
from .hybrid import initialize as hybrid_initialize

STRATEGIES = {
    "semantic": semantic_initialize,
    "bm25": bm25_initialize,
    "hybrid": hybrid_initialize,
}


def get_retriever(strategy: str, vectorstore, documents=None, top_k=5, **kwargs):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy: {strategy}. Options: {list(STRATEGIES.keys())}")
    return STRATEGIES[strategy](
        vectorstore=vectorstore,
        documents=documents,
        top_k=top_k,
        **kwargs,
    )
