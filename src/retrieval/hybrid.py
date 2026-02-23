from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def initialize(vectorstore, documents, top_k=5, weights=None, **_kwargs):
    if documents is None:
        raise ValueError("Hybrid retriever requires documents for BM25 indexing.")

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = top_k

    if weights is None:
        weights = [0.5, 0.5]

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=weights,
    )
    return ensemble_retriever
