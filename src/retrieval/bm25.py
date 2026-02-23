from langchain_community.retrievers import BM25Retriever


def initialize(vectorstore, documents, top_k=5, **_kwargs):
    if documents is None:
        raise ValueError("BM25 retriever requires documents for keyword indexing.")
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = top_k
    return keyword_retriever
