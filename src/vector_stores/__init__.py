"""Legacy LangChain vector stores for custom config-driven pipelines.

The app itself uses src.knowledge.KnowledgeBase (FAISS + BM25). These need the optional
langchain-community package (and chromadb / pinecone for those stores)."""
import importlib

_MODULES = {"chroma": "src.vector_stores.chroma_db", "faiss": "src.vector_stores.faiss_db",
            "pinecone": "src.vector_stores.pinecone_db"}


def get_vector_store(store_name: str, chunks, embeddings, **kwargs):
    if store_name not in _MODULES:
        raise ValueError(f"Unknown vector store: {store_name}. Options: {list(_MODULES)}")
    return importlib.import_module(_MODULES[store_name]).initialize(chunks=chunks, embeddings=embeddings, **kwargs)
