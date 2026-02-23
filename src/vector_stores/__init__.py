from .chroma_db import initialize as chroma_initialize

STORES = {
    "chroma": chroma_initialize,
}

try:
    from .faiss_db import initialize as faiss_initialize
    STORES["faiss"] = faiss_initialize
except ImportError:
    pass

try:
    from .pinecone_db import initialize as pinecone_initialize
    STORES["pinecone"] = pinecone_initialize
except ImportError:
    pass


def get_vector_store(store_name: str, chunks, embeddings, **kwargs):
    if store_name not in STORES:
        raise ValueError(f"Unknown vector store: {store_name}. Options: {list(STORES.keys())}")
    return STORES[store_name](chunks=chunks, embeddings=embeddings, **kwargs)
