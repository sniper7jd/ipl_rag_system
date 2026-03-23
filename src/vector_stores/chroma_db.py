from langchain_community.vectorstores import Chroma


def initialize(chunks, embeddings, persist_directory="./chroma_db", collection_name=None, **kwargs):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name or "langchain",
    )


def from_existing_collection(collection_name: str, embeddings, persist_directory: str = "./chroma_db"):
    """Connect to an existing Chroma collection (e.g. from uploads) for retrieval."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
