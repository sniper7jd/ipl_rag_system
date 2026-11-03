from langchain_experimental.text_splitter import SemanticChunker


def process(documents, embeddings_model=None, **_kwargs):
    if embeddings_model is None:
        from src.embeddings import get_embeddings

        embeddings_model = get_embeddings("huggingface")
    splitter = SemanticChunker(embeddings_model)
    return splitter.split_documents(documents)
