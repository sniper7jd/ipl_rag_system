from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


def process(documents, embeddings_model=None, **_kwargs):
    if not embeddings_model:
        embeddings_model = OpenAIEmbeddings()

    splitter = SemanticChunker(embeddings_model)
    return splitter.split_documents(documents)
