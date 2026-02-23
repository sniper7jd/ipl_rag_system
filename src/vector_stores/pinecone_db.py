from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings


def initialize(chunks, embeddings, index_name=None, **kwargs):
    if not index_name:
        index_name = "ipl-rag"
    return Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
