from langchain_community.vectorstores import FAISS


def initialize(chunks, embeddings, **kwargs):
    return FAISS.from_documents(documents=chunks, embedding=embeddings)
