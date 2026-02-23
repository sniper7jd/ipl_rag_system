from langchain_community.vectorstores import Chroma


def initialize(chunks, embeddings, persist_directory="./chroma_db", **kwargs):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
