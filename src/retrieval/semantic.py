def initialize(vectorstore, documents=None, top_k=5, **_kwargs):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
