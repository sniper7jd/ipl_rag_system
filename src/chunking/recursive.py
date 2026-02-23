from langchain_text_splitters import RecursiveCharacterTextSplitter


def process(documents, chunk_size=1000, chunk_overlap=100, **_kwargs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
