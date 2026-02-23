from langchain_text_splitters import RecursiveCharacterTextSplitter


def process(documents, chunk_size=1000, chunk_overlap=100, **_kwargs):
    # Parent-child: larger parent chunks with smaller child chunks for retrieval
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 2,
        chunk_overlap=chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    parent_chunks = parent_splitter.split_documents(documents)
    return child_splitter.split_documents(parent_chunks)
