from langchain_text_splitters import RecursiveCharacterTextSplitter


def process(documents, chunk_size=1000, chunk_overlap=100, **_kwargs):
    # Paragraph-aware: split on paragraphs first, then by size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
