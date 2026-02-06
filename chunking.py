from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)