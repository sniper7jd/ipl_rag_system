"""
Central Orchestrator - Config-driven RAG pipeline.
Reads config.yaml and dynamically plugs in the selected strategies.
"""
import os
import sys
from dotenv import load_dotenv

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config
from src.loaders import fetch_team_data
from src.chunking import get_chunker
from src.embeddings import get_embeddings
from src.vector_stores import get_vector_store
from src.retrieval import get_retriever
from src.llms import get_llm_chain

load_dotenv()


def main():
    # 1. Read Config
    config = load_config()
    pipeline = config["pipeline"]
    params = config.get("params", {})

    # 2. Load Data
    teams = ["Chennai Super Kings", "Mumbai Indians", "Delhi Capitals"]
    print("Step 1: Fetching...")
    documents = fetch_team_data(teams)

    # 3. Chunking
    chunker = get_chunker(pipeline["chunking"])
    embeddings_for_chunking = None
    if pipeline["chunking"] == "semantic":
        embeddings_for_chunking = get_embeddings(pipeline["embeddings"])

    print(f"Step 2: Chunking ({pipeline['chunking']})...")
    chunks = chunker(
        documents,
        embeddings_model=embeddings_for_chunking,
        chunk_size=params.get("chunk_size", 1000),
        chunk_overlap=params.get("chunk_overlap", 100),
    )

    # 4. Embeddings & Vector Store
    embeddings = get_embeddings(pipeline["embeddings"])
    print(f"Step 3: Indexing ({pipeline['vector_store']})...")
    vectorstore = get_vector_store(
        pipeline["vector_store"],
        chunks=chunks,
        embeddings=embeddings,
    )

    # 5. Retriever
    print(f"Step 4: Retrieval ({pipeline['retrieval']})...")
    retriever = get_retriever(
        pipeline["retrieval"],
        vectorstore=vectorstore,
        documents=chunks if pipeline["retrieval"] in ("bm25", "hybrid") else None,
        top_k=params.get("top_k", 5),
    )

    # 6. LLM Chain
    print(f"Step 5: Initializing LLM ({pipeline['llm']})...")
    rag_chain = get_llm_chain(pipeline["llm"], retriever=retriever)

    # Run query
    query = input("Ask a question about IPL teams: ")
    response = rag_chain.invoke(query)
    print(f"\nAI Response:\n{response}")


if __name__ == "__main__":
    main()
