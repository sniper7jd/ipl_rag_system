import os
from dotenv import load_dotenv
from loader import fetch_team_data
from chunking import get_text_chunks
from database import create_vector_store
from engine import get_rag_chain

load_dotenv() # Load API Key from .env

def main():
    teams = ["Chennai Super Kings", "Mumbai Indians", "Delhi Capitals"]
    
    print("Step 1: Fetching...")
    docs = fetch_team_data(teams)
    
    print("Step 2: Chunking...")
    chunks = get_text_chunks(docs)
    
    print("Step 3: Indexing...")
    retriever = create_vector_store(chunks)
    
    print("Step 4: Initializing Engine...")
    rag_chain = get_rag_chain(retriever)
    
    # Run a test
    query = input("Ask a question about IPL teams: " )
    response = rag_chain.invoke(query)
    print(f"\nAI Response:\n{response}")

if __name__ == "__main__":
    main()