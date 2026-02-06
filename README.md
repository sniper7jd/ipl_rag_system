# 🏏 IPL Expert RAG System (2026)

A modular Retrieval-Augmented Generation (RAG) system built with **LangChain v1.0** and **ChromaDB**. This system specifically tracks the 2026 IPL season, handling complex queries regarding team rebrands and major player transfers (e.g., Sanju Samson to CSK).

## 🚀 Features
- **Modular Architecture**: Separate components for Data Loading, Chunking, Vector Storage, and the LLM Engine.
- **Contextual Reasoning**: Handles historical team names (e.g., "Delhi Daredevils") by correcting users based on current 2026 data.
- **Persistent Storage**: Uses a local ChromaDB instance to avoid re-embedding data on every run.

## 📁 Project Structure
- `main.py`: Orchestrates the entire pipeline.
- `loader.py`: Fetches real-time data from Wikipedia.
- `chunking.py`: Implements recursive character splitting.
- `database.py`: Manages embeddings and the persistent vector store.
- `engine.py`: Defines the LCEL chain and expert system prompt.

## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/sniper7jd/ipl-rag-system.git](https://github.com/sniper7jd/ipl-rag-system.git)
   cd ipl-rag-system