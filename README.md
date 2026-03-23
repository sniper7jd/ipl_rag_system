# IPL Expert RAG System

A config-driven Retrieval-Augmented Generation (RAG) pipeline built with **LangChain** and **ChromaDB**. Answer questions about IPL teams using Wikipedia-sourced data.

## Features

- **Config-driven pipeline**: Change chunking, embeddings, vector store, retrieval, and LLM via `config.yaml`
- **Pluggable strategies**: Recursive, semantic, or parent-child chunking; semantic, BM25, or hybrid retrieval
- **Persistent storage**: ChromaDB with optional FAISS and Pinecone support
- **Multiple LLM backends**: OpenAI, Gemini, Claude, Groq

## Project structure

```
ipl_rag_system/
├── main.py              # Central orchestrator
├── config.yaml          # Pipeline configuration
├── src/
│   ├── chunking/        # recursive, semantic, parent_child, paragraph
│   ├── embeddings/      # openai, huggingface
│   ├── vector_stores/   # chroma, faiss, pinecone
│   ├── retrieval/       # semantic, bm25, hybrid
│   ├── llms/            # openai, gemini, claude, groq
│   └── loaders/         # wikipedia
└── utils/               # logger, helpers
```

## Configuration

Edit `config.yaml` to switch strategies:

```yaml
pipeline:
  chunking: "recursive"       # recursive | semantic | parent_child | paragraph
  embeddings: "huggingface"   # openai | huggingface
  vector_store: "chroma"      # chroma | faiss | pinecone
  retrieval: "semantic"       # semantic | bm25 | hybrid
  llm: "openai"               # openai | gemini | groq | claude

params:
  chunk_size: 1000
  chunk_overlap: 100
  top_k: 5
```

## Setup

1. **Clone and enter the repo**:
   ```bash
   git clone https://github.com/sniper7jd/ipl-rag-system.git
   cd ipl-rag-system
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your API keys** in `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. **Run**:
   ```bash
   python main.py
   ```

## Web app (lightweight chat UI)

Run a minimal Streamlit chat UI with a configurable context window:

```bash
streamlit run web_app.py
```

Use the sidebar to adjust `top_k` (context window) and optionally show the retrieved context.
