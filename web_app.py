import os
import sys
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config, save_config
from src.loaders import fetch_team_data
from src.chunking import get_chunker
from src.embeddings import get_embeddings
from src.vector_stores import get_vector_store
from src.retrieval import get_retriever
from src.llms import get_llm_chain
from src.embeddings import PROVIDERS as EMBEDDING_PROVIDERS
from src.agents import create_rag_agent, invoke_agent
from src.ingestion import ingest_files_to_chroma
from src.vector_stores.chroma_db import from_existing_collection

load_dotenv()

st.set_page_config(page_title="IPL RAG Chat", page_icon="🏏", layout="wide")


def _format_docs(docs, max_chars=600):
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        formatted.append(f"[{i}] {source}\n{content}")
    return "\n\n".join(formatted)


@st.cache_resource
def _build_vectorstore_wikipedia(config):
    """Build vectorstore from Wikipedia (IPL teams)."""
    pipeline = config["pipeline"]
    params = config.get("params", {})

    teams = ["Chennai Super Kings", "Mumbai Indians", "Delhi Capitals", "Kolkata Knight Riders", "Royal Challengers Bengaluru", "Sunrisers Hyderabad", "Rajasthan Royals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"]
    documents = fetch_team_data(teams)

    chunker = get_chunker(pipeline["chunking"])
    embeddings_for_chunking = None
    if pipeline["chunking"] == "semantic":
        embeddings_for_chunking = get_embeddings(pipeline["embeddings"])

    chunks = chunker(
        documents,
        embeddings_model=embeddings_for_chunking,
        chunk_size=params.get("chunk_size", 1000),
        chunk_overlap=params.get("chunk_overlap", 100),
    )

    embeddings = get_embeddings(pipeline["embeddings"])
    vectorstore = get_vector_store(
        pipeline["vector_store"],
        chunks=chunks,
        embeddings=embeddings,
    )
    return chunks, vectorstore


def _get_vectorstore_and_chunks(config, data_source: str, query_collection: str, chroma_persist_dir: str):
    """Return (vectorstore, chunks or None) based on data source. Chunks only for Wikipedia (BM25/hybrid)."""
    if data_source == "uploads":
        embeddings = get_embeddings(config["pipeline"]["embeddings"])
        vectorstore = from_existing_collection(
            collection_name=query_collection,
            embeddings=embeddings,
            persist_directory=chroma_persist_dir,
        )
        return vectorstore, None  # No in-memory chunks for uploads

    chunks, vectorstore = _build_vectorstore_wikipedia(config)
    return vectorstore, chunks


def _get_chain_and_context(question, config, data_source, query_collection, chroma_persist_dir, llm_kwargs=None):
    vectorstore, chunks = _get_vectorstore_and_chunks(config, data_source, query_collection, chroma_persist_dir)
    pipeline = config["pipeline"]
    params = config.get("params", {})

    retrieval = pipeline["retrieval"]
    if data_source == "uploads" and retrieval in ("bm25", "hybrid"):
        retrieval = "semantic"  # BM25/hybrid need in-memory docs; uploads are in Chroma only

    retriever = get_retriever(
        retrieval,
        vectorstore=vectorstore,
        documents=chunks if retrieval in ("bm25", "hybrid") else None,
        top_k=params.get("top_k", 5),
        use_reranker=params.get("use_reranker", True),
    )
    if llm_kwargs is None:
        llm_kwargs = {}
    chain = get_llm_chain(pipeline["llm"], retriever=retriever, **llm_kwargs)

    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, docs


def _get_agent_response(question, config, data_source, query_collection, chroma_persist_dir, llm_kwargs=None):
    """Agentic path: agent decides whether to retrieve or answer directly."""
    vectorstore, chunks = _get_vectorstore_and_chunks(config, data_source, query_collection, chroma_persist_dir)
    pipeline = config["pipeline"]
    params = config.get("params", {})

    retrieval = pipeline["retrieval"]
    if data_source == "uploads" and retrieval in ("bm25", "hybrid"):
        retrieval = "semantic"

    retriever = get_retriever(
        retrieval,
        vectorstore=vectorstore,
        documents=chunks if retrieval in ("bm25", "hybrid") else None,
        top_k=params.get("top_k", 5),
        use_reranker=params.get("use_reranker", True),
    )
    llm_kwargs = llm_kwargs or {}
    agent = create_rag_agent(
        retriever,
        llm_provider=pipeline["llm"],
        llm_kwargs=llm_kwargs,
    )
    return invoke_agent(agent, question)


st.title("🏏 IPL Expert RAG Chat")

st.subheader("Upload & Index Documents")
with st.expander("Upload PDF/DOCX/HTML/CSV and index into Chroma", expanded=True):
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "html", "htm", "csv", "txt", "md"],
        accept_multiple_files=True,
    )
    collection_name = st.text_input("Chroma collection", value="ipl_uploads")
    persist_dir = st.text_input("Chroma persist directory", value="./chroma_db")

    if st.button("Index uploads", disabled=not uploaded):
        # Save uploads locally first
        save_dir = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for f in uploaded:
            out_path = save_dir / f.name
            out_path.write_bytes(f.getbuffer())
            saved_paths.append(str(out_path))

        config = load_config()
        params = config.get("params", {})
        chunker = get_chunker(config["pipeline"]["chunking"])
        embeddings = get_embeddings(config["pipeline"]["embeddings"])

        statuses = ingest_files_to_chroma(
            saved_paths,
            collection=collection_name,
            chunker_fn=chunker,
            embeddings=embeddings,
            chroma_persist_dir=persist_dir,
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 100),
        )
        st.success("Indexing finished.")
        st.json(statuses)
        st.cache_resource.clear()

with st.sidebar:
    st.header("Controls")
    config_current = load_config()

    st.subheader("Data source")
    data_source = st.radio(
        "Query from",
        ["wikipedia", "uploads"],
        format_func=lambda x: "Wikipedia (IPL teams)" if x == "wikipedia" else "Uploaded documents",
        index=0,
    )
    query_collection = "ipl_uploads"
    chroma_persist_dir = "./chroma_db"
    if data_source == "uploads":
        query_collection = st.text_input("Chroma collection (uploads)", value="ipl_uploads")
        chroma_persist_dir = st.text_input("Chroma persist dir", value="./chroma_db")

    st.subheader("Pipeline")
    chunking_options = ["recursive", "semantic", "parent_child", "paragraph"]
    embeddings_options = list(EMBEDDING_PROVIDERS.keys())
    if not embeddings_options:
        st.error("No embeddings providers are available. Install one of the embeddings integrations.")
        st.stop()
    vector_store_options = ["chroma", "faiss", "pinecone"]
    retrieval_options = ["semantic", "bm25", "hybrid"]
    llm_options = ["openai", "gemini", "groq", "claude"]

    chunking = st.selectbox(
        "Chunking",
        chunking_options,
        index=chunking_options.index(config_current["pipeline"]["chunking"]),
    )
    current_embeddings = config_current["pipeline"]["embeddings"]
    if current_embeddings not in embeddings_options:
        st.warning(f"Embeddings '{current_embeddings}' is not available. Select another option.")
        current_embeddings = embeddings_options[0]

    embeddings = st.selectbox(
        "Embeddings",
        embeddings_options,
        index=embeddings_options.index(current_embeddings),
    )
    vector_store = st.selectbox(
        "Vector store",
        vector_store_options,
        index=vector_store_options.index(config_current["pipeline"]["vector_store"]),
    )
    retrieval = st.selectbox(
        "Retrieval",
        retrieval_options,
        index=retrieval_options.index(config_current["pipeline"]["retrieval"]),
    )
    llm = st.selectbox(
        "LLM",
        llm_options,
        index=llm_options.index(config_current["pipeline"]["llm"]),
    )
    groq_model = None
    gemini_model = None
    if llm == "groq":
        groq_model = st.text_input(
            "Groq model (override)",
            value=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
            help="Set a model supported by your Groq API key.",
        )
    elif llm == "gemini":
        gemini_model = st.text_input(
            "Gemini model (override)",
            value=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            help="Set a model supported by your Gemini API key.",
        )

    params = config_current.get("params", {})
    use_agentic = st.checkbox(
        "Agentic mode (conditional retrieval)",
        value=bool(params.get("use_agentic", True)),
        help="Agent decides when to query the vector DB vs answer directly. Industry standard.",
    )
    chunk_size = st.number_input(
        "Chunk size",
        min_value=200,
        max_value=4000,
        value=int(params.get("chunk_size", 1000)),
        step=100,
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=1000,
        value=int(params.get("chunk_overlap", 100)),
        step=50,
    )
    top_k = st.slider(
        "Context window (top_k)",
        min_value=2,
        max_value=12,
        value=int(params.get("top_k", 5)),
        step=1,
    )
    use_reranker = st.checkbox(
        "Dedupe + rerank (cross-encoder)",
        value=bool(params.get("use_reranker", True)),
        help="Remove duplicate chunks and rerank with a lightweight cross-encoder.",
    )
    show_context = st.checkbox("Show retrieved context", value=True)

    updated_config = {
        "pipeline": {
            "chunking": chunking,
            "embeddings": embeddings,
            "vector_store": vector_store,
            "retrieval": retrieval,
            "llm": llm,
        },
        "params": {
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "top_k": int(top_k),
            "use_reranker": use_reranker,
            "use_agentic": use_agentic,
        },
    }

    if st.button("Save to config.yaml"):
        save_config(updated_config)
        st.cache_resource.clear()
        st.success("Saved. Pipeline cache cleared.")

    st.divider()
    st.caption("Active configuration")
    st.json(updated_config)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context"):
            with st.expander("Context window"):
                st.markdown(msg["context"])

prompt = st.chat_input("Ask a question..." + (" (uploaded docs)" if data_source == "uploads" else " (IPL teams)"))
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..." if use_agentic else "Retrieving context and generating answer..."):
        llm_kwargs = {}
        if llm == "groq" and groq_model:
            llm_kwargs["model"] = groq_model.strip()
        elif llm == "gemini" and gemini_model:
            llm_kwargs["model"] = gemini_model.strip()

        if use_agentic:
            answer, context_str = _get_agent_response(
                prompt, config=updated_config,
                data_source=data_source, query_collection=query_collection, chroma_persist_dir=chroma_persist_dir,
                llm_kwargs=llm_kwargs,
            )
            context_text = context_str if context_str else "*(No retrieval — answered directly)*"
        else:
            answer, docs = _get_chain_and_context(
                prompt, config=updated_config,
                data_source=data_source, query_collection=query_collection, chroma_persist_dir=chroma_persist_dir,
                llm_kwargs=llm_kwargs,
            )
            context_text = _format_docs(docs)

    assistant_msg = {"role": "assistant", "content": answer}
    if show_context:
        assistant_msg["context"] = context_text
    st.session_state["messages"].append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_context:
            with st.expander("Context window"):
                st.markdown(context_text)
