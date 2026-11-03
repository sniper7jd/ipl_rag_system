"""IPL Expert - a public RAG demo over a weekly-refreshed Wikipedia snapshot."""
import json
import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agents import RAGContext, answer_question, linkify  # noqa: E402
from src.chunking import STRATEGIES as CHUNKING_STRATEGIES  # noqa: E402
from src.chunking.structured import process as chunk_uploads  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.ingestion.loaders import load_file_as_documents  # noqa: E402
from src.knowledge import KnowledgeBase, build_from_pages, load_default, load_manifest, load_pages  # noqa: E402
from src.llms import PROVIDER_MODULES, default_model, key_env  # noqa: E402
from src.retrieval import MODE_ALIASES, SearchSettings  # noqa: E402
from src.retrieval.rerank import rerank_scores  # noqa: E402
from utils.helpers import load_config, save_config  # noqa: E402

load_dotenv()
st.set_page_config(page_title="IPL Expert · RAG over Wikipedia", page_icon="🏏", layout="wide")

QUESTION_CAP = int(os.getenv("QUESTION_CAP", "20"))
MAX_QUESTION_CHARS = 500
MAX_UPLOAD_FILES = 5
MAX_UPLOAD_CHUNKS = 600
ADMIN = os.getenv("RAG_ADMIN") == "1"
PROVIDER_LABELS = {"gemini": "Google Gemini", "openai": "OpenAI", "groq": "Groq", "claude": "Anthropic Claude"}
KEY_HELP = {
    "gemini": "Free key: https://aistudio.google.com/apikey",
    "groq": "Free key: https://console.groq.com/keys",
    "openai": "https://platform.openai.com/api-keys",
    "claude": "https://console.anthropic.com/settings/keys",
}
EXAMPLES = [
    "Who won IPL 2026 and how?",
    "Who was the top run scorer of IPL 2026?",
    "Compare RCB and Gujarat Titans in 2026",
    "Who captained CSK in 2026, and who coached them?",
    "Which teams have won the most IPL titles?",
    "Is Sanju Samson in the CSK squad this season?",
]


def secret(name: str):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading the knowledge base (FAISS + BM25) and models...")
def get_kb():
    kb = load_default()
    kb.dense_search("warm up", 1)
    rerank_scores("warm up", ["warm up"])
    return kb


@st.cache_resource(show_spinner="Re-chunking and re-embedding the snapshot with a different strategy (1-3 minutes)...")
def get_kb_variant(chunking: str):
    return build_from_pages(load_pages(), chunking=chunking)


@st.cache_resource
def answer_cache():
    return OrderedDict()


@st.cache_data
def get_manifest():
    return load_manifest() or {}


@st.cache_data
def eval_results():
    out = {}
    for name in ("retrieval", "answers"):
        path = ROOT / "eval" / f"results_{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def cache_get(key):
    cache = answer_cache()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def cache_put(key, value, max_items: int = 300):
    cache = answer_cache()
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


# ---------------------------------------------------------------- state
config = load_config()
params = config.get("params", {})
manifest = get_manifest()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("questions_on_app_key", 0)
ss.setdefault("uploads_kb", None)
ss.setdefault("uploads_names", [])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏏 IPL Expert")
    if manifest:
        st.caption(
            f"Knowledge: **{manifest['num_pages']} Wikipedia pages**, {manifest['num_chunks']} chunks · "
            f"data as of **{manifest['built_at'][:10]}** · latest season **{manifest['latest_season']}**"
        )

    st.subheader("Model")
    providers = [p for p in PROVIDER_MODULES if p == "gemini" or secret(key_env(p))]
    default_provider = config["pipeline"].get("llm", "gemini")
    provider = st.selectbox(
        "Provider", providers, format_func=lambda p: PROVIDER_LABELS.get(p, p),
        index=providers.index(default_provider) if default_provider in providers else 0,
    )
    app_key = secret(key_env(provider))
    key_choice = "own"
    if app_key:
        key_choice = st.radio(
            "API key", ["app", "own"], horizontal=True,
            format_func=lambda c: "App's key" if c == "app" else "My own key",
        )
    user_key = None
    if key_choice == "own":
        user_key = st.text_input(f"Your {PROVIDER_LABELS[provider]} API key", type="password",
                                 help=KEY_HELP.get(provider) + " - used only for this session, never stored.")
    using_app_key = key_choice == "app"
    quota_slot = st.empty()

    st.subheader("Your documents")
    uploaded = st.file_uploader(
        "Ask questions about your own files (kept in memory for this session only)",
        type=["pdf", "docx", "html", "htm", "csv", "txt", "md"], accept_multiple_files=True,
    )
    c1, c2 = st.columns(2)
    if c1.button("Index files", disabled=not uploaded, width="stretch"):
        files = uploaded[:MAX_UPLOAD_FILES]
        docs, names = [], []
        with st.spinner("Chunking and embedding your files..."):
            for f in files:
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(f.getbuffer())
                try:
                    loaded = load_file_as_documents(tmp.name)
                    for d in loaded:
                        d.metadata["filename"] = f.name
                    docs.extend(loaded)
                    names.append(f.name)
                except Exception as exc:
                    st.warning(f"{f.name}: {exc}")
                finally:
                    os.unlink(tmp.name)
            chunks = chunk_uploads(docs)[:MAX_UPLOAD_CHUNKS]
            if chunks:
                ss.uploads_kb = KnowledgeBase.from_documents(chunks, get_embeddings("huggingface"), name="uploads")
                ss.uploads_names = names
        if len(uploaded) > MAX_UPLOAD_FILES:
            st.info(f"Only the first {MAX_UPLOAD_FILES} files were indexed.")
    if c2.button("Remove", disabled=ss.uploads_kb is None, width="stretch"):
        ss.uploads_kb, ss.uploads_names = None, []
    if ss.uploads_kb is not None:
        st.caption(f"Indexed {len(ss.uploads_kb)} chunks from: {', '.join(ss.uploads_names)}")

    with st.expander("Advanced RAG settings"):
        agentic = st.toggle("Agentic mode", value=bool(params.get("use_agentic", True)),
                            help="The agent can run extra searches (up to the limit below). IPL questions are always retrieved first.")
        mode_options = ["hybrid", "dense", "bm25"]
        mode_default = MODE_ALIASES.get(config["pipeline"].get("retrieval", "hybrid"), "hybrid")
        mode = st.selectbox("Retrieval", mode_options, index=mode_options.index(mode_default),
                            format_func={"hybrid": "Hybrid (dense + BM25, RRF)", "dense": "Dense only (embeddings)",
                                         "bm25": "BM25 only (keywords)"}.get)
        use_reranker = st.toggle("Cross-encoder rerank", value=bool(params.get("use_reranker", True)))
        use_filters = st.toggle("Team / season metadata filters", value=bool(params.get("use_filters", True)))
        top_k = st.slider("Chunks per search (top_k)", 2, 12, int(params.get("top_k", 6)))
        token_budget = st.slider("Context token budget", 500, 6000, int(params.get("context_token_budget", 3000)), step=250)
        max_searches = st.slider("Max searches per question", 1, 5, int(params.get("max_searches", 3)))
        chunk_options = list(CHUNKING_STRATEGIES)
        chunking = st.selectbox("Chunking", chunk_options, index=chunk_options.index("structured"),
                                help="'structured' uses the prebuilt index. Other strategies rebuild the index from the snapshot.")
        model_name = st.text_input("Model", value=default_model(provider))
        show_trace = st.toggle("Show 'How this answer was made'", value=True)

        if ADMIN and st.button("Save as defaults (config.yaml)"):
            config["pipeline"].update({"chunking": chunking, "retrieval": mode, "llm": provider})
            config["params"].update({"top_k": top_k, "use_reranker": use_reranker, "use_filters": use_filters,
                                     "use_agentic": agentic, "max_searches": max_searches,
                                     "context_token_budget": token_budget})
            save_config(config)
            st.success("Saved.")

    if st.button("Clear chat", width="stretch"):
        ss.messages = []
        st.rerun()


# ---------------------------------------------------------------- rendering helpers
ROUTE_TEXT = {
    "forced_retrieval": "IPL question detected → retrieval forced before the agent runs",
    "agent_decides": "No IPL entities detected → the agent decided whether to search",
    "always_retrieve": "Classic RAG chain → always retrieve, then generate",
}


def render_sources(sources, cited):
    used = [s for s in sources if s["n"] in cited] or sources
    with st.expander(f"Sources ({len(cited)} cited of {len(sources)} retrieved)"):
        for s in used:
            label = f"**[{s['n']}]** {s['title']} › {s['section']}"
            st.markdown(f"{label} — [open]({s['url']})" if s.get("url") else label)
            body = s["text"].split("\n", 1)[-1]
            st.caption(body[:400] + ("…" if len(body) > 400 else ""))


def render_trace(trace):
    with st.expander("How this answer was made"):
        t = trace.get("timings_ms", {})
        usage = trace.get("llm", {}).get("usage", {})
        c = st.columns(4)
        c[0].metric("Searches", len(trace.get("searches", [])))
        c[1].metric("Total time", f"{(t.get('total') or 0) / 1000:.1f}s")
        c[2].metric("First token", f"{(t.get('time_to_first_token') or 0) / 1000:.1f}s")
        c[3].metric("Tokens in / out", f"{usage.get('input_tokens', 0)} / {usage.get('output_tokens', 0)}")

        a = trace.get("analysis", {})
        lines = [f"- **Route:** {ROUTE_TEXT.get(trace.get('route'), trace.get('route'))}"]
        if trace.get("rewrite") and trace["rewrite"]["to"] != trace["rewrite"]["from"]:
            lines.append(f"- **Follow-up rewritten** ({trace['rewrite']['ms']} ms): “{trace['rewrite']['to']}”")
        lines.append(f"- **Detected teams:** {', '.join(a.get('teams') or []) or 'none'} · "
                     f"**season:** {a.get('season') or 'none'} ({a.get('season_reason', '')})")
        if trace.get("decomposed"):
            lines.append(f"- **Query decomposition:** one filtered search per team ({', '.join(trace['decomposed'])})")
        for tc in trace.get("tool_calls", []):
            lines.append(f"- **Agent tool call:** `{tc['name']}({json.dumps(tc['args'])})`")
        llm = trace.get("llm", {})
        model_line = f"- **Model:** {llm.get('model') or llm.get('requested_model')}"
        if llm.get("model") and llm.get("requested_model") and llm["requested_model"] not in llm["model"]:
            model_line += f" (requested {llm['requested_model']})"
        if usage.get("cache_read_tokens"):
            model_line += f" · {usage['cache_read_tokens']} prompt tokens served from cache"
        lines.append(model_line)
        st.markdown("\n".join(lines))

        for i, s in enumerate(trace.get("searches", []), start=1):
            f = s.get("filter", {})
            if f.get("applied"):
                ftxt = f"filtered to {f['candidates']} of {f['of']} chunks"
            elif f.get("fallback"):
                ftxt = f["fallback"]
            else:
                ftxt = "no filter"
            q = s.get("expanded_query") or s["query"]
            tm = s.get("timings_ms", {})
            st.markdown(
                f"**Search {i}** ({s.get('kb')}) · `{q}` · {ftxt} · "
                f"dense {tm.get('dense', '-')} ms, BM25 {tm.get('bm25', '-')} ms, rerank {tm.get('rerank', '-')} ms · "
                f"{s.get('context_tokens')} context tokens"
            )
            final = s.get("final", [])
            cit = dict(zip(final, s.get("citations", [])))
            rows = []
            for cand in s.get("candidates", [])[:12]:
                rows.append({
                    "cite": f"[{cit[cand['chunk']]}]" if cand["chunk"] in cit else "",
                    "source": f"{cand['title']} › {cand['section']}",
                    "dense #": cand["dense_rank"], "BM25 #": cand["bm25_rank"],
                    "RRF": cand["rrf"], "rerank": cand["rerank"],
                })
            if rows:
                df = pd.DataFrame(rows)
                df[["dense #", "BM25 #"]] = df[["dense #", "BM25 #"]].astype("Int64")
                st.dataframe(df, hide_index=True, width="stretch")


def render_assistant(msg):
    st.markdown(linkify(msg["content"], msg.get("sources", [])))
    if msg.get("cached"):
        st.caption("Served from the answer cache.")
    if msg.get("sources"):
        render_sources(msg["sources"], msg.get("trace", {}).get("cited", []))
    if show_trace and msg.get("trace"):
        render_trace(msg["trace"])


# ---------------------------------------------------------------- main
st.title("🏏 IPL Expert")
st.caption("Ask anything about the Indian Premier League, up to and including the "
           f"{manifest.get('latest_season', 2026)} season. Answers are grounded in Wikipedia and cited.")

chat_tab, how_tab = st.tabs(["Chat", "How it works"])

prompt = st.chat_input("Ask about IPL teams, players, seasons, records...")

with chat_tab:
    if not ss.messages and not prompt and "pending" not in ss:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
                ss.pending = ex
                st.rerun()

    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant(msg)
            else:
                st.markdown(msg["content"])

    question = prompt or ss.pop("pending", None)
    if question:
        question = question.strip()[:MAX_QUESTION_CHARS]
        api_key = app_key if using_app_key else (user_key or None)
        history = [(m["role"], m["content"]) for m in ss.messages if not m.get("error")]
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                st.warning(f"Add a {PROVIDER_LABELS[provider]} API key in the sidebar to ask questions. "
                           f"{KEY_HELP.get(provider, '')}")
                ss.messages.pop()
                st.stop()
            if using_app_key and ss.questions_on_app_key >= QUESTION_CAP:
                st.warning(f"You've used the {QUESTION_CAP} free questions for this session. "
                           "Switch to 'My own key' in the sidebar to keep going.")
                ss.messages.pop()
                st.stop()

            kb = get_kb() if chunking == "structured" else get_kb_variant(chunking)
            settings = SearchSettings(mode=mode, top_k=top_k, fetch_k=int(params.get("fetch_k", 30)),
                                      rrf_k=int(params.get("rrf_k", 60)), use_reranker=use_reranker,
                                      use_filters=use_filters, token_budget=token_budget)
            cache_key = None
            if not history and ss.uploads_kb is None:
                cache_key = json.dumps([question.lower(), provider, model_name, agentic, settings.__dict__,
                                        max_searches, chunking, manifest.get("content_hash")])
            cached = cache_get(cache_key) if cache_key else None

            if cached:
                msg = {**cached, "cached": True}
                render_assistant(msg)
            else:
                ctx = RAGContext(kb=kb, manifest=manifest, provider=provider, api_key=api_key, model=model_name,
                                 settings=settings, agentic=agentic, max_searches=max_searches,
                                 uploads_kb=ss.uploads_kb)
                status = st.status("Thinking...", expanded=False)
                placeholder = st.empty()
                text, final = "", None
                t0 = time.perf_counter()
                for ev in answer_question(question, history, ctx):
                    kind = ev["type"]
                    if kind == "status":
                        status.update(label=ev["text"])
                    elif kind == "search":
                        s = ev["trace"]
                        status.write(f"🔎 `{s.get('expanded_query') or s['query']}` → {len(s['final'])} chunks "
                                     f"({s.get('context_tokens')} tokens)")
                    elif kind == "token":
                        text += ev["text"]
                        placeholder.markdown(text + "▌")
                    elif kind == "reset":
                        text = ""
                        placeholder.empty()
                    else:
                        final = ev
                if final and final["type"] == "done":
                    n = len(final["trace"].get("searches", []))
                    status.update(label=f"Answered in {time.perf_counter() - t0:.1f}s · {n} search{'es' if n != 1 else ''}",
                                  state="complete")
                    placeholder.empty()
                    msg = {"role": "assistant", "content": final["answer"], "sources": final["sources"],
                           "trace": final["trace"]}
                    render_assistant(msg)
                    if using_app_key:
                        ss.questions_on_app_key += 1
                    if cache_key:
                        cache_put(cache_key, msg)
                else:
                    status.update(label="Something went wrong", state="error")
                    err = (final or {}).get("text", "No answer was produced.")
                    placeholder.error(err)
                    msg = {"role": "assistant", "content": err, "error": True}
            ss.messages.append({k: v for k, v in msg.items() if k != "cached"})

if using_app_key:
    quota_slot.caption(f"{max(QUESTION_CAP - ss.questions_on_app_key, 0)} of {QUESTION_CAP} "
                       "free questions left this session.")

with how_tab:
    st.subheader("Pipeline")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=11];
  q [label="Question\\n+ chat history"]; rw [label="Follow-up rewrite\\n(fast LLM, structured output)"];
  qa [label="Query understanding\\nteam aliases · season · expansion"]; route [label="Router\\nIPL question?", shape=diamond];
  dense [label="Dense search\\nbge-small + FAISS"]; bm25 [label="BM25\\nkeyword search"];
  rrf [label="Reciprocal Rank Fusion\\n+ metadata filters"]; rr [label="Cross-encoder rerank\\ndedupe · token budget"];
  agent [label="Agent (create_agent)\\nsearch_ipl tool · ≤3 searches\\nmodel fallback"]; ans [label="Cited answer\\n(streamed)"];
  q -> rw -> qa -> route; route -> dense [label="yes"]; route -> bm25 [label="yes"]; route -> agent [label="no"];
  dense -> rrf; bm25 -> rrf; rrf -> rr -> agent; agent -> dense [label="more searches", style=dashed]; agent -> ans;
}
""")
    st.markdown("""
**Techniques used, and why**

| Stage | Technique | Why it matters |
|---|---|---|
| Ingestion | MediaWiki parse API + table/infobox extraction | Squads, points tables and results live in tables, which plain-text extracts drop |
| Chunking | Structure-aware chunks with breadcrumb headers; tables split by rows with column names repeated | A chunk is findable even if its body never names the team or season (lightweight contextual retrieval) |
| Embeddings | `BAAI/bge-small-en-v1.5` on ONNX Runtime (FastEmbed) | Strong small retriever, no PyTorch, low memory |
| Retrieval | Dense + BM25 fused with Reciprocal Rank Fusion (k=60) | Meaning *and* exact names/numbers, without tuning score scales |
| Query understanding | Team-alias + season detection, alias expansion, follow-up rewriting | "RCB this season" becomes a filter on Royal Challengers Bengaluru + 2026 |
| Precision | Cross-encoder rerank, dedupe, token budget | Best ~6 chunks within ~3k tokens - predictable cost and latency |
| Generation | Forced retrieval for IPL questions, agent for extra searches, numbered citations | No answers from stale model memory; every claim links to a Wikipedia section |
| Reliability | Model fallback middleware, tool-call limit, per-session cap, answer cache | Survives provider overloads and keeps the free tier usable |
""")

    results = eval_results()
    if "retrieval" in results:
        r = results["retrieval"]
        st.subheader("Retrieval evaluation")
        st.caption(f"{r['num_questions']} hand-written questions with known answers · snapshot {r.get('snapshot', '')} · "
                   "Hit@5 = an answer-bearing chunk is in the top 5; MRR = mean reciprocal rank of the first one.")
        st.dataframe(pd.DataFrame(r["results"]), hide_index=True, width="stretch")
    if "answers" in results:
        a = results["answers"]
        st.subheader("Answer evaluation (LLM-as-judge)")
        st.caption(f"{a['num_questions']} questions · judge model {a.get('judge_model')} · {a.get('date', '')}")
        st.dataframe(pd.DataFrame([a["summary"]]), hide_index=True, width="stretch")

    if manifest:
        st.subheader("Knowledge snapshot")
        st.caption(f"Built {manifest['built_at']} · content hash `{manifest['content_hash']}` · "
                   f"embedding model `{manifest['embedding_model']}` · refreshed weekly by GitHub Actions")
        st.dataframe(pd.DataFrame(manifest["pages"]), hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("url")})
