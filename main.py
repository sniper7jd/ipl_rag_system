"""
Command-line chat over the IPL knowledge snapshot (same pipeline as the web app).

    python main.py                      # interactive
    python main.py "Who won IPL 2026?"  # one-shot
"""
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents import RAGContext, answer_question  # noqa: E402
from src.knowledge import load_default, load_manifest  # noqa: E402
from src.llms import key_env  # noqa: E402
from src.retrieval import MODE_ALIASES, SearchSettings  # noqa: E402
from utils.helpers import load_config  # noqa: E402

load_dotenv()


def main():
    config = load_config()
    pipeline, params = config["pipeline"], config.get("params", {})
    provider = pipeline["llm"]
    if not os.getenv(key_env(provider)):
        sys.exit(f"Set {key_env(provider)} in .env to use the '{provider}' provider.")

    print("Loading knowledge base...")
    ctx = RAGContext(
        kb=load_default(),
        manifest=load_manifest() or {},
        provider=provider,
        settings=SearchSettings(
            mode=MODE_ALIASES.get(pipeline.get("retrieval", "hybrid"), "hybrid"),
            top_k=params.get("top_k", 6),
            fetch_k=params.get("fetch_k", 30),
            use_reranker=params.get("use_reranker", True),
            use_filters=params.get("use_filters", True),
            token_budget=params.get("context_token_budget", 3000),
        ),
        agentic=params.get("use_agentic", True),
        max_searches=params.get("max_searches", 3),
    )

    history = []
    one_shot = " ".join(sys.argv[1:]).strip()
    while True:
        question = one_shot or input("\nAsk about the IPL (empty to quit): ").strip()
        if not question:
            break
        answer, sources = "", []
        for ev in answer_question(question, history, ctx):
            if ev["type"] == "search":
                print(f"  [search] {ev['trace'].get('expanded_query') or ev['trace']['query']}")
            elif ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                answer, sources = ev["answer"], ev["sources"]
                cited = set(ev["trace"].get("cited", []))
                print("\n\nSources:")
                for s in sources:
                    if s["n"] in cited:
                        print(f"  [{s['n']}] {s['title']} > {s['section']}  {s.get('url') or ''}")
            elif ev["type"] == "error":
                print(f"\nError: {ev['text']}")
        history += [("user", question), ("assistant", answer)]
        if one_shot:
            break


if __name__ == "__main__":
    main()
