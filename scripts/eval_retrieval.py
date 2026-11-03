"""
Retrieval evaluation: compare retrieval strategies on the golden question set.

    python scripts/eval_retrieval.py                    # print table, write eval/results_retrieval.json
    python scripts/eval_retrieval.py --check            # exit 1 if the full pipeline regresses vs eval/baseline.json
    python scripts/eval_retrieval.py --write-baseline   # record the current full-pipeline scores as the baseline

No API key needed: this measures retrieval only (Hit@k and MRR of the first answer-bearing chunk).
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import first_hit_rank, load_golden, unanswerable  # noqa: E402
from src.knowledge import load_default, load_manifest  # noqa: E402
from src.retrieval import SearchSettings, search  # noqa: E402
from src.retrieval.query import analyze  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results_retrieval.json"
BASELINE = ROOT / "eval" / "baseline.json"
K = 5
TOLERANCE = 0.05

BASE = SearchSettings(top_k=10, token_budget=100_000)
CONFIGS = [
    ("Dense only (bge-small + FAISS)", replace(BASE, mode="dense", use_reranker=False, use_filters=False, expand_query=False)),
    ("BM25 only", replace(BASE, mode="bm25", use_reranker=False, use_filters=False, expand_query=False)),
    ("Hybrid (RRF)", replace(BASE, mode="hybrid", use_reranker=False, use_filters=False, expand_query=False)),
    ("Hybrid + rerank", replace(BASE, mode="hybrid", use_reranker=True, use_filters=False, expand_query=False)),
    ("Hybrid + rerank + query understanding", replace(BASE, mode="hybrid", use_reranker=True, use_filters=True, expand_query=True)),
]
FULL = CONFIGS[-1][0]


def evaluate(kb, items, latest, settings):
    ranks, latency = [], []
    for it in items:
        a = analyze(it["question"], latest)
        teams = a.teams if settings.use_filters else []
        season = a.season if settings.use_filters else None
        t = time.perf_counter()
        hits, _ = search(kb, it["question"], teams, season, settings)
        latency.append((time.perf_counter() - t) * 1000)
        ranks.append(first_hit_rank([h.doc.page_content for h in hits], it))
    n = len(items)
    return {
        "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
        f"hit@{K}": round(sum(1 for r in ranks if r and r <= K) / n, 3),
        "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
        "avg_ms": int(sum(latency) / n),
    }, ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    kb, manifest = load_default(), load_manifest() or {}
    latest = int(manifest.get("latest_season", 2026))
    items = load_golden()

    missing = unanswerable(items, [d.page_content for d in kb.docs])
    if missing:
        print(f"WARNING: {len(missing)} golden answers are no longer in the snapshot (data drift): {missing}")
    items = [it for it in items if it["id"] not in missing]

    rows, per_question = [], {}
    for name, settings in CONFIGS:
        scores, ranks = evaluate(kb, items, latest, settings)
        rows.append({"method": name, **scores})
        per_question[name] = dict(zip([it["id"] for it in items], ranks))
        print(f"{name:42s} " + "  ".join(f"{k}={v}" for k, v in scores.items()))

    misses = [qid for qid, r in per_question[FULL].items() if not r or r > K]
    if misses:
        print(f"Full pipeline misses at @{K}: {misses}")

    RESULTS.write_text(json.dumps({
        "snapshot": manifest.get("content_hash"), "built_at": manifest.get("built_at"),
        "num_questions": len(items), "dropped_for_drift": missing, "k": K,
        "results": rows, "full_pipeline_misses": misses, "ranks": per_question,
    }, indent=2) + "\n")

    print("\n| Method | Hit@1 | Hit@5 | MRR@10 | Avg latency |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['method']} | {r['hit@1']:.0%} | {r[f'hit@{K}']:.0%} | {r['mrr@10']:.3f} | {r['avg_ms']} ms |")

    full = next(r for r in rows if r["method"] == FULL)
    if args.write_baseline:
        BASELINE.write_text(json.dumps({f"hit@{K}": full[f"hit@{K}"], "mrr@10": full["mrr@10"]}, indent=2) + "\n")
        print(f"Wrote baseline to {BASELINE}")
    if args.check:
        base = json.loads(BASELINE.read_text())
        ok = full[f"hit@{K}"] >= base[f"hit@{K}"] - TOLERANCE and full["mrr@10"] >= base["mrr@10"] - TOLERANCE
        print(f"Regression check vs baseline {base}: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
