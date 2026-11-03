"""
End-to-end answer evaluation with an LLM judge (needs GOOGLE_API_KEY or another provider key).

For each golden question the full pipeline answers; a judge model then grades
  - correct:   does the answer agree with the reference answer?
  - grounded:  is every factual claim supported by the retrieved sources (faithfulness)?
  - cited:     does the answer cite at least one source?

    python scripts/eval_answers.py                 # all questions
    python scripts/eval_answers.py --limit 10      # quick run

Results are written after every question, so a run cut short by quota still leaves a
valid (partial) eval/results_answers.json.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import RAGContext, run_to_completion  # noqa: E402
from src.evaluation import load_golden  # noqa: E402
from src.knowledge import load_default, load_manifest  # noqa: E402
from src.llms import get_chat_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results_answers.json"

JUDGE_PROMPT = """You are grading a retrieval-augmented QA system about the Indian Premier League.

Question: {question}
Reference answer: {reference}

Retrieved sources given to the system:
{sources}

System answer:
{answer}

Grade strictly:
- correct: the system answer states the same key fact(s) as the reference (extra correct detail is fine).
- grounded: every factual claim in the system answer is supported by the retrieved sources above.
"""


class Grade(BaseModel):
    correct: bool = Field(description="Answer agrees with the reference answer")
    grounded: bool = Field(description="All claims are supported by the retrieved sources")
    reason: str = Field(description="One short sentence explaining the grade")


class DailyQuotaExhausted(RuntimeError):
    pass


def _is_rate_limit(text: str) -> bool:
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "rate limit" in text


def with_rate_limit_retry(fn, *args, attempts: int = 4, wait: float = 40.0):
    for i in range(attempts):
        try:
            result = fn(*args)
        except Exception as exc:
            msg = str(exc)
            if "per day" in msg or "PerDay" in msg:
                raise DailyQuotaExhausted(msg[:200]) from exc
            if not _is_rate_limit(msg) or i == attempts - 1:
                raise
            print(f"  rate limited - waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        if isinstance(result, dict) and result.get("type") == "error" and _is_rate_limit(result.get("text", "")) \
                and i < attempts - 1:
            print(f"  rate limited - waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        return result
    return result


def summarize(rows):
    graded = [r for r in rows if "error" not in r]
    n = max(len(graded), 1)
    return {
        "graded": len(graded),
        "correct": f"{sum(r['correct'] for r in graded) / n:.0%}",
        "grounded (faithful)": f"{sum(r['grounded'] for r in graded) / n:.0%}",
        "cited": f"{sum(r['cited'] for r in graded) / n:.0%}",
        "avg latency": f"{sum(r['latency_s'] for r in graded) / n:.1f}s",
        "avg searches": round(sum(r["searches"] for r in graded) / n, 2),
        "avg input tokens": int(sum(r["input_tokens"] for r in graded) / n),
    }


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=13.0, help="Seconds between questions (free tier: 5 requests/min/model)")
    args = ap.parse_args()

    kb, manifest = load_default(), load_manifest() or {}
    items = load_golden()[: args.limit or None]
    judge = get_chat_model(args.provider, model=args.judge_model).with_structured_output(Grade)

    rows, stopped = [], None

    def save():
        RESULTS.write_text(json.dumps({
            "date": date.today().isoformat(), "snapshot": manifest.get("content_hash"),
            "judge_model": args.judge_model, "num_questions": len(items), "stopped_early": stopped,
            "summary": summarize(rows), "rows": rows,
        }, indent=2, ensure_ascii=False) + "\n")

    for it in items:
        try:
            ctx = RAGContext(kb=kb, manifest=manifest, provider=args.provider)
            t = time.perf_counter()
            out = with_rate_limit_retry(run_to_completion, it["question"], [], ctx)
            latency = time.perf_counter() - t
            if out.get("type") != "done":
                if "per day" in out.get("text", ""):
                    raise DailyQuotaExhausted(out["text"])
                rows.append({"id": it["id"], "error": out.get("text")})
                print(f"ERR  {it['id']}: {out.get('text')}")
                continue
            sources = "\n\n".join(f"[{s['n']}] {s['text']}" for s in out["sources"]) or "(none)"
            grade = with_rate_limit_retry(judge.invoke, JUDGE_PROMPT.format(
                question=it["question"], reference=it["reference"], sources=sources[:12000], answer=out["answer"]))
        except DailyQuotaExhausted as exc:
            stopped = f"daily quota exhausted after {len(rows)} questions: {exc}"
            print(f"STOP {stopped}")
            break
        except Exception as exc:
            rows.append({"id": it["id"], "error": str(exc)[:300]})
            print(f"ERR  {it['id']}: {str(exc)[:200]}")
            save()
            continue
        tr = out["trace"]
        rows.append({
            "id": it["id"], "category": it["category"], "correct": grade.correct, "grounded": grade.grounded,
            "cited": bool(tr.get("cited")), "searches": len(tr.get("searches", [])),
            "latency_s": round(latency, 2), "model": tr.get("llm", {}).get("model"),
            "input_tokens": tr.get("llm", {}).get("usage", {}).get("input_tokens", 0),
            "reason": grade.reason, "answer": out["answer"],
        })
        save()
        flag = "OK " if grade.correct and grade.grounded else "BAD"
        print(f"{flag}  {it['id']:28s} correct={grade.correct!s:5} grounded={grade.grounded!s:5} "
              f"{latency:.1f}s  {grade.reason}", flush=True)
        time.sleep(args.pause)

    save()
    print("\n" + json.dumps(summarize(rows), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
