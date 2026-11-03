"""Shared helpers for retrieval/answer evaluation against eval/golden_2026.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_2026.yaml"


def load_golden(path: Path = GOLDEN) -> List[Dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def is_hit(text: str, item: Dict) -> bool:
    low = text.lower()
    return any(a.lower() in low for a in item["answers"]) and all(a.lower() in low for a in item.get("all_of", []))


def first_hit_rank(texts: Sequence[str], item: Dict) -> Optional[int]:
    for rank, t in enumerate(texts, start=1):
        if is_hit(t, item):
            return rank
    return None


def unanswerable(items: List[Dict], corpus: Sequence[str]) -> List[str]:
    """Golden items whose answer no longer appears anywhere in the corpus (data drift)."""
    return [it["id"] for it in items if not any(is_hit(t, it) for t in corpus)]
