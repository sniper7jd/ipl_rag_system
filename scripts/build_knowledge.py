"""
Fetch the IPL Wikipedia pages, chunk them, embed them and write the knowledge snapshot.

    python scripts/build_knowledge.py            # rebuild only if Wikipedia content changed
    python scripts/build_knowledge.py --force    # always rebuild
    python scripts/build_knowledge.py --offline  # re-chunk/re-embed the committed pages.jsonl

Outputs (in knowledge/): pages.jsonl, chunks.jsonl, faiss.index, manifest.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge.build import (  # noqa: E402
    KNOWLEDGE_DIR, build_from_pages, content_hash, load_manifest, load_pages,
    page_titles, save_pages, write_manifest,
)
from src.loaders.wikipedia import fetch_pages  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026, help="Latest IPL season to include")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    if args.offline:
        pages = load_pages()
        print(f"Loaded {len(pages)} pages from {KNOWLEDGE_DIR / 'pages.jsonl'}")
    else:
        titles = page_titles(args.season)
        print(f"Fetching {len(titles)} Wikipedia pages...")
        pages = fetch_pages(titles)
        print(f"  fetched {len(pages)} pages in {time.time() - t0:.1f}s")

    manifest = load_manifest()
    new_hash = content_hash(pages)
    if manifest and manifest.get("content_hash") == new_hash and not (args.force or args.offline):
        print(f"No content changes (hash {new_hash}); snapshot is up to date.")
        return 0

    t1 = time.time()
    kb = build_from_pages(pages)
    print(f"Chunked + embedded {len(kb)} chunks in {time.time() - t1:.1f}s")

    if not args.offline:
        save_pages(pages)
    kb.save(KNOWLEDGE_DIR)
    manifest = write_manifest(pages, kb, latest_season=args.season)
    print(f"Wrote snapshot {manifest['content_hash']} ({manifest['num_chunks']} chunks) to {KNOWLEDGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
