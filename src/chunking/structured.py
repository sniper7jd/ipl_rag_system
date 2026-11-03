"""
Structure-aware chunking with contextual headers.

- Text is split per Wikipedia section, and every chunk starts with its breadcrumb
  ("2026 Chennai Super Kings season > Squad") so it is retrievable even when the chunk
  body never names the team or season (a lightweight form of "contextual retrieval").
- Tables are split into row groups and every chunk repeats the column names, so a slice
  of a squad list or points table is still self-describing.
- Infoboxes become "Label: value" chunks - they hold dense facts (captain, coach, titles).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retrieval.query import detect_teams, detect_years

CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def render_row(headers: List[str], row: Dict) -> str:
    pairs = []
    for h, v in zip(headers, row["c"]):
        if v and v != h:
            pairs.append(f"{h}: {v}" if not h.startswith("Column ") else v)
    text = " | ".join(pairs)
    return f"({row['g']}) {text}" if row.get("g") else text


def render_block(block: Dict) -> str:
    if block["type"] == "text":
        return block["text"]
    if block["type"] == "infobox":
        return "\n".join(f"{k}: {v}" for k, v in block["items"])
    lines = [f"Table: {block['caption']}" if block.get("caption") else "Table"]
    lines += [render_row(block["headers"], r) for r in block["rows"]]
    return "\n".join(lines)


def _page_season(title: str) -> Optional[int]:
    m = re.match(r"^(20\d\d)\b", title)
    return int(m.group(1)) if m else None


def _page_kind(title: str) -> str:
    if re.match(r"^20\d\d .+ season$", title):
        return "team_season"
    if re.match(r"^20\d\d Indian Premier League$", title):
        return "league_season"
    return "team" if detect_teams(title) and "Indian Premier League" not in title else "league"


def _make_doc(text: str, header: str, meta: Dict, block_type: str) -> Document:
    content = f"{header}\n{text}".strip()
    page_teams = [meta["page_team"]] if meta.get("page_team") else []
    teams = list(dict.fromkeys(page_teams + detect_teams(content)))
    seasons = sorted(set(detect_years(content)) | ({meta["page_season"]} if meta.get("page_season") else set()))
    return Document(
        page_content=content,
        metadata={**meta, "block_type": block_type, "teams": teams, "seasons": seasons,
                  "tokens": approx_tokens(content)},
    )


def _flush_text(text_buf: List[str], splitter, header: str, meta: Dict, docs: List[Document]) -> None:
    if text_buf:
        for piece in splitter.split_text("\n\n".join(text_buf)):
            docs.append(_make_doc(piece, header, meta, "text"))
        text_buf.clear()


def chunk_pages(pages: Iterable[Dict], chunk_tokens: int = 400, overlap_tokens: int = 50) -> List[Document]:
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars, chunk_overlap=overlap_tokens * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: List[Document] = []
    for page in pages:
        title = page["title"]
        page_teams = detect_teams(title)
        base = {
            "title": title,
            "page_url": page["url"],
            "page_kind": _page_kind(title),
            "page_team": page_teams[0] if page_teams and _page_kind(title) in ("team", "team_season") else None,
            "page_season": _page_season(title),
            "revision_timestamp": page.get("revision_timestamp"),
        }
        for sec in page["sections"]:
            path = " > ".join(sec["path"])
            header = f"Source: {title} > {path}"
            url = page["url"] + (f"#{sec['anchor']}" if sec.get("anchor") else "")
            meta = {**base, "section": path, "url": url}

            text_buf: List[str] = []
            for block in sec["blocks"]:
                if block["type"] == "text":
                    text_buf.append(block["text"])
                    continue
                _flush_text(text_buf, splitter, header, meta, docs)
                if block["type"] == "infobox":
                    body = render_block(block)
                    for piece in splitter.split_text(body):
                        docs.append(_make_doc(piece, f"{header} > Infobox", meta, "infobox"))
                    continue
                table_header = f"{header} > Table: {block.get('caption') or 'untitled'}\nColumns: {', '.join(block['headers'])}"
                budget = max_chars - len(table_header)
                group: List[str] = []
                size = 0
                for row in block["rows"]:
                    line = render_row(block["headers"], row)[: budget]
                    if group and size + len(line) > budget:
                        docs.append(_make_doc("\n".join(group), table_header, meta, "table"))
                        group, size = [], 0
                    group.append(line)
                    size += len(line) + 1
                if group:
                    docs.append(_make_doc("\n".join(group), table_header, meta, "table"))
            _flush_text(text_buf, splitter, header, meta, docs)

    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i
    return docs


def process(documents: List[Document], chunk_size: int = 1600, chunk_overlap: int = 200, **_kwargs) -> List[Document]:
    """Generic documents (uploads): recursive split plus a contextual source header."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: List[Document] = []
    for doc in documents:
        name = doc.metadata.get("filename") or doc.metadata.get("title") or doc.metadata.get("source", "document")
        page = doc.metadata.get("page")
        header = f"Source: {name}" + (f" (page {int(page) + 1})" if isinstance(page, (int, float)) else "")
        for piece in splitter.split_text(doc.page_content):
            meta = {**doc.metadata, "title": name, "section": header.replace("Source: ", ""),
                    "url": None, "block_type": "upload"}
            out.append(_make_doc(piece, header, meta, "upload"))
    for i, d in enumerate(out):
        d.metadata["chunk_id"] = i
    return out
