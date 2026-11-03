"""
Structured Wikipedia loader.

Uses the MediaWiki parse API (not the plain-text extracts API) so that infoboxes and
wikitables - squads, points tables, results - survive as structured blocks instead of
being dropped. Each page becomes:

    {"title", "url", "pageid", "revid", "revision_timestamp",
     "sections": [{"path": [...], "anchor", "blocks": [text | table | infobox]}]}
"""
from __future__ import annotations

import re
import time
from typing import Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_core.documents import Document

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikimedia rejects (HTTP 429) clients that don't send a descriptive User-Agent.
USER_AGENT = "ipl-rag-system/2.0 (https://github.com/sniper7jd/ipl_rag_system)"

SKIP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "footnotes", "citations", "bibliography", "sources", "notes and references",
}
SKIP_CLASSES = {
    "navbox", "sidebar", "metadata", "ambox", "hatnote", "reflist", "refbegin",
    "shortdescription", "thumb", "gallery", "noprint", "mw-references-wrap",
    "sistersitebox", "side-box", "portalbox", "toc", "mw-empty-elt",
}

_session: Optional[requests.Session] = None


def _http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def _get(params: Dict, retries: int = 4) -> Dict:
    params = {"format": "json", "formatversion": 2, **params}
    for attempt in range(retries):
        resp = _http().get(API_URL, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?)\]%])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def _node_text(node: Tag) -> str:
    return clean_text(node.get_text(" ", strip=True))


def _strip_noise(root: Tag) -> None:
    for sel in ("sup.reference", "span.mw-editsection", "style", "link", "meta",
                "span.mw-cite-backlink", "sup.noprint", ".sortkey", "img", ".navbar"):
        for el in root.select(sel):
            el.decompose()
    for el in root.select("[style]"):
        if "display:none" in el.get("style", "").replace(" ", ""):
            el.decompose()


def _has_skip_class(el: Tag) -> bool:
    return bool(SKIP_CLASSES.intersection(el.get("class", [])))


def _expand_grid(table: Tag) -> List[List[Tag]]:
    """Expand rowspan/colspan into a rectangular grid of cell tags."""
    grid: List[List[Optional[Tag]]] = []
    pending: Dict[int, List] = {}  # col -> [cell, remaining_rows]
    body_rows = []
    for section in table.find_all(["thead", "tbody", "tfoot"], recursive=False) or [table]:
        body_rows.extend(section.find_all("tr", recursive=False))
    for tr in body_rows:
        row: List[Optional[Tag]] = []
        col = 0
        cells = tr.find_all(["td", "th"], recursive=False)
        ci = 0
        while ci < len(cells) or col in pending:
            if col in pending:
                cell, remaining = pending[col]
                row.append(cell)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col][1] = remaining - 1
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            colspan = min(int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1), 30)
            rowspan = min(int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1), 200)
            for _ in range(colspan):
                row.append(cell)
                if rowspan > 1:
                    pending[col] = [cell, rowspan - 1]
                col += 1
        grid.append(row)
    return grid


def parse_table(table: Tag) -> Optional[Dict]:
    caption_el = table.find("caption")
    caption = _node_text(caption_el) if caption_el else ""
    grid = _expand_grid(table)
    if not grid:
        return None

    header_rows = []
    while grid and all(c is not None and c.name == "th" for c in grid[0]) and len(set(map(id, grid[0]))) > 1:
        header_rows.append(grid.pop(0))
    width = max((len(r) for r in grid + header_rows), default=0)
    headers: List[str] = []
    for col in range(width):
        parts: List[str] = []
        for hr in header_rows:
            if col < len(hr):
                t = _node_text(hr[col])
                if t and t not in parts:
                    parts.append(t)
        headers.append(" - ".join(parts) or f"Column {col + 1}")

    rows = []
    group = None
    for r in grid:
        if not r:
            continue
        distinct = list(dict.fromkeys(map(id, r)))
        if len(distinct) == 1:
            label = _node_text(r[0])
            if label:
                group = label
            continue
        cells, seen = [], set()
        for c in r:
            # A merged (colspan) cell should only be emitted once per row.
            cells.append("" if id(c) in seen else _node_text(c))
            seen.add(id(c))
        if any(cells):
            rows.append({"g": group, "c": cells})
    if not rows:
        return None
    return {"type": "table", "caption": caption, "headers": headers, "rows": rows}


def parse_infobox(table: Tag) -> Optional[Dict]:
    items = []
    group = None
    for tr in table.find_all("tr"):
        header = tr.find(class_="infobox-header")
        if header:
            group = _node_text(header) or group
            continue
        label, data = tr.find(class_="infobox-label"), tr.find(class_="infobox-data")
        if label and data:
            label_text, value = _node_text(label), _node_text(data)
            if label_text and value:
                items.append([f"{group} - {label_text}" if group else label_text, value])
        elif data and data.get("colspan"):
            text = _node_text(data)
            if text and len(text) > 3:
                items.append([group or "Info", text])
    if not items:
        return None
    return {"type": "infobox", "items": items}


def _iter_content(root: Tag) -> Iterable[Tag]:
    """Yield top-level content elements, descending into plain wrapper divs."""
    for child in root.children:
        if isinstance(child, NavigableString) or not isinstance(child, Tag):
            continue
        if child.name == "div" and not _has_skip_class(child) and "mw-heading" not in child.get("class", []):
            if child.find("table", class_="wikitable") or child.find(class_="mw-heading"):
                yield from _iter_content(child)
                continue
        yield child


def parse_html(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    _strip_noise(root)

    sections: List[Dict] = []
    current = {"path": ["Overview"], "anchor": "", "blocks": []}
    path: List[str] = []
    skipping = False

    def flush():
        if current["blocks"]:
            sections.append(dict(current))

    for el in _iter_content(root):
        classes = el.get("class", [])
        heading = None
        if el.name == "div" and "mw-heading" in classes:
            heading = el.find(re.compile(r"^h[2-6]$"))
        elif re.fullmatch(r"h[2-6]", el.name or ""):
            heading = el
        if heading is not None:
            level = int(heading.name[1])
            title = _node_text(heading)
            flush()
            path = path[: level - 2] + [title]
            skipping = any(p.lower() in SKIP_SECTIONS for p in path)
            current = {"path": list(path), "anchor": heading.get("id", ""), "blocks": []}
            continue
        if skipping or _has_skip_class(el):
            continue

        if el.name == "table":
            if "infobox" in classes:
                block = parse_infobox(el)
            elif "wikitable" in classes:
                block = parse_table(el)
            else:
                text = _node_text(el)
                block = {"type": "text", "text": text} if len(text) > 40 else None
            if block:
                current["blocks"].append(block)
            continue

        if el.name in ("p", "ul", "ol", "dl", "blockquote", "div"):
            if el.name in ("ul", "ol"):
                items = [_node_text(li) for li in el.find_all("li", recursive=False)]
                text = "\n".join(f"- {t}" for t in items if t)
            else:
                text = _node_text(el)
            if text and not text.startswith("^"):
                current["blocks"].append({"type": "text", "text": text})
    flush()
    return sections


def _page_meta(titles: List[str]) -> Dict[str, Dict]:
    data = _get({
        "action": "query", "prop": "revisions|info", "rvprop": "timestamp|ids",
        "inprop": "url", "titles": "|".join(titles), "redirects": 1,
    })
    redirects = {r["from"]: r["to"] for r in data["query"].get("redirects", [])}
    normalized = {n["from"]: n["to"] for n in data["query"].get("normalized", [])}
    by_title = {}
    for p in data["query"]["pages"]:
        if p.get("missing"):
            continue
        rev = p["revisions"][0]
        by_title[p["title"]] = {
            "pageid": p["pageid"], "revid": rev["revid"],
            "revision_timestamp": rev["timestamp"], "url": p["fullurl"],
        }
    out = {}
    for t in titles:
        resolved = redirects.get(normalized.get(t, t), normalized.get(t, t))
        if resolved in by_title:
            out[t] = {"title": resolved, **by_title[resolved]}
    return out


def fetch_pages(titles: List[str], pause: float = 0.3) -> List[Dict]:
    """Fetch and parse Wikipedia pages into structured section/block dicts."""
    meta: Dict[str, Dict] = {}
    for i in range(0, len(titles), 20):
        meta.update(_page_meta(titles[i:i + 20]))

    pages = []
    for requested in titles:
        if requested not in meta:
            print(f"  ! missing page: {requested}")
            continue
        m = meta[requested]
        parsed = _get({"action": "parse", "pageid": m["pageid"], "prop": "text"})
        sections = parse_html(parsed["parse"]["text"])
        pages.append({**m, "requested_title": requested, "sections": sections})
        time.sleep(pause)
    return pages


def page_to_text(page: Dict) -> str:
    """Flatten a structured page into plain text (used by the non-structured chunkers)."""
    from src.chunking.structured import render_block

    parts = []
    for sec in page["sections"]:
        parts.append("== " + " > ".join(sec["path"]) + " ==")
        parts.extend(render_block(b) for b in sec["blocks"])
    return "\n\n".join(parts)


def fetch_team_data(teams: List[str]) -> List[Document]:
    """Backwards-compatible helper: one Document per page with flattened text."""
    return [
        Document(page_content=page_to_text(p), metadata={"source": p["url"], "title": p["title"]})
        for p in fetch_pages(teams)
    ]
