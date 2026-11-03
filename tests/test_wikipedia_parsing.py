from bs4 import BeautifulSoup

from src.chunking.structured import chunk_pages, render_row
from src.loaders.wikipedia import parse_html, parse_infobox, parse_table

HTML = """
<div class="mw-parser-output">
  <table class="infobox"><tbody>
    <tr><th class="infobox-header" colspan="2">Personnel</th></tr>
    <tr><th class="infobox-label">Captain</th><td class="infobox-data">Ruturaj Gaikwad<sup class="reference">[1]</sup></td></tr>
  </tbody></table>
  <p>The 2026 season was Chennai Super Kings' seventeenth appearance.</p>
  <div class="mw-heading mw-heading2"><h2 id="Squad">Squad</h2><span class="mw-editsection">edit</span></div>
  <table class="wikitable"><caption>CSK squad</caption><tbody>
    <tr><th>Name</th><th>Role</th><th>Nationality</th></tr>
    <tr><th colspan="3">Batters</th></tr>
    <tr><td>Ruturaj Gaikwad</td><td rowspan="2">Batter</td><td>India</td></tr>
    <tr><td>Ayush Mhatre</td><td>India</td></tr>
  </tbody></table>
  <div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
  <p>Should be skipped.</p>
</div>
"""


def test_parse_table_expands_spans_and_groups():
    table = BeautifulSoup(HTML, "html.parser").find("table", class_="wikitable")
    block = parse_table(table)
    assert block["headers"] == ["Name", "Role", "Nationality"]
    assert block["rows"][0] == {"g": "Batters", "c": ["Ruturaj Gaikwad", "Batter", "India"]}
    assert block["rows"][1]["c"] == ["Ayush Mhatre", "Batter", "India"]
    assert render_row(block["headers"], block["rows"][1]) == "(Batters) Name: Ayush Mhatre | Role: Batter | Nationality: India"


def test_parse_infobox_uses_labels_and_headers():
    box = parse_infobox(BeautifulSoup(HTML, "html.parser").find("table", class_="infobox"))
    assert box["items"][0][0] == "Personnel - Captain"
    assert box["items"][0][1].startswith("Ruturaj Gaikwad")


def test_parse_html_sections_skip_references_and_strip_noise():
    sections = parse_html(HTML)
    paths = [" > ".join(s["path"]) for s in sections]
    assert paths == ["Overview", "Squad"]
    infobox = sections[0]["blocks"][0]
    assert infobox["items"] == [["Personnel - Captain", "Ruturaj Gaikwad"]]


def test_chunks_carry_contextual_header_and_metadata():
    page = {"title": "2026 Chennai Super Kings season", "url": "https://en.wikipedia.org/wiki/X",
            "revision_timestamp": "2026-10-12T00:00:00Z", "sections": parse_html(HTML)}
    docs = chunk_pages([page])
    table = next(d for d in docs if d.metadata["block_type"] == "table")
    assert table.page_content.startswith("Source: 2026 Chennai Super Kings season > Squad > Table: CSK squad")
    assert "Columns: Name, Role, Nationality" in table.page_content
    assert table.metadata["page_team"] == "Chennai Super Kings"
    assert 2026 in table.metadata["seasons"]
    assert table.metadata["url"].endswith("#Squad")
