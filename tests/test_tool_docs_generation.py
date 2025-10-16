from pathlib import Path

import scripts.generate_tool_docs as tool_docs
from modules.Tools.manifest_loader import load_manifest_entries

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def test_generated_docs_match_snapshots():
    entries = load_manifest_entries()
    markdown = tool_docs.render_markdown(entries)
    html_doc = tool_docs.render_html(entries)

    expected_markdown = (SNAPSHOT_DIR / "tool_manifest.md").read_text(encoding="utf-8")
    expected_html = (SNAPSHOT_DIR / "tool_manifest.html").read_text(encoding="utf-8")

    assert markdown == expected_markdown
    assert html_doc == expected_html
