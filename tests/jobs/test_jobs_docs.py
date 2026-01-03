from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS_DOCS = ROOT / "docs" / "jobs"


@pytest.mark.parametrize(
    "filename", [
        "index.md",
        "manifest.md",
        "lifecycle.md",
        "api.md",
        "scheduling.md",
        "ui.md",
    ],
)
def test_jobs_docs_exist_with_headings(filename):
    path = JOBS_DOCS / filename
    assert path.exists(), f"Missing jobs doc: {filename}"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# "), f"Document {filename} should start with a heading"


def test_tasks_overview_references_job_guides():
    tasks_doc = ROOT / "docs" / "tasks" / "overview.md"
    text = tasks_doc.read_text(encoding="utf-8")
    assert "../jobs/lifecycle.md" in text
    assert "jobs.metrics.lifecycle" in text
    assert "../jobs/ui.md" in text
