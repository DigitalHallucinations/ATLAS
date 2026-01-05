from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from core.utils import change_validation


def test_unscoped_files_flag_out_of_scope(tmp_path: Path) -> None:
    agent_file = tmp_path / "AGENTS.md"
    agent_file.write_text("- **Docs Agent**: Works in `docs/`\n", encoding="utf-8")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    in_scope = docs_dir / "page.md"
    in_scope.write_text("---\nsource_of_truth: docs/_audit/style-guide.md\n---\n", encoding="utf-8")

    out_of_scope = tmp_path / "misc.txt"
    out_of_scope.write_text("hello\n", encoding="utf-8")

    scopes = change_validation.discover_scope_paths(tmp_path)
    warnings = change_validation.unscoped_files(
        [
            in_scope.relative_to(tmp_path),
            out_of_scope.relative_to(tmp_path),
        ],
        scopes,
        tmp_path,
    )

    assert any("misc.txt" in warning for warning in warnings)
    assert not any("page.md" in warning for warning in warnings)


def test_inventory_staleness_detection(tmp_path: Path) -> None:
    audit_dir = tmp_path / "docs" / "_audit"
    audit_dir.mkdir(parents=True)
    stale_date = (date.today() - timedelta(days=change_validation.STALE_LAST_AUDITED_DAYS + 1)).isoformat()
    past_review = (date.today() - timedelta(days=1)).isoformat()

    inventory = audit_dir / "inventory.md"
    inventory.write_text(
        f"""---
audience: test
status: active
last_verified: {date.today().isoformat()}
source_of_truth: ./style-guide.md
---

| path | owner | last_audited | audit_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| docs/example.md | @docs | {stale_date} | Needs review | none | {past_review} | |
""",
        encoding="utf-8",
    )

    warnings = change_validation.check_inventories(
        [inventory.relative_to(tmp_path)],
        tmp_path,
        full_scan=False,
    )

    assert any("last_audited" in warning for warning in warnings)
    assert any("next_review" in warning for warning in warnings)


def test_validate_docs_checks_front_matter_and_links(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    missing_front_matter = docs_dir / "missing.md"
    missing_front_matter.write_text("No front matter\n", encoding="utf-8")

    broken_link = docs_dir / "broken.md"
    broken_link.write_text(
        """---
source_of_truth: modules/example.py
---

See [nowhere](./nope.md) for details.
""",
        encoding="utf-8",
    )

    report = change_validation.validate_docs(
        [
            missing_front_matter.relative_to(tmp_path),
            broken_link.relative_to(tmp_path),
        ],
        tmp_path,
    )

    assert any("missing.md" in error for error in report.errors)
    assert any("unresolved link" in error for error in report.errors)


def test_backend_summaries_use_diff_provider(tmp_path: Path) -> None:
    backend_dir = tmp_path / "ATLAS"
    backend_dir.mkdir()
    backend_file = backend_dir / "module.py"
    backend_file.write_text("print('hello')\n", encoding="utf-8")

    summaries = change_validation.summarize_backend_changes(
        [backend_file.relative_to(tmp_path)],
        base_ref=None,
        diff_provider=lambda path: f"diff for {path.name}",
    )

    assert summaries == [f"{backend_file.relative_to(tmp_path)}: diff for module.py"]
