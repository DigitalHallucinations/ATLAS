from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_PATTERN = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
STALE_LAST_AUDITED_DAYS = 365


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backend_summaries: List[str] = field(default_factory=list)

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.backend_summaries.extend(other.backend_summaries)


def _is_relative_to(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
    except ValueError:
        return False
    return True


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ATLAS changes for scope and docs hygiene.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Changed files to inspect. When omitted, staged changes are inspected.",
    )
    parser.add_argument(
        "--base-ref",
        help="Git ref to diff against (uses staged changes when omitted).",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero when only warnings are present.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Inspect all tracked files instead of just changed files.",
    )
    return parser.parse_args(argv)


def _git_lines(command: Sequence[str]) -> List[str]:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_changed_files(
    files: Sequence[str],
    base_ref: Optional[str],
    full_scan: bool,
) -> List[Path]:
    if full_scan:
        return [Path(line) for line in _git_lines(["git", "ls-files"])]

    if files:
        return [Path(file_path) for file_path in files]

    if base_ref:
        diff_target = f"{base_ref}...HEAD"
        diff_command = ["git", "diff", "--name-only", diff_target]
    else:
        diff_command = ["git", "diff", "--name-only", "--cached"]

    return [Path(line) for line in _git_lines(diff_command)]


def discover_scope_paths(repo_root: Path) -> Set[Path]:
    scope_paths: Set[Path] = set()
    for agent_file in repo_root.rglob("AGENTS.md"):
        content = agent_file.read_text(encoding="utf-8")
        for token in re.findall(r"`([^`]+)`", content):
            candidate = token.strip().rstrip("/")
            if not candidate:
                continue
            relative_candidate = (agent_file.parent / candidate).resolve()
            root_candidate = (repo_root / candidate).resolve()
            for option in {relative_candidate, root_candidate}:
                if option.exists():
                    scope_paths.add(option)
    return scope_paths


def unscoped_files(changed_files: Iterable[Path], scope_paths: Set[Path], repo_root: Path) -> List[str]:
    warnings: List[str] = []
    normalized_scope_paths = {path.resolve() for path in scope_paths}
    for file_path in changed_files:
        absolute = (repo_root / file_path).resolve() if not file_path.is_absolute() else file_path.resolve()
        if absolute.name == "AGENTS.md":
            continue
        if any(absolute == scope or _is_relative_to(absolute, scope) for scope in normalized_scope_paths):
            continue
        warnings.append(f"{file_path} is not covered by any declared agent scope")
    return warnings


def _extract_front_matter(path: Path) -> Optional[MutableMapping[str, str]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip().startswith("---"):
        return None

    lines = text.splitlines()
    try:
        start = lines.index("---")
        end = lines.index("---", start + 1)
    except ValueError:
        return None

    block = lines[start + 1 : end]
    data: MutableMapping[str, str] = {}
    for line in block:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _extract_links(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8")
    for match in LINK_PATTERN.finditer(text):
        yield match.group(1).strip()


def _is_checkable_link(link: str) -> bool:
    if link.startswith(("http://", "https://", "mailto:")):
        return False
    if link.startswith("#"):
        return False
    return True


def _link_target_paths(link: str, source: Path, repo_root: Path) -> Tuple[Path, Path]:
    target = link.split("#", 1)[0]
    relative_target = (source.parent / target).resolve()
    root_target = (repo_root / target.lstrip("/")).resolve()
    return relative_target, root_target


def is_doc_file(path: Path) -> bool:
    if path.suffix != ".md":
        return False
    if path.name == "AGENTS.md":
        return False
    parts = path.parts
    return "docs" in parts or "_audit" in parts


def validate_docs(changed_files: Iterable[Path], repo_root: Path) -> ValidationReport:
    report = ValidationReport()
    for file_path in changed_files:
        if not is_doc_file(file_path):
            continue

        absolute = (repo_root / file_path).resolve() if not file_path.is_absolute() else file_path
        if not absolute.exists():
            continue
        front_matter = _extract_front_matter(absolute)
        if front_matter is None:
            report.errors.append(f"{file_path}: missing YAML front matter")
        else:
            if not front_matter.get("source_of_truth"):
                report.warnings.append(f"{file_path}: source_of_truth is missing or empty")

        for link in _extract_links(absolute):
            if not _is_checkable_link(link):
                continue
            relative_target, root_target = _link_target_paths(link, absolute, repo_root)
            if not (relative_target.exists() or root_target.exists()):
                report.errors.append(f"{file_path}: unresolved link '{link}'")
    return report


def _parse_inventory_rows(path: Path) -> Iterable[MutableMapping[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: Optional[List[str]] = None
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if headers is None:
            headers = cells
            continue
        if all(not cell or set(cell) <= {"-"} for cell in cells):
            continue
        if len(cells) != len(headers):
            continue
        yield dict(zip(headers, cells))


def _parse_date_from_cell(value: str) -> Optional[date]:
    match = DATE_PATTERN.search(value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def check_inventories(changed_files: Iterable[Path], repo_root: Path, full_scan: bool) -> List[str]:
    inventory_files: Set[Path] = set()
    if full_scan:
        inventory_files.update(repo_root.glob("**/_audit/inventory.md"))
    for file_path in changed_files:
        if file_path.name == "inventory.md" and "_audit" in file_path.parts:
            absolute = (repo_root / file_path).resolve() if not file_path.is_absolute() else file_path
            inventory_files.add(absolute)

    warnings: List[str] = []
    today = date.today()
    for inventory in sorted(inventory_files):
        if not inventory.exists():
            continue
        front_matter = _extract_front_matter(inventory)
        if not front_matter or not front_matter.get("source_of_truth"):
            warnings.append(f"{inventory.relative_to(repo_root)}: source_of_truth missing from front matter")
        for row in _parse_inventory_rows(inventory):
            last_audited = _parse_date_from_cell(row.get("last_audited", ""))
            next_review = _parse_date_from_cell(row.get("next_review", ""))

            if last_audited and last_audited <= today - timedelta(days=STALE_LAST_AUDITED_DAYS):
                warnings.append(
                    f"{inventory.relative_to(repo_root)}: {row.get('path', 'row')} last_audited "
                    f"{last_audited.isoformat()} is older than {STALE_LAST_AUDITED_DAYS} days",
                )
            if next_review and next_review < today:
                warnings.append(
                    f"{inventory.relative_to(repo_root)}: {row.get('path', 'row')} next_review "
                    f"{next_review.isoformat()} is in the past",
                )
    return warnings


def is_backend_file(path: Path) -> bool:
    relative = path if path.is_absolute() else Path(path)
    top_level = relative.parts[0] if relative.parts else ""
    if top_level in {"ATLAS", "modules", "server"}:
        return True
    if relative.name in {"atlas_provider.py", "main.py"}:
        return True
    return False


def summarize_backend_changes(
    changed_files: Iterable[Path],
    base_ref: Optional[str],
    diff_provider: Optional[Callable[[Path], str]] = None,
) -> List[str]:
    backend_files = [Path(file_path) for file_path in changed_files if is_backend_file(Path(file_path))]
    summaries: List[str] = []
    if not backend_files:
        return summaries

    if diff_provider is None:
        diff_provider = _make_diff_provider(base_ref)

    for file_path in backend_files:
        diff_text = diff_provider(file_path)
        diff_preview = diff_text.strip() if diff_text.strip() else "No diff available"
        summaries.append(f"{file_path}: {diff_preview[:400]}")
    return summaries


def _make_diff_provider(base_ref: Optional[str]) -> Callable[[Path], str]:
    def provider(path: Path) -> str:
        if base_ref:
            command = ["git", "diff", "--unified=3", f"{base_ref}...HEAD", "--", str(path)]
        else:
            command = ["git", "diff", "--unified=3", "--cached", "--", str(path)]
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout

    return provider


def build_report(
    argv: Optional[Sequence[str]] = None,
    args: Optional[argparse.Namespace] = None,
) -> ValidationReport:
    if args is None:
        args = _parse_args(argv)
    changed_files = collect_changed_files(args.files, args.base_ref, args.full_scan)

    scope_paths = discover_scope_paths(REPO_ROOT)
    report = ValidationReport()
    report.warnings.extend(unscoped_files(changed_files, scope_paths, REPO_ROOT))

    inventory_warnings = check_inventories(changed_files, REPO_ROOT, args.full_scan)
    report.warnings.extend(inventory_warnings)

    report.merge(validate_docs(changed_files, REPO_ROOT))
    report.backend_summaries.extend(summarize_backend_changes(changed_files, args.base_ref))

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = build_report(argv, args)
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")
    if report.backend_summaries:
        print("Backend change summaries:")
        for summary in report.backend_summaries:
            print(f"- {summary}")

    if report.errors:
        return 1
    if args.fail_on_warn and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
