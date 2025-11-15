"""Generate markdown and HTML documentation for registered tools."""

from __future__ import annotations

import argparse
import html
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    __import__("yaml")
except (ModuleNotFoundError, ImportError):
    if "yaml" not in sys.modules:
        sys.modules["yaml"] = SimpleNamespace(
            safe_load=lambda *_args, **_kwargs: {},
            dump=lambda *_args, **_kwargs: None,
        )

from modules.Tools.manifest_loader import ToolManifestEntry, load_manifest_entries


def render_markdown(entries: Iterable[ToolManifestEntry]) -> str:
    grouped = _group_entries(entries)
    lines: List[str] = ["# Tool Manifest", ""]

    for persona, persona_entries in grouped:
        lines.append(f"## Persona: {persona}")
        lines.append("")
        lines.append("| Name | Version | Capabilities | Safety Level | Auth Required | Description |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in persona_entries:
            capabilities = ", ".join(entry.capabilities) if entry.capabilities else "—"
            safety = entry.safety_level or "—"
            auth = _format_auth(entry)
            description = entry.description.replace("|", "\\|") or "—"
            version = entry.version or "—"
            lines.append(
                f"| {entry.name} | {version} | {capabilities} | {safety} | {auth} | {description} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_html(entries: Iterable[ToolManifestEntry]) -> str:
    grouped = _group_entries(entries)
    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        "  <title>ATLAS Tool Manifest</title>",
        "  <style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #ccc; padding: 0.5em; text-align: left;} th {background: #f5f5f5;} caption {text-align: left; font-weight: bold; margin: 1em 0 0.5em;}</style>",
        "</head>",
        "<body>",
        "  <h1>Tool Manifest</h1>",
    ]

    for persona, persona_entries in grouped:
        parts.append(f"  <h2>Persona: {html.escape(persona)}" + "</h2>")
        parts.append("  <table>")
        parts.append("    <thead>")
        parts.append(
            "      <tr><th>Name</th><th>Version</th><th>Capabilities</th><th>Safety Level</th><th>Auth Required</th><th>Description</th></tr>"
        )
        parts.append("    </thead>")
        parts.append("    <tbody>")
        for entry in persona_entries:
            capabilities = ", ".join(entry.capabilities) if entry.capabilities else "—"
            safety = entry.safety_level or "—"
            auth = _format_auth(entry)
            description = entry.description or "—"
            version = entry.version or "—"
            parts.append(
                "      <tr>"
                f"<td>{html.escape(entry.name)}</td>"
                f"<td>{html.escape(version)}</td>"
                f"<td>{html.escape(capabilities)}</td>"
                f"<td>{html.escape(safety)}</td>"
                f"<td>{html.escape(auth)}</td>"
                f"<td>{html.escape(description)}</td>"
                "</tr>"
            )
        parts.append("    </tbody>")
        parts.append("  </table>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts) + "\n"


def _group_entries(entries: Iterable[ToolManifestEntry]) -> List[Tuple[str, List[ToolManifestEntry]]]:
    buckets: Dict[str, List[ToolManifestEntry]] = {}
    for entry in entries:
        key = entry.persona or "Shared"
        buckets.setdefault(key, []).append(entry)

    ordered: List[Tuple[str, List[ToolManifestEntry]]] = []
    for persona in sorted(buckets.keys()):
        persona_entries = sorted(buckets[persona], key=lambda item: item.name.lower())
        ordered.append((persona, persona_entries))
    return ordered


def _format_auth(entry: ToolManifestEntry) -> str:
    auth = dict(entry.auth)
    required = "Yes" if entry.auth_required else "No"
    details = ", ".join(
        f"{key}: {value}" for key, value in sorted(auth.items()) if key != "required"
    )
    if details:
        return f"{required} ({details})"
    return required


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate tool manifest documentation.")
    default_output = Path(__file__).resolve().parents[1] / "docs" / "generated"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write the generated documentation.",
    )
    args = parser.parse_args(argv)

    entries = load_manifest_entries()
    markdown = render_markdown(entries)
    html_doc = render_html(entries)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "tools.md"
    html_path = output_dir / "tools.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
