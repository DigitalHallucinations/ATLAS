"""Command line helpers for exporting and importing persona bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from modules.Personas import (
    PersonaBundleError,
    export_persona_bundle_bytes,
    import_persona_bundle_bytes,
)


class _ConfigAdapter:
    """Lightweight adapter exposing ``get_app_root`` for persona helpers."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persona bundle utilities")
    parser.add_argument(
        "--app-root",
        dest="app_root",
        type=Path,
        help="Override the application root when loading personas.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a persona to a signed bundle")
    export_parser.add_argument("persona", help="Persona name to export")
    export_parser.add_argument("output", type=Path, help="Destination bundle path")
    _add_signing_key_arguments(export_parser)

    import_parser = subparsers.add_parser("import", help="Import a signed persona bundle")
    import_parser.add_argument("bundle", type=Path, help="Bundle file to import")
    import_parser.add_argument(
        "--rationale",
        default="Imported via CLI persona tools",
        help="Audit rationale recorded with the import.",
    )
    _add_signing_key_arguments(import_parser)

    return parser


def _add_signing_key_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signing-key", help="Signing key material provided directly")
    group.add_argument(
        "--signing-key-file",
        type=Path,
        help="Path to a file containing the signing key",
    )


def _load_signing_key(args: argparse.Namespace) -> str:
    if getattr(args, "signing_key", None):
        return str(args.signing_key)
    key_path: Optional[Path] = getattr(args, "signing_key_file", None)
    if key_path is None:
        raise PersonaBundleError("Signing key or signing key file must be provided.")
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - surfaced to CLI user
        raise PersonaBundleError(f"Failed to read signing key file: {key_path}") from exc


def _config_from_args(args: argparse.Namespace) -> Optional[_ConfigAdapter]:
    root: Optional[Path] = getattr(args, "app_root", None)
    return _ConfigAdapter(root) if root else None


def _cmd_export(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_bytes, persona = export_persona_bundle_bytes(
        args.persona,
        signing_key=signing_key,
        config_manager=config,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bundle_bytes)

    print(f"Exported persona '{persona.get('name', args.persona)}' to {output_path}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    signing_key = _load_signing_key(args)
    config = _config_from_args(args)
    bundle_path: Path = args.bundle
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise PersonaBundleError(f"Failed to read bundle file: {bundle_path}") from exc

    result = import_persona_bundle_bytes(
        bundle_bytes,
        signing_key=signing_key,
        config_manager=config,
        rationale=args.rationale,
    )

    persona = result.get("persona", {})
    persona_name = persona.get("name") or "unknown"
    print(f"Imported persona '{persona_name}' from {bundle_path}")

    warnings = result.get("warnings") or []
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "export":
            return _cmd_export(args)
        if args.command == "import":
            return _cmd_import(args)
    except PersonaBundleError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
