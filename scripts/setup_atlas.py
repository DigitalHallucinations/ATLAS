#!/usr/bin/env python3
"""Entrypoint for the ATLAS standalone setup utility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ATLAS.setup.cli import SetupUtility
except ModuleNotFoundError:  # pragma: no cover - exercised by smoke test
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ATLAS.setup.cli import SetupUtility


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ATLAS setup workflow.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Run setup without prompts using environment variables or a configuration "
            "file. See ATLAS.setup.cli.SetupUtility for supported inputs."
        ),
    )
    args = parser.parse_args(argv)

    utility = SetupUtility()
    try:
        utility.run(non_interactive=args.non_interactive)
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
