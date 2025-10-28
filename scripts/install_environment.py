#!/usr/bin/env python3
"""Prepare the Python environment required by ATLAS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_sys_path() -> None:
    """Ensure the repository root is importable."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create/update the ATLAS virtualenv and install dependencies.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Path to the Python interpreter that should own the virtualenv.",
    )
    args = parser.parse_args(argv)

    _bootstrap_sys_path()

    from ATLAS.setup.cli import SetupUtility

    utility = SetupUtility()
    project_root = Path(__file__).resolve().parents[1]
    venv_path = utility.ensure_virtualenv(project_root, python_executable=args.python_executable)
    print(f"Virtual environment ready at {venv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
