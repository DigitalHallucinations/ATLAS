#!/usr/bin/env python3
"""Prepare the Python environment required by ATLAS."""

from __future__ import annotations

import argparse
import os
import subprocess
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
    parser.add_argument(
        "--with-accelerators",
        action="store_true",
        help="Install optional Hugging Face fine-tuning dependencies.",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    venv_path = ensure_virtualenv(project_root, python_executable=args.python_executable)
    if args.with_accelerators:
        pip = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
        accelerator_requirements = project_root / "requirements-accelerators.txt"
        if accelerator_requirements.exists():
            print("Installing accelerator dependencies…")
            subprocess.run(
                [str(pip), "install", "-r", str(accelerator_requirements)],
                check=True,
            )
        else:
            print(
                "Accelerator requirements file not found; "
                "skipping optional dependency installation."
            )
    print(f"Virtual environment ready at {venv_path}")
    return 0


def ensure_virtualenv(project_root: Path, python_executable: str | None = None) -> Path:
    """Create or update the project's virtual environment and base dependencies."""

    _bootstrap_sys_path()

    venv_path = project_root / ".venv"
    python_executable = python_executable or sys.executable

    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}…")
        subprocess.run([python_executable, "-m", "venv", str(venv_path)], check=True)

    pip = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
    requirements = project_root / "requirements.txt"
    if requirements.exists():
        print("Installing Python requirements…")
        subprocess.run([str(pip), "install", "-r", str(requirements)], check=True)
    else:
        print("No requirements.txt found; skipping dependency installation.")

    return venv_path


if __name__ == "__main__":
    raise SystemExit(main())
