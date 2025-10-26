"""Smoke tests for the standalone setup script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_setup_script_help_executes():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "setup_atlas.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert "Run the ATLAS setup workflow." in result.stdout
