from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import subprocess
import sys

MODULE_PATH = Path(__file__).resolve().parent.parent / "modules/Tools/Base_Tools/labor_market_feed.py"

spec = importlib.util.spec_from_file_location("labor_market_feed", MODULE_PATH)
assert spec and spec.loader
labor_market_feed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labor_market_feed)
_score_signal = labor_market_feed._score_signal


def test_score_signal_is_deterministic_across_processes() -> None:
    region = "New York"
    skill = "Python"
    timeframe = "30d"

    score_main = _score_signal(region, skill, timeframe)
    score_repeat = _score_signal(region, skill, timeframe)
    assert score_main == score_repeat

    script = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('labor_market_feed', {str(MODULE_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"print(module._score_signal({region!r}, {skill!r}, {timeframe!r}))\n"
    )
    output = subprocess.check_output([sys.executable, "-c", script])
    score_subprocess = float(output.decode().strip())

    assert score_main == score_subprocess


def test_fetch_labor_market_signals_dedupes_inputs() -> None:
    result = asyncio.run(
        labor_market_feed.fetch_labor_market_signals(
            regions=[" New York ", "New York", ""],
            skills=["Python", None, "Python "],  # type: ignore[arg-type]
        )
    )

    assert result["regions"] == ("New York",)
    assert result["skills"] == ("Python",)
