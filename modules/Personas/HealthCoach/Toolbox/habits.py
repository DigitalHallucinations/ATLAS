"""Habit stacking planner for HealthCoach."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List

_GUIDANCE_PATH = Path(__file__).with_name("data").joinpath("habit_guidance.json")
with _GUIDANCE_PATH.open("r", encoding="utf-8") as fh:
    _GUIDANCE: Dict[str, List[Dict[str, str]]] = json.load(fh)


async def habit_stack_planner(goal: str, schedule: str, blockers: List[str]) -> Dict[str, object]:
    """Design a habit stack that blends nutrition, stress, and sleep anchors.

    Examples
    --------
    >>> await habit_stack_planner(
    ...     goal="Improve evening recovery",
    ...     schedule="Weeknights",
    ...     blockers=["Late laptop use"],
    ... )
    {'goal': 'Improve evening recovery', ...}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    recommendations = []
    for pillar, entries in _GUIDANCE.items():
        for entry in entries:
            recommendations.append({"pillar": pillar, **entry})

    blocker_notes = [
        {
            "blocker": blocker,
            "strategy": f"Pair blocker '{blocker}' with a micro-habit from the plan to neutralize it.",
        }
        for blocker in blockers
    ]

    return {
        "goal": goal,
        "schedule": schedule,
        "stacked_interventions": recommendations,
        "blocker_strategies": blocker_notes,
        "check_ins": [
            "Log perceived energy levels during the schedule window.",
            "Adjust hydration prompts alongside stress rituals if adherence slips.",
        ],
    }
