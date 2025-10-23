"""Weekly training microcycles for FitnessCoach."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Literal

_BASE_TEMPLATES: Dict[str, List[Dict[str, object]]] = {
    "strength": [
        {"focus": "Lower body push", "sessions": ["Back squat", "Split squat accessory"]},
        {"focus": "Upper body press", "sessions": ["Bench press", "Overhead stability work"]},
        {"focus": "Posterior chain", "sessions": ["Deadlift variations", "Hamstring isometrics"]},
    ],
    "endurance": [
        {"focus": "Aerobic base", "sessions": ["Zone 2 run", "Breathing ladder row"]},
        {"focus": "Tempo development", "sessions": ["Progressive tempo ride", "Core circuit"]},
        {"focus": "Speed play", "sessions": ["Fartlek intervals", "Mobility reset"]},
    ],
    "mobility": [
        {"focus": "Spine hygiene", "sessions": ["Segmental cat-cow", "Thoracic rotations"]},
        {"focus": "Hip unlock", "sessions": ["90/90 transitions", "Assisted deep squat holds"]},
        {"focus": "Shoulder care", "sessions": ["Wall slides", "Band external rotations"]},
    ],
}


async def microcycle_plan(
    goal: Literal["strength", "endurance", "mobility"],
    equipment: List[str],
    days: int,
) -> Dict[str, object]:
    """Generate a microcycle template adjusted for goal, equipment, and cadence.

    Examples
    --------
    >>> await microcycle_plan("strength", ["barbell", "bands"], 4)
    {'goal': 'strength', 'schedule': [...]}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    template = _BASE_TEMPLATES[goal]
    equipment_notes = [
        {
            "item": item,
            "usage": f"Incorporate {item} for progression and variation.",
        }
        for item in equipment
    ]

    schedule = []
    for day_index in range(days):
        block = template[day_index % len(template)]
        schedule.append(
            {
                "day": day_index + 1,
                "focus": block["focus"],
                "primary_sessions": block["sessions"],
                "accessory_note": "Blend core bracing and mobility finishers for recovery.",
            }
        )

    return {
        "goal": goal,
        "equipment_notes": equipment_notes,
        "schedule": schedule,
        "recovery_guidelines": [
            "Sleep 7-9 hours with consistent wake times.",
            "Prioritize protein and hydration windows post training.",
        ],
    }
