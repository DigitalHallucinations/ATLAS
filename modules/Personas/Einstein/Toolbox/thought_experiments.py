"""Relativity-themed explorations for Einstein."""

from __future__ import annotations

import asyncio
from typing import Dict, Literal

_SCENARIOS: Dict[str, Dict[str, object]] = {
    "time_dilation": {
        "title": "Twin paradox mission log",
        "setup": "One twin pilots a near-light-speed survey craft while the other maintains the Earth-based observatory.",
        "variables": [
            {"name": "velocity_fraction_c", "description": "Proportion of light speed sustained during cruise."},
            {"name": "mission_duration_years", "description": "Elapsed mission time reported by the traveler."},
        ],
        "discussion_prompts": [
            "Contrast proper time experienced by each twin.",
            "Map the observable consequences when the traveler returns.",
            "Connect the scenario to GPS satellite synchronization practices.",
        ],
    },
    "energy": {
        "title": "Photon-box bookkeeping",
        "setup": "Analyze a perfectly mirrored box as photons enter and exit, tracking energy-mass equivalence.",
        "variables": [
            {"name": "photon_frequency", "description": "Frequency of each injected photon."},
            {"name": "entry_interval", "description": "Time spacing between photon injections."},
        ],
        "discussion_prompts": [
            "Use conservation laws to show mass increase in the system.",
            "Relate findings to solar sail propulsion concepts.",
            "Outline edge cases where classical intuition breaks down.",
        ],
    },
    "gravity": {
        "title": "Elevator equivalence walkthrough",
        "setup": "Passengers analyze experiments inside an accelerating elevator to deduce gravitational effects without exterior cues.",
        "variables": [
            {"name": "acceleration", "description": "Measured acceleration inside the cabin."},
            {"name": "experiment_suite", "description": "Ordered list of physical tests performed."},
        ],
        "discussion_prompts": [
            "Explain how light beams curve within the cabin frame.",
            "Relate observations to spacetime curvature without invoking tensors explicitly.",
            "Suggest practical engineering tests inspired by the thought experiment.",
        ],
    },
}


async def relativity_scenario(topic: Literal["time_dilation", "energy", "gravity"]) -> Dict[str, object]:
    """Return a curated relativity scenario suited to Einstein's pedagogy.

    Examples
    --------
    >>> await relativity_scenario("gravity")
    {'title': 'Elevator equivalence walkthrough', ...}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    return dict(_SCENARIOS[topic])
