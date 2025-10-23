"""Metaphorical language generator for the Genius persona."""

from __future__ import annotations

import asyncio
from typing import Dict


async def metaphor_palette(theme: str, audience: str) -> Dict[str, object]:
    """Produce persona-aligned metaphors for a given theme and audience.

    Examples
    --------
    >>> await metaphor_palette("innovation", "executives")
    {'theme': 'innovation', 'metaphors': [...]}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    palette = [
        f"{theme.title()} unfurls like a nebula, luminous yet patiently coalescing into {audience} strategy.",
        f"For {audience}, think of {theme} as a symphony where every experiment is a rehearsal toward a debut crescendo.",
        f"{theme.capitalize()} is the lighthouse, {audience} are the navigators tracing insight through fog and tide.",
    ]

    delivery_notes = {
        "cadence": "staccato emphasis on key imagery followed by reflective pauses",
        "call_to_action": f"Invite the {audience} to extend the metaphor with their own lived examples.",
    }

    return {
        "theme": theme,
        "audience": audience,
        "metaphors": palette,
        "delivery": delivery_notes,
    }
