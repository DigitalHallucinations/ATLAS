"""Lightweight affect tagging helper for creative text."""

from __future__ import annotations

import asyncio
from typing import Mapping

_KEYWORDS = {
    "joy": {
        "aliases": {"joy", "delight", "smile", "laughter", "bliss", "spark"},
        "intensity": 0.7,
    },
    "longing": {
        "aliases": {"yearn", "long", "ache", "distance", "echo"},
        "intensity": 0.6,
    },
    "tension": {
        "aliases": {"conflict", "edge", "storm", "clash", "tension"},
        "intensity": 0.65,
    },
    "wonder": {
        "aliases": {"wonder", "awe", "starlit", "mystery", "glow"},
        "intensity": 0.5,
    },
    "resolve": {
        "aliases": {"resolve", "steady", "calm", "breathe", "anchor"},
        "intensity": 0.55,
    },
}


class EmotiveTagger:
    """Tag creative text with qualitative emotional markers."""

    async def run(
        self,
        *,
        text: str,
        include_intensity: bool = True,
        max_tags: int = 5,
    ) -> Mapping[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("EmotiveTagger requires non-empty text input.")

        await asyncio.sleep(0)

        normalized_text = text.lower()
        discovered: list[dict[str, object]] = []

        for label, payload in _KEYWORDS.items():
            aliases = payload["aliases"]
            intensity = payload["intensity"]
            if any(alias in normalized_text for alias in aliases):
                entry: dict[str, object]
                if include_intensity:
                    entry = {"label": label, "intensity": round(float(intensity), 2)}
                else:
                    entry = {"label": label}
                discovered.append(entry)

        if not discovered:
            baseline = {"label": "neutral"}
            if include_intensity:
                baseline["intensity"] = 0.5
            discovered.append(baseline)

        discovered = discovered[: max_tags if isinstance(max_tags, int) and max_tags > 0 else len(discovered)]
        dominant = discovered[0]["label"] if discovered else "neutral"

        return {
            "tags": discovered,
            "dominant_emotion": dominant,
            "analysis": f"Detected {len(discovered)} emotional threads in the passage.",
        }


__all__ = ["EmotiveTagger"]
