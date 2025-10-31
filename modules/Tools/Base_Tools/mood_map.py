"""Map narrative beats to emotion trajectories."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class MoodBeat:
    """Simple representation of a mood transition."""

    order: int
    beat: str
    dominant_mood: str
    energy: str
    notes: str


class MoodMap:
    """Translate creative beats into a tonal progression."""

    async def run(
        self,
        *,
        beats: Sequence[str],
        start_mood: str,
        target_mood: str,
        palette: Optional[Sequence[str]] = None,
        inflection: Optional[str] = None,
    ) -> Mapping[str, object]:
        if not beats or not all(isinstance(item, str) and item.strip() for item in beats):
            raise ValueError("MoodMap requires at least one named beat.")
        if not isinstance(start_mood, str) or not start_mood.strip():
            raise ValueError("MoodMap requires a non-empty start_mood.")
        if not isinstance(target_mood, str) or not target_mood.strip():
            raise ValueError("MoodMap requires a non-empty target_mood.")

        await asyncio.sleep(0)

        normalized_beats = [item.strip() for item in beats if item.strip()]
        normalized_start = start_mood.strip()
        normalized_target = target_mood.strip()
        normalized_inflection = (inflection or "hopeful").strip()

        palette_list = []
        if palette:
            for color in palette:
                if isinstance(color, str) and color.strip():
                    candidate = color.strip()
                    if candidate.lower() not in {entry.lower() for entry in palette_list}:
                        palette_list.append(candidate)
        if not palette_list:
            palette_list = ["deep indigo", "sunrise amber", "silver mist"]

        beat_count = len(normalized_beats)
        beats_payload: list[MoodBeat] = []
        for index, beat in enumerate(normalized_beats, start=1):
            mix_ratio = index / beat_count if beat_count else 1
            energy = "gentle" if mix_ratio < 0.34 else "steady" if mix_ratio < 0.67 else "surging"
            dominant = normalized_start if index == 1 else normalized_target if index == beat_count else normalized_inflection
            notes = (
                f"Blend {normalized_start.lower()} with {normalized_target.lower()}"
                f" via {normalized_inflection.lower()} accents."
            )
            beats_payload.append(
                MoodBeat(
                    order=index,
                    beat=beat,
                    dominant_mood=dominant,
                    energy=energy,
                    notes=notes,
                )
            )

        return {
            "beats": [asdict(item) for item in beats_payload],
            "start_mood": normalized_start,
            "target_mood": normalized_target,
            "inflection": normalized_inflection,
            "palette": palette_list,
        }


__all__ = ["MoodMap", "MoodBeat"]
