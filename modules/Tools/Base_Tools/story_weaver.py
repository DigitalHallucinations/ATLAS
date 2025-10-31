"""Synthetic story structuring helper for creative personas.

The :class:`StoryWeaver` tool assembles a lightweight multi-beat
outline from high level creative direction. It is intentionally
non-generative – the output focuses on scaffolding that the model can
expand on without reaching out to external services.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Mapping, MutableSequence, Optional, Sequence


@dataclass(frozen=True)
class StoryBeat:
    """Represents a structured beat in the story outline."""

    order: int
    title: str
    summary: str
    focus: str
    mood: str


def _normalize_list(items: Optional[Sequence[str]]) -> list[str]:
    if not items:
        return []
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and candidate.lower() not in {entry.lower() for entry in normalized}:
            normalized.append(candidate)
    return normalized


class StoryWeaver:
    """Build a deterministic story outline from creative prompts."""

    _fallback_beats: tuple[tuple[str, str, str], ...] = (
        ("Inciting spark", "An unexpected signal disrupts the familiar.", "tension"),
        ("Rising exploration", "The characters experiment, adapt, and uncover hidden angles.", "curiosity"),
        ("Turning point", "A difficult choice reshapes the path forward.", "urgency"),
        ("Climactic convergence", "Conflicting threads collide and demand synthesis.", "intensity"),
        ("Afterglow", "Consequences settle and new intentions surface.", "resolution"),
    )

    def __init__(self) -> None:
        self._beat_templates: tuple[tuple[str, str, str], ...] = self._fallback_beats

    async def run(
        self,
        *,
        theme: str,
        genre: Optional[str] = None,
        setting: Optional[str] = None,
        characters: Optional[Sequence[str]] = None,
        tone: Optional[str] = None,
        target_length: str = "short",
    ) -> Mapping[str, object]:
        """Return a structured outline honouring the requested theme."""

        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("StoryWeaver requires a non-empty theme.")

        await asyncio.sleep(0)

        normalized_theme = theme.strip()
        normalized_genre = (genre or "cross-genre").strip()
        normalized_setting = (setting or "unspecified").strip()
        normalized_tone = (tone or "balanced").strip()
        normalized_length = target_length.strip() if isinstance(target_length, str) else "short"
        cast = _normalize_list(characters)

        beats: MutableSequence[StoryBeat] = []
        for order, (title, template, mood) in enumerate(self._beat_templates, start=1):
            summary = (
                f"{template} The creative thread stays anchored on '{normalized_theme}'."
                f" Tone leans {normalized_tone.lower()} with genre nods to {normalized_genre}."
            )
            if cast:
                summary += " Key participants: " + ", ".join(cast) + "."
            if setting and order == 1:
                summary += f" The opening orients the audience within {normalized_setting}."
            beats.append(
                StoryBeat(
                    order=order,
                    title=title,
                    summary=summary,
                    focus=f"Advance the {normalized_theme.lower()} theme.",
                    mood=mood,
                )
            )

        logline = (
            f"A {normalized_length.lower()} {normalized_genre.lower()} journey set in {normalized_setting} "
            f"where {', '.join(cast) if cast else 'the protagonists'} pursue {normalized_theme.lower()}"
        )

        return {
            "theme": normalized_theme,
            "genre": normalized_genre,
            "setting": normalized_setting,
            "tone": normalized_tone,
            "target_length": normalized_length,
            "logline": logline,
            "beats": [asdict(beat) for beat in beats],
        }


__all__ = ["StoryWeaver", "StoryBeat"]
