"""Persona-safe lyric scaffolding utilities."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Optional, Sequence


def _normalize_structure(structure: Optional[Iterable[str]]) -> list[str]:
    if not structure:
        return ["verse", "chorus", "bridge"]
    normalized: list[str] = []
    for section in structure:
        if not isinstance(section, str):
            continue
        candidate = section.strip().lower()
        if candidate:
            normalized.append(candidate)
    return normalized or ["verse", "chorus", "bridge"]


def _normalize_keywords(keywords: Optional[Sequence[str]]) -> list[str]:
    if not keywords:
        return []
    collected: list[str] = []
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        candidate = keyword.strip()
        if candidate and candidate.lower() not in {item.lower() for item in collected}:
            collected.append(candidate)
    return collected


class Lyricist:
    """Generate structured lyric drafts without external dependencies."""

    async def run(
        self,
        *,
        concept: str,
        genre: Optional[str] = None,
        structure: Optional[Iterable[str]] = None,
        vibe: Optional[str] = None,
        tempo: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> Mapping[str, object]:
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("Lyricist requires a non-empty concept.")

        await asyncio.sleep(0)

        normalized_concept = concept.strip()
        normalized_genre = (genre or "experimental").strip()
        normalized_vibe = (vibe or "uplifting").strip()
        normalized_tempo = (tempo or "moderate").strip()
        structure_plan = _normalize_structure(structure)
        keyword_list = _normalize_keywords(keywords)

        lyrics = {}
        refrains = [
            f"Hold to the {normalized_concept.lower()}, let it color the air",
            f"Pulse of {normalized_genre.lower()}, beating steady and rare",
        ]

        for index, section in enumerate(structure_plan, start=1):
            lines = [
                f"{normalized_concept} in {normalized_genre.lower()} hues {normalized_vibe.lower()}ly unfolds",
                f"Tempo stays {normalized_tempo.lower()}, guiding where the feeling holds",
            ]
            if keyword_list:
                lines.append("Keywords woven: " + ", ".join(keyword_list))
            if section == "chorus":
                lines.extend(refrains)
            lyrics[f"{section}_{index}"] = lines

        hook = refrains[0]
        if keyword_list:
            hook += ". Keywords echo: " + ", ".join(keyword_list)

        return {
            "concept": normalized_concept,
            "genre": normalized_genre,
            "vibe": normalized_vibe,
            "tempo": normalized_tempo,
            "structure": structure_plan,
            "hook": hook,
            "lyrics": lyrics,
        }


__all__ = ["Lyricist"]
