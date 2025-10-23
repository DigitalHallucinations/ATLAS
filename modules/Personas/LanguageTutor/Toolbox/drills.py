"""Dialogue drill generator for the LanguageTutor persona."""

from __future__ import annotations

import asyncio
from typing import Dict, Literal

_LEVEL_NOTES = {
    "beginner": "Use simple sentences, cognates, and repetition.",
    "intermediate": "Blend past and future tense usage with contextual hints.",
    "advanced": "Incorporate idioms, register shifts, and cultural nuances.",
}


async def dialogue_drill(
    topic: str,
    level: Literal["beginner", "intermediate", "advanced"],
    language: str,
) -> Dict[str, object]:
    """Produce a conversational drill with prompts and correction cues.

    Examples
    --------
    >>> await dialogue_drill("ordering food", "beginner", "Spanish")
    {'topic': 'ordering food', 'language': 'Spanish', ...}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    prompts = [
        {
            "role": "tutor",
            "utterance": f"Let's practice {topic} in {language}. I'll guide you through the exchange.",
        },
        {
            "role": "learner",
            "utterance": "<respond here>",
            "correction": "Provide gentle corrections emphasizing pronunciation and structure.",
        },
        {
            "role": "tutor",
            "utterance": "Offer cultural insight or alternative phrasing to extend the drill.",
        },
    ]

    return {
        "topic": topic,
        "language": language,
        "level": level,
        "guidance": _LEVEL_NOTES[level],
        "dialogue": prompts,
        "follow_up": "Suggest homework phrases and spaced repetition cadence.",
    }
