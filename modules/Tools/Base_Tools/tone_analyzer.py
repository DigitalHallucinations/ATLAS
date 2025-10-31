"""Lightweight tone analysis helper for mediation workflows."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class ToneSignal:
    """Normalized tone signal extracted from a message."""

    speaker: str
    dominant: str
    intensity: str
    evidence: str


_POSITIVE_CUES = {
    "appreciate",
    "thanks",
    "grateful",
    "aligned",
    "together",
    "happy",
    "support",
    "progress",
}

_NEGATIVE_CUES = {
    "angry",
    "upset",
    "frustrated",
    "worried",
    "blocked",
    "concern",
    "issue",
    "problem",
    "delay",
}

_TENSION_CUES = {
    "disagree",
    "conflict",
    "escalate",
    "risk",
    "deadline",
    "slip",
    "pressure",
    "urgent",
    "escalation",
}


class ToneAnalyzer:
    """Summarise tonal patterns across a sequence of messages."""

    async def run(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        focus_topics: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(messages, Sequence) or not messages:
            raise ValueError("tone_analyzer requires at least one message.")

        normalized: list[ToneSignal] = []
        tone_counts: Counter[str] = Counter()
        focus_hits: MutableMapping[str, int] = Counter()

        await asyncio.sleep(0)

        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue
            speaker = (message.get("speaker") or "participant").strip() or "participant"
            lowered = content.lower()

            positive_hits = sum(1 for cue in _POSITIVE_CUES if cue in lowered)
            negative_hits = sum(1 for cue in _NEGATIVE_CUES if cue in lowered)
            tension_hits = sum(1 for cue in _TENSION_CUES if cue in lowered)

            dominant = "neutral"
            intensity = "steady"
            score_map = {
                "supportive": positive_hits,
                "tense": tension_hits,
                "concerned": negative_hits,
            }

            best_label, best_score = max(score_map.items(), key=lambda item: item[1])
            if best_score > 0:
                dominant = best_label
                intensity = "elevated" if best_score > 1 else "noted"

            evidence = content[:160]
            normalized.append(
                ToneSignal(
                    speaker=speaker,
                    dominant=dominant,
                    intensity=intensity,
                    evidence=evidence,
                )
            )
            tone_counts[dominant] += 1

            if focus_topics:
                for topic in focus_topics:
                    topic_normalized = topic.strip().lower()
                    if topic_normalized and topic_normalized in lowered:
                        focus_hits[topic_normalized] += 1

        if not normalized:
            raise ValueError("tone_analyzer requires non-empty message content.")

        dominant_tone = max(tone_counts.items(), key=lambda item: item[1])[0]
        signals = [signal.__dict__ for signal in normalized]
        focus_summary = {
            topic: count for topic, count in sorted(focus_hits.items(), key=lambda item: (-item[1], item[0]))
        }

        return {
            "dominant_tone": dominant_tone,
            "tone_summary": dict(tone_counts),
            "signals": signals,
            "focus_mentions": focus_summary,
            "guidance": _build_guidance(dominant_tone),
        }


def _build_guidance(label: str) -> str:
    """Return facilitator guidance based on the dominant tone."""

    if label == "supportive":
        return "Momentum is positive—invite the group to document commitments while the energy is aligned."
    if label == "tense":
        return "Tension detected—slow the pace, reflect the concerns, and confirm safety before moving forward."
    if label == "concerned":
        return "Participants are signalling unresolved risks—surface decision owners and mitigation paths."
    return "Keep mirroring key points and invite perspective checks to maintain alignment."


__all__ = ["ToneAnalyzer", "ToneSignal"]
