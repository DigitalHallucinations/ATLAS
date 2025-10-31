"""Heuristic tone analysis support for mediation flows."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Iterable, Mapping, Sequence

_POSITIVE_KEYWORDS = {
    "appreciate",
    "thanks",
    "grateful",
    "progress",
    "solution",
    "understand",
    "listening",
    "calm",
    "resolve",
    "together",
    "cooperate",
}

_NEGATIVE_KEYWORDS = {
    "angry",
    "upset",
    "frustrated",
    "blame",
    "disappointed",
    "tired",
    "stuck",
    "ignored",
    "unfair",
    "worried",
    "anxious",
    "tense",
}

_NEUTRAL_CUES = {
    "schedule",
    "meeting",
    "agenda",
    "update",
    "request",
    "plan",
    "detail",
    "clarify",
    "check",
    "note",
}

_SIGNAL_LABELS = {
    "positive": "reassuring",
    "negative": "tense",
    "neutral": "even",
}


def _tokenize(text: str) -> Iterable[str]:
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        yield "".join(ch for ch in raw if ch.isalpha())


def _score_tokens(tokens: Iterable[str]) -> Mapping[str, int]:
    counter = Counter({"positive": 0, "negative": 0, "neutral": 0})
    for token in tokens:
        if token in _POSITIVE_KEYWORDS:
            counter["positive"] += 1
        elif token in _NEGATIVE_KEYWORDS:
            counter["negative"] += 1
        elif token in _NEUTRAL_CUES:
            counter["neutral"] += 1
    return counter


def _slice_sentences(transcript: str) -> Sequence[str]:
    normalized = transcript.replace("\n", " ")
    raw_segments = [segment.strip() for segment in normalized.replace("?", ".").replace("!", ".").split(".")]
    return [segment for segment in raw_segments if segment]


class ToneAnalyzer:
    """Lightweight emotional signal detector for mediators."""

    async def run(
        self,
        *,
        transcript: str,
        focus: Sequence[str] | None = None,
        window: int = 5,
    ) -> Mapping[str, object]:
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("ToneAnalyzer requires a non-empty transcript.")

        await asyncio.sleep(0)

        sentences = _slice_sentences(transcript)
        tokens = list(_tokenize(transcript))
        scores = dict(_score_tokens(tokens))

        total = sum(scores.values()) or 1
        sentiment_index = round((scores["positive"] - scores["negative"]) / total, 2)

        dominant_key = max(scores, key=scores.get)
        dominant_signal = _SIGNAL_LABELS.get(dominant_key, "even")

        highlights: list[Mapping[str, object]] = []
        for sentence in sentences[: max(window, 1)]:
            sentence_tokens = list(_tokenize(sentence))
            sentence_scores = _score_tokens(sentence_tokens)
            local_key = max(sentence_scores, key=sentence_scores.get)
            highlight = {
                "excerpt": sentence,
                "tone": _SIGNAL_LABELS.get(local_key, "even"),
                "signals": dict(sentence_scores),
            }
            highlights.append(highlight)

        focus_terms = [term.lower() for term in (focus or []) if isinstance(term, str) and term.strip()]
        focus_observations: list[Mapping[str, str]] = []
        if focus_terms:
            for sentence in sentences:
                lowered = sentence.lower()
                matched = [term for term in focus_terms if term in lowered]
                if matched:
                    focus_observations.append(
                        {
                            "excerpt": sentence,
                            "matched_terms": matched,
                        }
                    )

        guidance_fragments = []
        if dominant_key == "negative":
            guidance_fragments.append("Acknowledge the tension explicitly and invite a pause for breathing space.")
        elif dominant_key == "positive":
            guidance_fragments.append("Reinforce the collaborative tone and surface the shared progress noted.")
        else:
            guidance_fragments.append("Maintain neutral framing while inviting each party to name specific needs.")
        if focus_observations:
            guidance_fragments.append("Return to the highlighted focus points and confirm each party feels heard.")

        return {
            "dominant_signal": dominant_signal,
            "scores": {
                "positive": scores["positive"],
                "negative": scores["negative"],
                "neutral": scores["neutral"],
                "sentiment_index": sentiment_index,
            },
            "highlights": highlights,
            "focus_observations": focus_observations,
            "guidance": " ".join(guidance_fragments),
        }


__all__ = ["ToneAnalyzer"]
