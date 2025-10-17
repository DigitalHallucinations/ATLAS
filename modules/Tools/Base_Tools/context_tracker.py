"""Utility helpers for constructing conversation context snapshots.

The AtlasReporter skill relies on this module to provide a structured view of
recent conversation state.  The callable entry point intentionally accepts a
flexible payload so tests and future skills can supply lightweight conversation
records without pulling in heavier dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

__all__ = ["context_tracker", "ConversationSnapshot"]


@dataclass(frozen=True)
class ConversationSnapshot:
    """Normalized view of the active conversation state."""

    conversation_id: str
    summary: str
    highlights: tuple[str, ...]
    participants: tuple[str, ...]
    message_count: int
    last_updated: str
    history: tuple[Mapping[str, object], ...]


def _normalize_history(
    history: Optional[Sequence[Mapping[str, object]]],
) -> tuple[Mapping[str, object], ...]:
    sanitized: list[Mapping[str, object]] = []
    if not history:
        return tuple()

    for index, message in enumerate(history):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).strip() or "unknown"
        content = str(message.get("content", "")).strip()
        entry: MutableMapping[str, object] = {
            "index": index,
            "role": role,
        }
        if content:
            entry["content"] = content
        timestamp = message.get("timestamp")
        if isinstance(timestamp, (str, int, float)):
            entry["timestamp"] = timestamp
        metadata = message.get("metadata")
        if isinstance(metadata, Mapping) and metadata:
            entry["metadata"] = dict(metadata)
        sanitized.append(entry)

    return tuple(sanitized)


def _derive_summary(history: Sequence[Mapping[str, object]]) -> str:
    if not history:
        return "No conversation history available."

    last_message = history[-1]
    speaker = str(last_message.get("role", "unknown")).strip() or "unknown"
    content = str(last_message.get("content", "")).strip()
    if not content:
        return f"Latest activity recorded from {speaker}."

    snippet = content[:200]
    if len(content) > 200:
        snippet = f"{snippet}…"
    return f"Last message from {speaker}: {snippet}"


def _derive_highlights(history: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    highlights: list[str] = []
    for entry in history[-3:]:
        content = entry.get("content")
        if isinstance(content, str):
            candidate = content.strip()
            if candidate:
                highlights.append(candidate)
    return tuple(highlights)


def _collect_participants(history: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    participants: list[str] = []
    seen: set[str] = set()
    for entry in history:
        role = str(entry.get("role", "")).strip()
        if not role:
            continue
        if role not in seen:
            participants.append(role)
            seen.add(role)
    return tuple(participants)


async def context_tracker(
    *,
    conversation_id: str,
    conversation_history: Optional[Sequence[Mapping[str, object]]] = None,
    summary: Optional[str] = None,
    highlights: Optional[Sequence[str]] = None,
) -> Mapping[str, object]:
    """Return a structured snapshot of the current conversation state."""

    await asyncio.sleep(0)  # ensure consistent async contract

    normalized_history = _normalize_history(conversation_history)
    derived_summary = summary.strip() if isinstance(summary, str) else None
    if not derived_summary:
        derived_summary = _derive_summary(normalized_history)

    if highlights is None:
        highlight_candidates = _derive_highlights(normalized_history)
    else:
        highlight_candidates = tuple(
            str(item).strip()
            for item in highlights
            if isinstance(item, str) and item.strip()
        )

    participants = _collect_participants(normalized_history)
    snapshot = ConversationSnapshot(
        conversation_id=str(conversation_id),
        summary=derived_summary,
        highlights=highlight_candidates,
        participants=participants,
        message_count=len(normalized_history),
        last_updated=datetime.now(timezone.utc).isoformat(),
        history=normalized_history,
    )
    return asdict(snapshot)
