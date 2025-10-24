"""Lightweight browser coordination helper.

This module intentionally avoids performing real network requests. The
``BrowserTool`` tracks requested navigations and returns a structured
payload that downstream agents can use to reason about visited pages.
The behaviour mirrors the rest of the tool surface: asynchronous
contracts, deterministic outputs, and no side effects outside of the
in-memory session history.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrowserVisit:
    """Represents a normalized record of a requested navigation."""

    url: str
    instructions: str
    annotations: tuple[str, ...]
    metadata: Mapping[str, object]
    timestamp: str


def _normalize_annotations(annotations: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not annotations:
        return tuple()
    normalized: list[str] = []
    for entry in annotations:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if candidate:
            normalized.append(candidate)
    return tuple(normalized)


def _coerce_metadata(metadata: Optional[Mapping[str, object]]) -> Mapping[str, object]:
    if metadata is None:
        return {}
    if isinstance(metadata, MutableMapping):
        return dict(metadata)
    return {str(key): value for key, value in metadata.items()}


class BrowserTool:
    """Track virtual navigation requests without hitting the network."""

    def __init__(self) -> None:
        self._history: list[BrowserVisit] = []

    async def run(
        self,
        *,
        url: str,
        instructions: Optional[str] = None,
        annotations: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        """Record a new navigation request and return the session snapshot."""

        if not isinstance(url, str) or not url.strip():
            raise ValueError("A non-empty URL is required for browser navigation.")

        await asyncio.sleep(0)  # ensure async contract

        normalized_url = url.strip()
        note = (instructions or "").strip()
        captured_annotations = _normalize_annotations(annotations)
        captured_metadata = _coerce_metadata(metadata)

        visit = BrowserVisit(
            url=normalized_url,
            instructions=note,
            annotations=captured_annotations,
            metadata=captured_metadata,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._history.append(visit)

        logger.debug("Recorded browser visit to %s", normalized_url)

        return {
            "visit": asdict(visit),
            "history_length": len(self._history),
            "recent_history": [asdict(entry) for entry in self._history[-5:]],
        }


__all__ = ["BrowserTool", "BrowserVisit"]

