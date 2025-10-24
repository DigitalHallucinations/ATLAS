"""Simple roadmap management helper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoadmapItem:
    """Represents a roadmap initiative."""

    initiative_id: str
    title: str
    status: str
    owner: str
    milestones: tuple[str, ...]
    updated_at: str
    notes: str


def _normalize_milestones(milestones: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not milestones:
        return tuple()
    normalized: list[str] = []
    for milestone in milestones:
        if not isinstance(milestone, str):
            continue
        candidate = milestone.strip()
        if candidate:
            normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


class RoadmapService:
    """Manage a lightweight roadmap catalogue."""

    def __init__(self) -> None:
        self._items: MutableMapping[str, RoadmapItem] = {}

    async def run(
        self,
        *,
        initiative_id: str,
        title: str,
        status: str,
        owner: str,
        milestones: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> Mapping[str, object]:
        if not initiative_id.strip():
            raise ValueError("Initiative identifier is required.")
        if not title.strip():
            raise ValueError("Initiative title is required.")
        if not owner.strip():
            raise ValueError("Owner is required.")

        await asyncio.sleep(0)

        item = RoadmapItem(
            initiative_id=initiative_id.strip(),
            title=title.strip(),
            status=status.strip().lower() or "planned",
            owner=owner.strip(),
            milestones=_normalize_milestones(milestones),
            updated_at=datetime.now(timezone.utc).isoformat(),
            notes=(notes or "").strip(),
        )

        self._items[item.initiative_id] = item
        logger.debug("Roadmap entry %s updated", item.initiative_id)

        return asdict(item)

    async def list_items(self) -> Mapping[str, Mapping[str, object]]:
        await asyncio.sleep(0)
        return {key: asdict(value) for key, value in self._items.items()}


__all__ = ["RoadmapService", "RoadmapItem"]

