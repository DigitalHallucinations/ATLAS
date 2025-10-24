"""Simplified CRM integration supporting engagement logging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InteractionRecord:
    """Represents a CRM interaction log."""

    contact_id: str
    interaction_type: str
    summary: str
    tags: tuple[str, ...]
    timestamp: str
    metadata: Mapping[str, object]


def _normalize_tags(tags: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not tags:
        return tuple()
    normalized: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        candidate = tag.strip()
        if candidate:
            normalized.append(candidate.lower())
    return tuple(dict.fromkeys(normalized))


def _normalize_metadata(metadata: Optional[Mapping[str, object]]) -> Mapping[str, object]:
    if metadata is None:
        return {}
    if isinstance(metadata, MutableMapping):
        return dict(metadata)
    return {str(key): value for key, value in metadata.items()}


class CRMService:
    """Record structured engagement notes for contacts."""

    def __init__(self) -> None:
        self._interactions: MutableMapping[str, list[InteractionRecord]] = {}

    async def run(
        self,
        *,
        contact_id: str,
        interaction_type: str,
        summary: str,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        if not contact_id.strip():
            raise ValueError("Contact identifier must be provided.")
        if not interaction_type.strip():
            raise ValueError("Interaction type must be provided.")
        if not summary.strip():
            raise ValueError("Interaction summary must be provided.")

        await asyncio.sleep(0)

        record = InteractionRecord(
            contact_id=contact_id.strip(),
            interaction_type=interaction_type.strip().lower(),
            summary=summary.strip(),
            tags=_normalize_tags(tags),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=_normalize_metadata(metadata),
        )

        interactions = self._interactions.setdefault(record.contact_id, [])
        interactions.append(record)
        logger.info("Logged interaction for contact %s", record.contact_id)

        return {
            "contact_id": record.contact_id,
            "interaction": asdict(record),
            "total_interactions": len(interactions),
        }

    async def history(self, contact_id: str) -> Mapping[str, object]:
        await asyncio.sleep(0)
        interactions = [asdict(item) for item in self._interactions.get(contact_id, [])]
        return {"contact_id": contact_id, "interactions": interactions}


__all__ = ["CRMService", "InteractionRecord"]

