"""In-memory workspace publishing helper for simulated brief distribution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspacePublication:
    """Represents a published payload within a workspace channel."""

    workspace_id: str
    channel_id: str
    body: Mapping[str, object]
    metadata: Mapping[str, object]
    published_at: str


def _normalize_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
    normalized: MutableMapping[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        candidate = key.strip()
        if not candidate:
            continue
        normalized[candidate] = value
    return dict(normalized)


class WorkspacePublisher:
    """Record published workspace briefs for audit and testing scenarios."""

    def __init__(self) -> None:
        self._history: list[WorkspacePublication] = []

    async def run(
        self,
        *,
        workspace_id: str,
        body: Mapping[str, object],
        channel_id: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("Workspace identifier must be provided.")
        if channel_id is not None and (not isinstance(channel_id, str) or not channel_id.strip()):
            raise ValueError("Channel identifier must be a non-empty string when provided.")
        if not isinstance(body, Mapping) or not body:
            raise ValueError("A body payload with at least one field is required.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("Metadata must be a mapping when provided.")

        await asyncio.sleep(0)

        normalized_workspace = workspace_id.strip()
        normalized_channel = channel_id.strip() if isinstance(channel_id, str) else "default"
        normalized_body = _normalize_mapping(body)
        if not normalized_body:
            raise ValueError("Body payload must contain at least one string-keyed entry.")
        normalized_metadata = _normalize_mapping(metadata or {})

        publication = WorkspacePublication(
            workspace_id=normalized_workspace,
            channel_id=normalized_channel,
            body=normalized_body,
            metadata=normalized_metadata,
            published_at=datetime.now(timezone.utc).isoformat(),
        )
        self._history.append(publication)

        logger.info(
            "Workspace %s publication dispatched to channel %s",  # noqa: G004 - structured logging not used
            publication.workspace_id,
            publication.channel_id,
        )

        return asdict(publication)

    def history(self) -> list[Mapping[str, object]]:
        """Return a copy of recorded publications."""

        return [asdict(entry) for entry in self._history]


__all__ = ["WorkspacePublisher", "WorkspacePublication"]
