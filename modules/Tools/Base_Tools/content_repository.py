"""Simple in-memory content repository helper."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

from .utils import coerce_metadata, dedupe_strings


@dataclass(frozen=True)
class ContentRecord:
    """Structured representation of repository content."""

    content_id: str
    title: str
    body: str
    tags: tuple[str, ...]
    attachments: tuple[str, ...]
    metadata: Mapping[str, object]
    updated_at: str




class ContentRepository:
    """Store reusable content blocks for persona workflows."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, ContentRecord] = {}

    async def run(
        self,
        *,
        content_id: str,
        title: str,
        body: str,
        tags: Optional[Sequence[str]] = None,
        attachments: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        if not isinstance(content_id, str) or not content_id.strip():
            raise ValueError("Content identifier must be provided.")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Content title must be provided.")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Content body must be provided.")

        await asyncio.sleep(0)

        record = ContentRecord(
            content_id=content_id.strip(),
            title=title.strip(),
            body=body.strip(),
            tags=dedupe_strings(tags, lower=True),
            attachments=dedupe_strings(attachments),
            metadata=coerce_metadata(metadata),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        self._records[record.content_id] = record
        logger.debug("Content repository stored entry %s", record.content_id)

        return {
            "content_id": record.content_id,
            "title": record.title,
            "body": record.body,
            "tags": record.tags,
            "attachments": record.attachments,
            "metadata": record.metadata,
            "updated_at": record.updated_at,
        }

    async def fetch(self, content_id: str) -> Mapping[str, object]:
        await asyncio.sleep(0)
        record = self._records.get(content_id)
        if record is None:
            raise KeyError(f"Content '{content_id}' was not found.")
        return asdict(record)


__all__ = ["ContentRepository", "ContentRecord"]
