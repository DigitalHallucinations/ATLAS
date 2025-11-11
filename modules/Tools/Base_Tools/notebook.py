"""In-memory research notebook helper.

The :class:`NotebookTool` emulates the behaviour of a collaborative
notebook service by capturing structured notes keyed by notebook
identifier. The implementation is intentionally lightweight so that it
remains usable inside the constrained test environment while preserving
the semantics required by the tool manifests.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

from .utils import coerce_metadata, dedupe_strings


@dataclass(frozen=True)
class NotebookEntry:
    """Represents a single notebook entry."""

    entry_id: str
    content: str
    tags: tuple[str, ...]
    metadata: Mapping[str, object]
    created_at: str




class NotebookTool:
    """Persist structured notes grouped by notebook identifier."""

    def __init__(self) -> None:
        self._entries: MutableMapping[str, list[NotebookEntry]] = defaultdict(list)
        self._counter = 0

    async def run(
        self,
        *,
        notebook_id: str,
        content: str,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
        replace: bool = False,
    ) -> Mapping[str, object]:
        """Add or replace a notebook entry and return the updated state."""

        if not isinstance(notebook_id, str) or not notebook_id.strip():
            raise ValueError("Notebook identifier must be a non-empty string.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Notebook content must be a non-empty string.")

        await asyncio.sleep(0)

        normalized_id = notebook_id.strip()
        normalized_content = content.strip()
        tag_tuple = dedupe_strings(tags, lower=True)
        metadata_map = coerce_metadata(metadata)

        if replace:
            logger.debug("Replacing notebook %s with a single entry", normalized_id)
            self._entries[normalized_id] = []

        self._counter += 1
        entry = NotebookEntry(
            entry_id=f"entry-{self._counter}",
            content=normalized_content,
            tags=tag_tuple,
            metadata=metadata_map,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._entries[normalized_id].append(entry)

        return {
            "notebook_id": normalized_id,
            "entry": asdict(entry),
            "total_entries": len(self._entries[normalized_id]),
            "all_entries": [asdict(item) for item in self._entries[normalized_id]],
        }


__all__ = ["NotebookTool", "NotebookEntry"]

