"""Filter lightweight mediation memory payloads."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MemoryRecord:
    """Simple representation of a stored mediation memory."""

    topic: str
    summary: str
    tags: tuple[str, ...]
    timestamp: str

    @property
    def normalized_topic(self) -> str:
        return self.topic.lower()


class MemoryRecall:
    """Return the memories that best match a mediation query."""

    async def run(
        self,
        *,
        query: str,
        memories: Sequence[Mapping[str, object]],
        limit: int = 5,
    ) -> Mapping[str, object]:
        query_normalized = (query or "").strip().lower()
        if not query_normalized:
            raise ValueError("memory_recall requires a non-empty query.")

        if not isinstance(memories, Sequence) or not memories:
            raise ValueError("memory_recall requires at least one memory candidate.")

        await asyncio.sleep(0)

        words = {token for token in query_normalized.split() if len(token) > 2}
        normalized_memories = [_normalize_memory(entry) for entry in memories]

        scored: list[tuple[int, MemoryRecord]] = []
        for record in normalized_memories:
            overlap = sum(1 for tag in record.tags if tag in words)
            if record.normalized_topic in words:
                overlap += 2
            if any(word in record.summary.lower() for word in words):
                overlap += 1
            scored.append((overlap, record))

        scored.sort(key=lambda item: (item[0], _parse_timestamp(item[1].timestamp)), reverse=True)
        top_matches = [record for score, record in scored if score > 0][: limit]

        response = {
            "query": query,
            "results": [record.__dict__ for record in top_matches],
            "count": len(top_matches),
            "evaluated": len(normalized_memories),
        }

        if not top_matches:
            response["notes"] = "No direct memory matches found—consider capturing a new context_tracker entry."

        return response


def _normalize_memory(entry: Mapping[str, object]) -> MemoryRecord:
    topic = str(entry.get("topic") or "").strip() or "unspecified"
    summary = str(entry.get("summary") or "").strip()
    raw_tags = entry.get("tags")
    tags: tuple[str, ...]
    if isinstance(raw_tags, Iterable):
        tags = tuple(str(item).strip().lower() for item in raw_tags if str(item).strip())
    else:
        tags = ()
    timestamp = str(entry.get("timestamp") or "")
    return MemoryRecord(topic=topic, summary=summary, tags=tags, timestamp=timestamp)


def _parse_timestamp(timestamp: str) -> datetime:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


__all__ = ["MemoryRecall", "MemoryRecord"]
