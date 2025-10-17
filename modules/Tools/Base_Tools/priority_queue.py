"""Prioritised task list helpers for AtlasReporter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

__all__ = ["priority_queue", "PrioritizedTaskList", "PrioritizedTask"]


@dataclass(frozen=True)
class PrioritizedTask:
    """Representation of a single prioritized task entry."""

    id: str
    description: str
    priority: int
    status: str
    due_at: Optional[str]
    tags: tuple[str, ...]
    urgency_score: float


@dataclass(frozen=True)
class PrioritizedTaskList:
    """Container describing a sorted list of prioritized tasks."""

    generated_at: str
    total_tasks: int
    pending_tasks: int
    tasks: tuple[Mapping[str, object], ...]


def _parse_datetime(candidate: object) -> Optional[datetime]:
    if isinstance(candidate, datetime):
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)
    if isinstance(candidate, str):
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _normalize_tasks(
    tasks: Optional[Sequence[Mapping[str, object]]],
    *,
    default_priority: int,
    reference: datetime,
) -> list[PrioritizedTask]:
    normalized: list[PrioritizedTask] = []
    if not tasks:
        return normalized

    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        identifier = str(task.get("id") or f"task-{index + 1}")
        description = str(task.get("description") or identifier)
        try:
            priority = int(task.get("priority", default_priority))
        except (TypeError, ValueError):
            priority = default_priority
        status = str(task.get("status", "pending")).strip() or "pending"
        due_at_candidate = _parse_datetime(task.get("due_at"))
        due_at = due_at_candidate.isoformat() if due_at_candidate else None

        tags: list[str] = []
        raw_tags = task.get("tags")
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes, bytearray)):
            for entry in raw_tags:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate:
                        tags.append(candidate)

        urgency_seconds = 0.0
        if due_at_candidate is not None:
            delta = reference - due_at_candidate
            urgency_seconds = max(delta.total_seconds(), 0.0)

        urgency_score = float(priority * 1000) + (urgency_seconds / 3600.0)

        normalized.append(
            PrioritizedTask(
                id=identifier,
                description=description,
                priority=priority,
                status=status,
                due_at=due_at,
                tags=tuple(tags),
                urgency_score=urgency_score,
            )
        )

    normalized.sort(
        key=lambda task: (
            -task.priority,
            task.due_at or "",  # lexicographic order is fine for ISO timestamps
            task.id,
        )
    )
    return normalized


async def priority_queue(
    *,
    tasks: Optional[Sequence[Mapping[str, object]]] = None,
    default_priority: int = 3,
    limit: Optional[int] = None,
    reference_time: Optional[object] = None,
) -> Mapping[str, object]:
    """Return a sorted, structured task queue for AtlasReporter."""

    await asyncio.sleep(0)

    reference = _parse_datetime(reference_time) or datetime.now(timezone.utc)
    normalized_tasks = _normalize_tasks(
        tasks,
        default_priority=default_priority,
        reference=reference,
    )

    if isinstance(limit, int) and limit > 0:
        normalized_tasks = normalized_tasks[:limit]

    serialized_tasks = tuple(asdict(task) for task in normalized_tasks)
    pending = sum(1 for task in normalized_tasks if task.status.lower() not in {"done", "complete", "completed"})
    summary = PrioritizedTaskList(
        generated_at=reference.isoformat(),
        total_tasks=len(normalized_tasks),
        pending_tasks=pending,
        tasks=serialized_tasks,
    )
    payload = asdict(summary)
    payload["tasks"] = [dict(item) for item in payload.get("tasks", [])]
    return payload
