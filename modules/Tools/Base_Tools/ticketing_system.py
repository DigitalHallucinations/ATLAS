"""Stub ticketing integration for escalation workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging
import itertools

logger = logging.getLogger(__name__)

from .utils import dedupe_strings


@dataclass(frozen=True)
class TicketRecord:
    """Represents a tracked ticket."""

    ticket_id: str
    title: str
    description: str
    priority: str
    assignees: tuple[str, ...]
    tags: tuple[str, ...]
    created_at: str
    status: str


class TicketingSystem:
    """Maintain an in-memory queue of tickets."""

    def __init__(self) -> None:
        self._tickets: MutableMapping[str, TicketRecord] = {}
        self._counter = itertools.count(1)

    async def run(
        self,
        *,
        title: str,
        description: str,
        priority: str = "medium",
        assignees: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        status: str = "open",
    ) -> Mapping[str, object]:
        if not title.strip():
            raise ValueError("Ticket title must be provided.")
        if not description.strip():
            raise ValueError("Ticket description must be provided.")

        await asyncio.sleep(0)

        identifier = f"TKT-{next(self._counter):05d}"
        record = TicketRecord(
            ticket_id=identifier,
            title=title.strip(),
            description=description.strip(),
            priority=priority.strip().lower() or "medium",
            assignees=dedupe_strings(assignees),
            tags=dedupe_strings(tags),
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status.strip().lower() or "open",
        )

        self._tickets[identifier] = record
        logger.info("Created ticket %s", identifier)

        return asdict(record)

    async def get_ticket(self, ticket_id: str) -> Mapping[str, object]:
        await asyncio.sleep(0)
        record = self._tickets.get(ticket_id)
        if record is None:
            raise KeyError(f"Unknown ticket '{ticket_id}'")
        return asdict(record)


__all__ = ["TicketingSystem", "TicketRecord"]

