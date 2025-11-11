"""In-memory calendar booking helper for personas without native calendar access."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

from .utils import coerce_metadata, dedupe_strings


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into an aware ``datetime`` instance."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamps must be provided as non-empty ISO 8601 strings.")

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # pragma: no cover - defensive guardrail
        raise ValueError(f"Invalid ISO 8601 timestamp: {candidate}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)




@dataclass(frozen=True)
class CalendarSlot:
    """Represents a lightweight calendar booking."""

    slot_id: str
    calendar_id: str
    title: str
    start: datetime
    end: datetime
    created_at: datetime
    description: str
    location: Optional[str]
    attendees: tuple[str, ...]
    metadata: Mapping[str, object]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "slot_id": self.slot_id,
            "calendar_id": self.calendar_id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "location": self.location,
            "attendees": list(self.attendees),
            "metadata": dict(self.metadata),
        }


class CalendarService:
    """Book or retrieve calendar slots for personas lacking a native backend."""

    def __init__(self) -> None:
        self._calendars: MutableMapping[str, MutableMapping[str, CalendarSlot]] = {}

    def _calendar(self, calendar_id: str) -> MutableMapping[str, CalendarSlot]:
        return self._calendars.setdefault(calendar_id, {})

    async def run(
        self,
        *,
        operation: str,
        calendar_id: str = "primary",
        slot_id: Optional[str] = None,
        title: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        attendees: Optional[Sequence[str]] = None,
        location: Optional[str] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        """Dispatch a calendar operation."""

        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("Operation must be a non-empty string.")

        op = operation.strip().lower()
        calendar_key = (calendar_id or "primary").strip() or "primary"

        await asyncio.sleep(0)

        if op == "book":
            if not title or not title.strip():
                raise ValueError("Title must be provided when booking a slot.")
            if start is None or end is None:
                raise ValueError("Start and end timestamps must be provided when booking a slot.")

            start_dt = _parse_timestamp(start)
            end_dt = _parse_timestamp(end)

            if end_dt <= start_dt:
                raise ValueError("End timestamp must be after the start timestamp.")

            slots = self._calendar(calendar_key)
            for existing in slots.values():
                if not (end_dt <= existing.start or start_dt >= existing.end):
                    raise ValueError("Requested slot overlaps with an existing booking.")

            slot = CalendarSlot(
                slot_id=f"slot-{uuid.uuid4().hex}",
                calendar_id=calendar_key,
                title=title.strip(),
                start=start_dt,
                end=end_dt,
                created_at=datetime.now(timezone.utc),
                description=(description or "").strip(),
                location=(location or "").strip() or None,
                attendees=dedupe_strings(attendees),
                metadata=coerce_metadata(metadata),
            )
            slots[slot.slot_id] = slot

            logger.info("Booked slot %s on calendar %s", slot.slot_id, calendar_key)
            return {"status": "booked", "slot": slot.to_dict()}

        if op in {"get", "retrieve"}:
            if not slot_id:
                raise ValueError("slot_id must be provided when retrieving a slot.")
            slot = self._calendar(calendar_key).get(slot_id)
            if slot is None:
                raise KeyError(f"Slot {slot_id} was not found in calendar {calendar_key}.")
            return {"slot": slot.to_dict()}

        if op in {"list", "availability"}:
            slots = list(self._calendar(calendar_key).values())
            window_start = _parse_timestamp(start) if start else None
            window_end = _parse_timestamp(end) if end else None

            if window_start and window_end and window_end < window_start:
                raise ValueError("When filtering, end must not be earlier than start.")

            filtered = []
            for slot in slots:
                if window_start and slot.end <= window_start:
                    continue
                if window_end and slot.start >= window_end:
                    continue
                filtered.append(slot.to_dict())

            filtered.sort(key=lambda payload: payload["start"])
            return {"calendar_id": calendar_key, "count": len(filtered), "slots": filtered}

        raise ValueError(f"Unsupported calendar operation: {operation}")


__all__ = ["CalendarService", "CalendarSlot"]
