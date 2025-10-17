"""Async interface for Debian 12 calendar access.

This module provides a normalized wrapper around the Debian 12 calendar
backends.  It currently supports reading local ICS stores and can be
extended to speak to the desktop DBus interface.  The tool exposes three
operations that the assistant runtime can call via tool manifests:

``list``
    Return upcoming events within an optional time window.

``detail``
    Return a single event by identifier.

``search``
    Perform a text search across the available events.

The implementation is intentionally defensive so the assistant can surface
clear error messages when calendar access has not been configured.  All
configuration is resolved through :class:`ATLAS.config.ConfigManager`,
allowing installations to point at a custom Debian 12 calendar path or
account identifier without code changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)
from uuid import uuid4

try:  # ConfigManager is optional in some test contexts
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - exercised in environments without the manager
    ConfigManager = None  # type: ignore

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Debian12CalendarError(RuntimeError):
    """Base exception for Debian 12 calendar failures."""


class CalendarBackendError(Debian12CalendarError):
    """Raised when the underlying calendar backend cannot be accessed."""


class EventNotFoundError(Debian12CalendarError):
    """Raised when an event identifier cannot be resolved."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CalendarEvent:
    """Normalized calendar event representation."""

    id: str
    title: str
    start: _dt.datetime
    end: _dt.datetime
    all_day: bool
    location: Optional[str] = None
    description: Optional[str] = None
    calendar: Optional[str] = None
    attendees: List[Dict[str, Optional[str]]] = field(default_factory=list)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event into a JSON friendly dict."""

        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "all_day": self.all_day,
            "location": self.location,
            "description": self.description,
            "calendar": self.calendar,
            "attendees": [dict(attendee) for attendee in self.attendees],
            "raw": dict(self.raw),
        }


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class CalendarBackend:
    """Protocol-like base class for calendar backends."""

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        raise NotImplementedError

    async def get_event(
        self, event_id: str, calendar: Optional[str] = None
    ) -> CalendarEvent:
        raise NotImplementedError

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        raise NotImplementedError

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        raise NotImplementedError

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        raise NotImplementedError

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        raise NotImplementedError


class NullCalendarBackend(CalendarBackend):
    """Backend used when calendar access has not been configured."""

    async def list_events(
        self, start: _dt.datetime, end: _dt.datetime, calendar: Optional[str] = None
    ) -> Sequence[CalendarEvent]:
        return []

    async def get_event(
        self, event_id: str, calendar: Optional[str] = None
    ) -> CalendarEvent:
        raise EventNotFoundError("Calendar access has not been configured")

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        return []

    async def create_event(
        self, payload: Mapping[str, Any], calendar: Optional[str] = None
    ) -> CalendarEvent:
        raise CalendarBackendError("Calendar access has not been configured")

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        raise CalendarBackendError("Calendar access has not been configured")

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        raise CalendarBackendError("Calendar access has not been configured")


class ICSCalendarBackend(CalendarBackend):
    """Read-only backend for Debian 12 local ICS stores."""

    def __init__(
        self,
        calendar_paths: Sequence[Path],
        default_timezone: ZoneInfo,
    ) -> None:
        self._paths = [path for path in calendar_paths if path.exists()]
        self._default_tz = default_timezone

    async def list_events(
        self, start: _dt.datetime, end: _dt.datetime, calendar: Optional[str] = None
    ) -> Sequence[CalendarEvent]:
        events: List[CalendarEvent] = []
        for path in self._iter_paths(calendar):
            events.extend(await self._load_path(path))
        return [event for event in events if self._in_range(event, start, end)]

    async def get_event(
        self, event_id: str, calendar: Optional[str] = None
    ) -> CalendarEvent:
        for path in self._iter_paths(calendar):
            events = await self._load_path(path)
            for event in events:
                if event.id == event_id:
                    return event
        raise EventNotFoundError(f"Calendar event '{event_id}' was not found")

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        normalized = query.lower().strip()
        if not normalized:
            return await self.list_events(start, end, calendar=calendar)

        results: List[CalendarEvent] = []
        for event in await self.list_events(start, end, calendar=calendar):
            haystacks = [
                event.title,
                event.description or "",
                event.location or "",
            ]
            if any(normalized in (value or "").lower() for value in haystacks):
                results.append(event)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_paths(self, calendar: Optional[str]) -> Iterable[Path]:
        if calendar:
            candidates = [path for path in self._paths if path.stem == calendar]
            if candidates:
                return candidates
        return tuple(self._paths)

    async def _load_path(self, path: Path) -> Sequence[CalendarEvent]:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return self._parse_ics(text, calendar_name=path.stem)

    def _parse_ics(self, text: str, calendar_name: str) -> Sequence[CalendarEvent]:
        events: List[CalendarEvent] = []
        current: MutableMapping[str, Any] = {}
        current_params: Dict[str, Dict[str, str]] = {}
        last_key: Optional[str] = None

        for raw_line in text.splitlines():
            if raw_line.startswith(" ") and last_key:
                previous = current.get(last_key)
                continuation = raw_line.strip()
                if isinstance(previous, list):
                    if previous:
                        previous[-1] = f"{previous[-1]}{continuation}"
                elif previous is not None:
                    current[last_key] = f"{previous}{continuation}"
                continue

            line = raw_line.strip()
            if not line:
                continue

            if line == "BEGIN:VEVENT":
                current = {}
                current_params = {}
                last_key = None
                continue

            if line == "END:VEVENT":
                if current:
                    event = self._build_event(current, current_params, calendar_name)
                    if event:
                        events.append(event)
                current = {}
                current_params = {}
                last_key = None
                continue

            if ":" not in line:
                last_key = None
                continue

            field, value = line.split(":", 1)
            base, *raw_params = field.split(";")
            base_key = base.upper()
            params: Dict[str, str] = {}
            for raw_param in raw_params:
                if "=" in raw_param:
                    key, param_value = raw_param.split("=", 1)
                    params[key.upper()] = param_value
                else:
                    params[raw_param.upper()] = "TRUE"
            if params:
                current_params[base_key] = params

            previous_value = current.get(base_key)
            if previous_value is None:
                current[base_key] = value
            elif isinstance(previous_value, list):
                previous_value.append(value)
            else:
                current[base_key] = [previous_value, value]
            last_key = base_key

        return events

    def _build_event(
        self,
        data: Mapping[str, Any],
        params: Mapping[str, Mapping[str, str]],
        calendar_name: str,
    ) -> Optional[CalendarEvent]:
        uid = str(data.get("UID") or "").strip()
        if not uid:
            logger.debug("Skipping event without UID in calendar '%s'", calendar_name)
            return None

        summary = str(data.get("SUMMARY") or "").strip() or "Untitled event"

        start, start_all_day = self._parse_ics_datetime(data, params, "DTSTART")
        end, end_all_day = self._parse_ics_datetime(data, params, "DTEND")
        if not start:
            logger.debug("Skipping event '%s' without DTSTART", uid)
            return None

        all_day = start_all_day or end_all_day
        if end is None:
            end = start + (_dt.timedelta(days=1) if all_day else _dt.timedelta())

        location_value = data.get("LOCATION")
        description_value = data.get("DESCRIPTION")
        attendees = self._parse_attendees(data.get("ATTENDEE"))

        return CalendarEvent(
            id=uid,
            title=summary,
            start=start,
            end=end,
            all_day=all_day,
            location=self._coerce_optional_str(location_value),
            description=self._coerce_optional_str(description_value),
            calendar=calendar_name,
            attendees=attendees,
            raw=data,
        )

    def _parse_ics_datetime(
        self,
        data: Mapping[str, Any],
        params: Mapping[str, Mapping[str, str]],
        field: str,
    ) -> tuple[Optional[_dt.datetime], bool]:
        value = data.get(field)
        if value is None:
            return None, False

        timezone = self._default_tz
        field_params = params.get(field)
        if field_params:
            tzid = field_params.get("TZID")
            if tzid:
                tz = self._coerce_timezone(tzid)
                if tz is not None:
                    timezone = tz

        if isinstance(value, list):
            raw_value = value[0]
        else:
            raw_value = value

        raw_value = str(raw_value).strip()
        if not raw_value:
            return None, False

        # VALUE=DATE indicates an all-day event represented as YYYYMMDD
        value_kind = (field_params or {}).get("VALUE", "DATE-TIME").upper()
        if value_kind == "DATE" or (len(raw_value) == 8 and raw_value.isdigit()):
            try:
                parsed_date = _dt.datetime.strptime(raw_value, "%Y%m%d").date()
            except ValueError:
                logger.debug("Unable to parse DATE value '%s' for %s", raw_value, field)
                return None, False
            return (
                _dt.datetime.combine(parsed_date, _dt.time.min, tzinfo=timezone),
                True,
            )

        if raw_value.endswith("Z"):
            try:
                parsed = _dt.datetime.strptime(raw_value, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=_dt.timezone.utc
                )
            except ValueError:
                logger.debug("Unable to parse UTC datetime '%s' for %s", raw_value, field)
                return None, False
            return parsed, False

        try:
            parsed = _dt.datetime.strptime(raw_value, "%Y%m%dT%H%M%S")
        except ValueError:
            logger.debug("Unable to parse datetime '%s' for %s", raw_value, field)
            return None, False

        return parsed.replace(tzinfo=timezone), False

    def _coerce_timezone(self, tzid: str) -> Optional[ZoneInfo]:
        try:
            return ZoneInfo(tzid)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone '%s' in calendar entry", tzid)
            return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0]
        text = str(value).strip()
        return text or None

    @staticmethod
    def _in_range(event: CalendarEvent, start: _dt.datetime, end: _dt.datetime) -> bool:
        return not (event.end < start or event.start > end)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def create_event(
        self, payload: Mapping[str, Any], calendar: Optional[str] = None
    ) -> CalendarEvent:
        path, calendar_name = self._resolve_write_target(calendar)
        event_id = str(payload.get("id") or uuid4())
        normalized = self._normalize_write_payload(payload, default_id=event_id)

        block = self._render_event_block(normalized)

        try:
            await asyncio.to_thread(
                self._write_with_lock, path, lambda text: self._insert_block(text, block)
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to create Debian 12 calendar event") from exc

        return self._payload_to_event(normalized, calendar_name)

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        if not event_id:
            raise ValueError("event_id is required for updates")

        path, calendar_name = self._resolve_write_target(calendar)

        def _transform(existing_text: str) -> str:
            block, existing_payload = self._locate_event_block(
                existing_text, event_id, calendar_name
            )
            if block is None or existing_payload is None:
                raise EventNotFoundError(f"Calendar event '{event_id}' was not found")

            merged = self._merge_payload(existing_payload, payload)
            new_block = self._render_event_block(merged)
            replacement = self._replace_block(existing_text, block, new_block)
            return replacement

        try:
            await asyncio.to_thread(self._write_with_lock, path, _transform)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self._is_event_not_found(exc):
                raise EventNotFoundError(f"Calendar event '{event_id}' was not found") from exc
            raise CalendarBackendError("Failed to update Debian 12 calendar event") from exc

        merged_payload = await asyncio.to_thread(
            self._load_updated_payload, path, event_id, calendar_name
        )
        return self._payload_to_event(merged_payload, calendar_name)

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        if not event_id:
            raise ValueError("event_id is required for deletion")

        path, calendar_name = self._resolve_write_target(calendar)

        def _transform(existing_text: str) -> str:
            block, _payload = self._locate_event_block(
                existing_text, event_id, calendar_name
            )
            if block is None:
                raise EventNotFoundError(f"Calendar event '{event_id}' was not found")
            return self._replace_block(existing_text, block, "")

        try:
            await asyncio.to_thread(self._write_with_lock, path, _transform)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self._is_event_not_found(exc):
                raise EventNotFoundError(f"Calendar event '{event_id}' was not found") from exc
            raise CalendarBackendError("Failed to delete Debian 12 calendar event") from exc

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _normalize_write_payload(
        self, payload: Mapping[str, Any], *, default_id: str
    ) -> Dict[str, Any]:
        now = _dt.datetime.now(tz=_dt.timezone.utc)

        start = payload.get("start")
        if not isinstance(start, _dt.datetime):
            raise ValueError("start datetime is required")

        end = payload.get("end")
        all_day = bool(payload.get("all_day", False))
        if not isinstance(end, _dt.datetime):
            end = start + (_dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1))

        attendees_payload = payload.get("attendees") or []
        attendees: List[Dict[str, Optional[str]]] = []
        for attendee in attendees_payload:
            attendees.append(
                {
                    "email": attendee.get("email"),
                    "name": attendee.get("name"),
                    "role": attendee.get("role"),
                    "status": attendee.get("status"),
                }
            )

        title = str(payload.get("title") or "").strip() or "Untitled event"

        return {
            "id": default_id,
            "title": title,
            "start": start,
            "end": end,
            "all_day": all_day,
            "location": payload.get("location"),
            "description": payload.get("description"),
            "attendees": attendees,
            "created": payload.get("created") or now,
            "last_modified": payload.get("last_modified") or now,
        }

    def _render_event_block(self, payload: Mapping[str, Any]) -> str:
        start_field = self._format_datetime("DTSTART", payload.get("start"), payload.get("all_day", False))
        end_field = self._format_datetime("DTEND", payload.get("end"), payload.get("all_day", False))
        stamp_field = self._format_datetime("DTSTAMP", payload.get("last_modified"), False)
        created_field = self._format_datetime("CREATED", payload.get("created"), False)
        modified_field = self._format_datetime("LAST-MODIFIED", payload.get("last_modified"), False)

        lines = [
            "BEGIN:VEVENT",
            f"UID:{payload['id']}",
            stamp_field,
            created_field,
            modified_field,
            start_field,
            end_field,
            f"SUMMARY:{self._escape_text(str(payload['title']))}",
        ]

        location = payload.get("location")
        if location:
            lines.append(f"LOCATION:{self._escape_text(str(location))}")
        description = payload.get("description")
        if description:
            lines.append(f"DESCRIPTION:{self._escape_text(str(description))}")

        for attendee in payload.get("attendees", []) or []:
            attendee_line = self._format_attendee(attendee)
            if attendee_line:
                lines.append(attendee_line)

        lines.append("END:VEVENT")
        return "\n".join(lines) + "\n"

    def _payload_to_event(
        self, payload: Mapping[str, Any], calendar_name: str
    ) -> CalendarEvent:
        return CalendarEvent(
            id=str(payload["id"]),
            title=str(payload["title"]),
            start=payload["start"],
            end=payload["end"],
            all_day=bool(payload["all_day"]),
            location=self._coerce_optional_str(payload.get("location")),
            description=self._coerce_optional_str(payload.get("description")),
            calendar=calendar_name,
            attendees=[
                {
                    "email": attendee.get("email"),
                    "name": attendee.get("name"),
                    "role": attendee.get("role"),
                    "status": attendee.get("status"),
                }
                for attendee in (payload.get("attendees") or [])
            ],
            raw={},
        )

    def _resolve_write_target(self, calendar: Optional[str]) -> Tuple[Path, str]:
        if calendar:
            for path in self._paths:
                if path.stem == calendar:
                    return path, path.stem
        if self._paths:
            primary = self._paths[0]
            return primary, primary.stem
        raise CalendarBackendError("No Debian 12 calendar paths configured for writes")

    def _write_with_lock(self, path: Path, transform: Callable[[str], str]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if path.exists() else "w+"
        with open(path, mode, encoding="utf-8", newline="\n") as handle:
            with self._exclusive_lock(handle):
                handle.seek(0)
                existing = handle.read()
                if not existing:
                    existing = self._empty_calendar()
                new_text = transform(existing)
                handle.seek(0)
                handle.write(new_text)
                handle.truncate()
                return new_text

    def _insert_block(self, text: str, block: str) -> str:
        marker = "END:VCALENDAR"
        index = text.rfind(marker)
        if index == -1:
            text = self._empty_calendar()
            index = text.rfind(marker)
        prefix = text[:index]
        suffix = text[index:]
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return f"{prefix}{block}{suffix}"

    def _replace_block(self, text: str, block: str, replacement: str) -> str:
        index = text.find(block)
        if index == -1:
            raise EventNotFoundError("Unable to locate calendar event block")
        new_text = text[:index] + replacement + text[index + len(block) :]
        return new_text

    def _load_updated_payload(
        self, path: Path, event_id: str, calendar_name: str
    ) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        block, payload = self._locate_event_block(text, event_id, calendar_name)
        if block is None or payload is None:
            raise EventNotFoundError(f"Calendar event '{event_id}' was not found after update")
        return payload

    def _locate_event_block(
        self, text: str, event_id: str, calendar_name: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        cursor = 0
        while True:
            begin = text.find("BEGIN:VEVENT", cursor)
            if begin == -1:
                return None, None
            end_index = text.find("END:VEVENT", begin)
            if end_index == -1:
                return None, None
            end_line = text.find("\n", end_index)
            if end_line == -1:
                end_line = end_index + len("END:VEVENT")
            else:
                end_line += 1
            block = text[begin:end_line]
            if self._block_matches(block, event_id):
                payload = self._parse_block(block, calendar_name)
                return block, payload
            cursor = end_line

    def _merge_payload(
        self, existing: Mapping[str, Any], updates: Mapping[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(existing)
        for key in ("title", "start", "end", "all_day", "location", "description", "attendees"):
            if key in updates and updates[key] is not None:
                merged[key] = updates[key]

        merged.setdefault("created", existing.get("created") or _dt.datetime.now(tz=_dt.timezone.utc))
        merged["last_modified"] = updates.get("last_modified") or _dt.datetime.now(tz=_dt.timezone.utc)

        start = merged.get("start")
        if not isinstance(start, _dt.datetime):
            raise ValueError("start datetime is required")
        end = merged.get("end")
        all_day = bool(merged.get("all_day", False))
        if not isinstance(end, _dt.datetime):
            merged["end"] = start + (
                _dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1)
            )

        attendees = merged.get("attendees") or []
        normalized_attendees: List[Dict[str, Optional[str]]] = []
        for attendee in attendees:
            normalized_attendees.append(
                {
                    "email": attendee.get("email"),
                    "name": attendee.get("name"),
                    "role": attendee.get("role"),
                    "status": attendee.get("status"),
                }
            )
        merged["attendees"] = normalized_attendees

        return merged

    def _parse_block(self, block: str, calendar_name: str) -> Dict[str, Any]:
        events = self._parse_ics(block, calendar_name)
        if not events:
            raise CalendarBackendError("Unable to parse calendar event block")
        event = events[0]
        return {
            "id": event.id,
            "title": event.title,
            "start": event.start,
            "end": event.end,
            "all_day": event.all_day,
            "location": event.location,
            "description": event.description,
            "attendees": event.attendees,
            "created": self._parse_timestamp(event.raw.get("CREATED")),
            "last_modified": self._parse_timestamp(event.raw.get("LAST-MODIFIED")),
        }

    def _block_matches(self, block: str, event_id: str) -> bool:
        for line in block.splitlines():
            if line.upper().startswith("UID"):
                _, _, value = line.partition(":")
                if value.strip() == event_id:
                    return True
        return False

    def _empty_calendar(self) -> str:
        return "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//ATLAS//EN\nEND:VCALENDAR\n"

    @contextlib.contextmanager
    def _exclusive_lock(self, handle: Any) -> Iterable[None]:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        except Exception:  # pragma: no cover - fallback for non-posix
            yield

    def _format_datetime(
        self, field: str, value: Optional[_dt.datetime], all_day: bool
    ) -> str:
        if value is None:
            value = _dt.datetime.now(tz=_dt.timezone.utc)
        if all_day:
            return f"{field};VALUE=DATE:{value.date().strftime('%Y%m%d')}"
        value = value.astimezone(_dt.timezone.utc)
        return f"{field}:{value.strftime('%Y%m%dT%H%M%SZ')}"

    @staticmethod
    def _escape_text(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )

    def _format_attendee(self, attendee: Mapping[str, Optional[str]]) -> Optional[str]:
        email = attendee.get("email")
        if not email:
            return None
        parts = ["ATTENDEE"]
        name = attendee.get("name")
        if name:
            parts.append(f"CN={self._escape_text(str(name))}")
        role = attendee.get("role")
        if role:
            parts.append(f"ROLE={self._escape_text(str(role))}")
        status = attendee.get("status")
        if status:
            parts.append(f"PARTSTAT={self._escape_text(str(status))}")
        header = ";".join(parts)
        return f"{header}:mailto:{email}"

    def _parse_attendees(self, raw_value: Any) -> List[Dict[str, Optional[str]]]:
        if raw_value is None:
            return []
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        attendees: List[Dict[str, Optional[str]]] = []
        for value in values:
            parsed = self._parse_attendee_line(str(value))
            if parsed:
                attendees.append(parsed)
        return attendees

    def _parse_attendee_line(self, line: str) -> Optional[Dict[str, Optional[str]]]:
        if not line:
            return None
        header, _, tail = line.partition(":")
        if not tail:
            return None
        email = tail.strip()
        if email.lower().startswith("mailto:"):
            email = email[6:]
        params = header.split(";")
        meta: Dict[str, Optional[str]] = {}
        for token in params[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                meta[key.upper()] = value
        return {
            "email": email or None,
            "name": meta.get("CN"),
            "role": meta.get("ROLE"),
            "status": meta.get("PARTSTAT"),
        }

    def _parse_timestamp(self, value: Any) -> Optional[_dt.datetime]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return _dt.datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            return None

    def _is_event_not_found(self, exc: BaseException) -> bool:
        current: Optional[BaseException] = exc
        while current is not None:
            if isinstance(current, EventNotFoundError):
                return True
            current = current.__cause__ or current.__context__
        return False


# ---------------------------------------------------------------------------
# Tool facade
# ---------------------------------------------------------------------------


class Debian12CalendarTool:
    """Facade responsible for resolving configuration and executing operations."""

    DEFAULT_LOOKAHEAD_DAYS = 30
    WRITE_OPERATIONS = {"create", "update", "delete"}

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        backend: Optional[CalendarBackend] = None,
    ) -> None:
        if config_manager is None:
            if ConfigManager is None:
                raise RuntimeError("ConfigManager is unavailable in this environment")
            config_manager = ConfigManager()

        self.config_manager = config_manager
        self._unset = getattr(config_manager, "UNSET", object())
        self._backend = backend or self._build_backend()

    async def list_events(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        calendar: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        start_dt, end_dt = self._resolve_range(start, end)
        try:
            events = await self._backend.list_events(start_dt, end_dt, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to list Debian 12 calendar events") from exc

        normalized = [self._to_dict(event) for event in events]
        if limit is not None and limit >= 0:
            return normalized[:limit]
        return normalized

    async def get_event_detail(
        self, event_id: str, calendar: Optional[str] = None
    ) -> Dict[str, Any]:
        if not event_id:
            raise ValueError("event_id is required")
        try:
            event = await self._backend.get_event(event_id, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to retrieve Debian 12 calendar event") from exc
        return self._to_dict(event)

    async def search_events(
        self,
        query: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        calendar: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        start_dt, end_dt = self._resolve_range(start, end)
        try:
            events = await self._backend.search_events(
                query, start_dt, end_dt, calendar=calendar
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to search Debian 12 calendar events") from exc

        normalized = [self._to_dict(event) for event in events]
        if limit is not None and limit >= 0:
            return normalized[:limit]
        return normalized

    async def create_event(
        self,
        title: str,
        start: Any,
        end: Optional[Any] = None,
        all_day: Any = False,
        calendar: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[Sequence[Mapping[str, Any]]] = None,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            raise ValueError("title is required")

        start_dt = self._parse_required_datetime(start, "start")
        end_dt = self._parse_datetime_optional(end)
        all_day_flag = self._coerce_bool(all_day, default=False)
        if end_dt is None:
            end_dt = start_dt + (
                _dt.timedelta(days=1) if all_day_flag else _dt.timedelta(hours=1)
            )

        payload = {
            "id": event_id,
            "title": normalized_title,
            "start": start_dt,
            "end": end_dt,
            "all_day": all_day_flag,
            "description": description,
            "location": location,
            "attendees": self._normalize_attendees(attendees),
        }

        try:
            event = await self._backend.create_event(payload, calendar=calendar)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to create Debian 12 calendar event") from exc

        return self._to_dict(event)

    async def update_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
        title: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        all_day: Optional[Any] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not event_id:
            raise ValueError("event_id is required")

        payload: Dict[str, Any] = {}
        if title is not None:
            normalized_title = str(title).strip()
            if not normalized_title:
                raise ValueError("title must not be empty when provided")
            payload["title"] = normalized_title
        if start is not None:
            payload["start"] = self._parse_required_datetime(start, "start")
        if end is not None:
            parsed_end = self._parse_datetime_optional(end)
            if parsed_end is None:
                raise ValueError("end must be a valid datetime when provided")
            payload["end"] = parsed_end
        if all_day is not None:
            maybe_bool = self._maybe_bool(all_day)
            if maybe_bool is None:
                raise ValueError("all_day must be a boolean value")
            payload["all_day"] = maybe_bool
        if description is not None:
            payload["description"] = description
        if location is not None:
            payload["location"] = location
        if attendees is not None:
            payload["attendees"] = self._normalize_attendees(attendees)

        try:
            event = await self._backend.update_event(event_id, payload, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to update Debian 12 calendar event") from exc

        return self._to_dict(event)

    async def delete_event(
        self, event_id: str, calendar: Optional[str] = None
    ) -> Dict[str, Any]:
        if not event_id:
            raise ValueError("event_id is required")

        try:
            await self._backend.delete_event(event_id, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CalendarBackendError("Failed to delete Debian 12 calendar event") from exc

        return {"status": "deleted", "event_id": event_id}

    async def run(self, operation: str, **kwargs: Any) -> Any:
        context = kwargs.pop("context", None)
        op = (operation or "").strip().lower()

        if (
            op in self.WRITE_OPERATIONS
            and not self._persona_allows_write_operations(context)
        ):
            raise CalendarBackendError(
                "Persona must enable 'personal_assistant.calendar_write_enabled' "
                "to modify the Debian 12 calendar."
            )
        if op == "list":
            return await self.list_events(
                start=kwargs.get("start"),
                end=kwargs.get("end"),
                calendar=kwargs.get("calendar"),
                limit=kwargs.get("limit"),
            )
        if op == "detail":
            return await self.get_event_detail(
                event_id=kwargs.get("event_id", ""),
                calendar=kwargs.get("calendar"),
            )
        if op == "search":
            return await self.search_events(
                query=kwargs.get("query", ""),
                start=kwargs.get("start"),
                end=kwargs.get("end"),
                calendar=kwargs.get("calendar"),
                limit=kwargs.get("limit"),
            )
        if op == "create":
            return await self.create_event(
                title=kwargs.get("title", ""),
                start=kwargs.get("start"),
                end=kwargs.get("end"),
                all_day=kwargs.get("all_day", False),
                calendar=kwargs.get("calendar"),
                description=kwargs.get("description"),
                location=kwargs.get("location"),
                attendees=kwargs.get("attendees"),
                event_id=kwargs.get("event_id"),
            )
        if op == "update":
            return await self.update_event(
                event_id=kwargs.get("event_id", ""),
                calendar=kwargs.get("calendar"),
                title=kwargs.get("title"),
                start=kwargs.get("start"),
                end=kwargs.get("end"),
                all_day=kwargs.get("all_day"),
                description=kwargs.get("description"),
                location=kwargs.get("location"),
                attendees=kwargs.get("attendees"),
            )
        if op == "delete":
            return await self.delete_event(
                event_id=kwargs.get("event_id", ""),
                calendar=kwargs.get("calendar"),
            )

        raise ValueError(f"Unsupported Debian 12 calendar operation '{operation}'")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _build_backend(self) -> CalendarBackend:
        calendar_paths = self._resolve_calendar_paths()
        if not calendar_paths:
            logger.info("Debian 12 calendar paths not configured; using null backend")
            return NullCalendarBackend()

        timezone = self._resolve_timezone()
        return ICSCalendarBackend(calendar_paths, timezone)

    def _resolve_calendar_paths(self) -> List[Path]:
        raw_paths = self._get_config("DEBIAN12_CALENDAR_PATHS")
        if raw_paths is None:
            raw_paths = self._get_config("DEVIAN12_CALENDAR_PATHS")
        paths: List[Path] = []
        if isinstance(raw_paths, str):
            raw_candidates = [candidate.strip() for candidate in raw_paths.split(":")]
        elif isinstance(raw_paths, Sequence):
            raw_candidates = [str(candidate).strip() for candidate in raw_paths]
        else:
            raw_candidates = []

        for candidate in raw_candidates:
            if candidate:
                paths.append(Path(candidate).expanduser())
        return paths

    def _resolve_timezone(self) -> ZoneInfo:
        configured = self._get_config("DEBIAN12_CALENDAR_TZ")
        if configured is None:
            configured = self._get_config("DEVIAN12_CALENDAR_TZ")
        if isinstance(configured, str) and configured.strip():
            tz = self._coerce_timezone(configured.strip())
            if tz is not None:
                return tz
        return ZoneInfo("UTC")

    def _get_config(self, key: str) -> Any:
        value = self.config_manager.get_config(key, self._unset)
        if value is self._unset:
            return None
        return value

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _resolve_range(
        self, start: Optional[str], end: Optional[str]
    ) -> tuple[_dt.datetime, _dt.datetime]:
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        start_dt = self._parse_datetime_input(start, fallback=now)
        if end:
            end_dt = self._parse_datetime_input(end, fallback=start_dt)
        else:
            end_dt = start_dt + _dt.timedelta(days=self.DEFAULT_LOOKAHEAD_DAYS)
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
        return start_dt, end_dt

    def _parse_datetime_input(
        self, value: Optional[str], fallback: _dt.datetime
    ) -> _dt.datetime:
        if not value:
            return fallback
        value = value.strip()
        try:
            parsed = _dt.datetime.fromisoformat(value)
        except ValueError:
            parsed = None

        if parsed is None:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = _dt.datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue

        if parsed is None:
            logger.debug("Unable to parse datetime input '%s'; using fallback", value)
            return fallback

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=fallback.tzinfo)
        return parsed

    def _parse_required_datetime(self, value: Any, field: str) -> _dt.datetime:
        parsed = self._parse_datetime_optional(value)
        if parsed is None:
            raise ValueError(f"{field} must be a valid ISO 8601 datetime string")
        return parsed

    def _parse_datetime_optional(self, value: Any) -> Optional[_dt.datetime]:
        if isinstance(value, _dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=_dt.timezone.utc)
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is None:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = _dt.datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed

    def _coerce_bool(self, value: Any, *, default: bool = False) -> bool:
        maybe = self._maybe_bool(value)
        if maybe is None:
            return default
        return maybe

    @staticmethod
    def _coerce_persona_flag(value: Any) -> bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on", "enabled"}:
                return True
            if lowered in {"false", "0", "no", "off", "disabled"}:
                return False
        return bool(value)

    def _persona_allows_write_operations(self, persona_context: Any) -> bool:
        if persona_context is None:
            return True

        payload: Optional[Mapping[str, Any]] = None
        if isinstance(persona_context, Mapping):
            for key in ("persona", "current_persona"):
                candidate = persona_context.get(key)
                if isinstance(candidate, Mapping):
                    payload = candidate
                    break
            else:
                payload = persona_context

        if not isinstance(payload, Mapping):
            return True

        type_payload: Optional[Mapping[str, Any]] = None
        raw_type = payload.get("type")
        if isinstance(raw_type, Mapping):
            type_payload = raw_type
        else:
            flags_block = payload.get("flags")
            if isinstance(flags_block, Mapping):
                nested_type = flags_block.get("type")
                if isinstance(nested_type, Mapping):
                    type_payload = nested_type

        if not isinstance(type_payload, Mapping):
            return True

        personal_assistant = type_payload.get("personal_assistant")
        if not isinstance(personal_assistant, Mapping):
            return True

        write_enabled = personal_assistant.get("calendar_write_enabled")
        if write_enabled is None:
            return True
        if not self._coerce_persona_flag(write_enabled):
            return False

        access_enabled = personal_assistant.get("access_to_calendar")
        if access_enabled is not None and not self._coerce_persona_flag(access_enabled):
            return False

        return True

    def _maybe_bool(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return None

    def _normalize_attendees(
        self, attendees: Optional[Sequence[Mapping[str, Any]]]
    ) -> List[Dict[str, Optional[str]]]:
        normalized: List[Dict[str, Optional[str]]] = []
        if not attendees:
            return normalized

        if isinstance(attendees, Mapping):
            entries: Sequence[Any] = [attendees]
        elif isinstance(attendees, (str, bytes)):
            entries = [attendees]
        else:
            entries = attendees

        for entry in entries:
            email: Optional[str] = None
            name: Optional[str] = None
            role: Optional[str] = None
            status: Optional[str] = None
            if isinstance(entry, Mapping):
                raw_email = entry.get("email") or entry.get("address")
                if isinstance(raw_email, str):
                    email = raw_email.strip()
                elif raw_email is not None:
                    email = str(raw_email).strip()
                raw_name = entry.get("name") or entry.get("display_name")
                if isinstance(raw_name, str):
                    name = raw_name.strip() or None
                raw_role = entry.get("role")
                if isinstance(raw_role, str):
                    role = raw_role.strip() or None
                raw_status = entry.get("status") or entry.get("response")
                if isinstance(raw_status, str):
                    status = raw_status.strip() or None
            else:
                raw_str = str(entry).strip()
                email = raw_str or None

            if not email:
                continue
            if email.lower().startswith("mailto:"):
                email = email[6:]
            normalized.append(
                {
                    "email": email,
                    "name": name,
                    "role": role,
                    "status": status,
                }
            )
        return normalized

    def _coerce_timezone(self, tzid: str) -> Optional[ZoneInfo]:
        try:
            return ZoneInfo(tzid)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone '%s' in configuration", tzid)
            return None

    @staticmethod
    def _to_dict(event: CalendarEvent) -> Dict[str, Any]:
        return event.to_dict()


async def debian12_calendar(operation: str, **kwargs: Any) -> Any:
    """Convenience entry point compatible with the tool manifest."""

    tool = Debian12CalendarTool()
    return await tool.run(operation, **kwargs)


__all__ = [
    "CalendarBackend",
    "CalendarBackendError",
    "CalendarEvent",
    "Debian12CalendarError",
    "Debian12CalendarTool",
    "EventNotFoundError",
    "debian12_calendar",
]

