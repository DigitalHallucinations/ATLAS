"""ICS file calendar backend.

This module provides a calendar backend that reads and writes local ICS
(iCalendar) files, supporting the standard VCALENDAR/VEVENT format.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import logging
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
    TYPE_CHECKING,
)
from uuid import uuid4

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import (
    CalendarBackend,
    CalendarBackendError,
    CalendarEvent,
    EventNotFoundError,
)

if TYPE_CHECKING:
    from ..config import CalendarConfig

logger = logging.getLogger(__name__)


class ICSCalendarBackend(CalendarBackend):
    """Backend for local ICS (iCalendar) file stores.

    Supports reading and writing events to .ics files on the local filesystem.
    Multiple calendar paths can be configured, with the first path used as
    the default for write operations.

    Example:
        backend = ICSCalendarBackend(
            calendar_paths=[Path("~/.calendars/personal.ics")],
            default_timezone=ZoneInfo("America/New_York"),
        )
        events = await backend.list_events(start, end)
    """

    def __init__(
        self,
        calendar_paths: Sequence[Path],
        default_timezone: ZoneInfo,
        *,
        calendar_name: Optional[str] = None,
    ) -> None:
        """Initialize the ICS backend.

        Args:
            calendar_paths: List of paths to ICS files.
            default_timezone: Default timezone for events without explicit timezone.
            calendar_name: Optional name override for this calendar.
        """
        self._paths = [Path(path).expanduser() for path in calendar_paths]
        self._default_tz = default_timezone
        self._calendar_name = calendar_name

    @classmethod
    def from_config(cls, config: "CalendarConfig") -> "ICSCalendarBackend":
        """Create an ICSCalendarBackend from configuration.

        Args:
            config: Calendar configuration.

        Returns:
            Configured ICSCalendarBackend instance.
        """
        paths: List[Path] = []
        if config.path:
            paths.append(config.path)
        if config.url and config.url.startswith("file://"):
            paths.append(Path(config.url[7:]))

        timezone = ZoneInfo("UTC")
        if config.timezone:
            try:
                timezone = ZoneInfo(config.timezone)
            except (ZoneInfoNotFoundError, ValueError):
                logger.warning("Invalid timezone '%s', using UTC", config.timezone)

        return cls(
            calendar_paths=paths,
            default_timezone=timezone,
            calendar_name=config.name,
        )

    @property
    def name(self) -> str:
        return self._calendar_name or "ics"

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        events: List[CalendarEvent] = []
        for path in self._iter_paths(calendar):
            events.extend(await self._load_path(path))
        return sorted(
            [e for e in events if self._in_range(e, start, end)],
            key=lambda e: e.start,
        )

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
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

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        path, calendar_name = self._resolve_write_target(calendar)
        event_id = str(payload.get("id") or uuid4())
        normalized = self._normalize_write_payload(payload, default_id=event_id)
        block = self._render_event_block(normalized)

        try:
            await asyncio.to_thread(
                self._write_with_lock,
                path,
                lambda text: self._insert_block(text, block),
            )
        except Exception as exc:
            raise CalendarBackendError("Failed to create ICS calendar event") from exc

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
            return self._replace_block(existing_text, block, new_block)

        try:
            await asyncio.to_thread(self._write_with_lock, path, _transform)
        except EventNotFoundError:
            raise
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(
                    f"Calendar event '{event_id}' was not found"
                ) from exc
            raise CalendarBackendError("Failed to update ICS calendar event") from exc

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
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(
                    f"Calendar event '{event_id}' was not found"
                ) from exc
            raise CalendarBackendError("Failed to delete ICS calendar event") from exc

    # -------------------------------------------------------------------------
    # Path Helpers
    # -------------------------------------------------------------------------

    def _iter_paths(self, calendar: Optional[str]) -> Iterable[Path]:
        """Iterate over paths, optionally filtered by calendar name."""
        def _existing(paths: Iterable[Path]) -> List[Path]:
            return [path for path in paths if path.exists()]

        if calendar:
            candidates = [path for path in self._paths if path.stem == calendar]
            existing_candidates = _existing(candidates)
            if existing_candidates:
                return tuple(existing_candidates)
            if candidates:
                return tuple()

        existing_paths = _existing(self._paths)
        if existing_paths:
            return tuple(existing_paths)
        return tuple()

    def _resolve_write_target(self, calendar: Optional[str]) -> Tuple[Path, str]:
        """Resolve the target path for write operations."""
        if calendar:
            for path in self._paths:
                if path.stem == calendar:
                    return path, path.stem
        existing_paths = [path for path in self._paths if path.exists()]
        if existing_paths:
            primary = existing_paths[0]
            return primary, primary.stem
        if self._paths:
            primary = self._paths[0]
            return primary, primary.stem
        raise CalendarBackendError("No ICS calendar paths configured for writes")

    # -------------------------------------------------------------------------
    # ICS Parsing
    # -------------------------------------------------------------------------

    async def _load_path(self, path: Path) -> Sequence[CalendarEvent]:
        """Load and parse events from an ICS file."""
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            logger.debug("Calendar path '%s' not found; treating as empty", path)
            return []
        return self._parse_ics(text, calendar_name=path.stem)

    def _parse_ics(self, text: str, calendar_name: str) -> Sequence[CalendarEvent]:
        """Parse ICS text into CalendarEvent objects."""
        events: List[CalendarEvent] = []
        current: MutableMapping[str, Any] = {}
        current_params: Dict[str, Dict[str, str]] = {}
        last_key: Optional[str] = None

        for raw_line in text.splitlines():
            # Handle line continuation
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
        """Build a CalendarEvent from parsed ICS data."""
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
            raw=dict(data),
        )

    def _parse_ics_datetime(
        self,
        data: Mapping[str, Any],
        params: Mapping[str, Mapping[str, str]],
        field: str,
    ) -> Tuple[Optional[_dt.datetime], bool]:
        """Parse an ICS datetime field."""
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

        # VALUE=DATE indicates an all-day event
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

        # UTC time ending with Z
        if raw_value.endswith("Z"):
            try:
                parsed = _dt.datetime.strptime(raw_value, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=_dt.timezone.utc
                )
            except ValueError:
                logger.debug("Unable to parse UTC datetime '%s' for %s", raw_value, field)
                return None, False
            return parsed, False

        # Local time
        try:
            parsed = _dt.datetime.strptime(raw_value, "%Y%m%dT%H%M%S")
        except ValueError:
            logger.debug("Unable to parse datetime '%s' for %s", raw_value, field)
            return None, False

        return parsed.replace(tzinfo=timezone), False

    # -------------------------------------------------------------------------
    # ICS Writing
    # -------------------------------------------------------------------------

    def _render_event_block(self, payload: Mapping[str, Any]) -> str:
        """Render event data as an ICS VEVENT block."""
        start_field = self._format_datetime(
            "DTSTART", payload.get("start"), payload.get("all_day", False)
        )
        end_field = self._format_datetime(
            "DTEND", payload.get("end"), payload.get("all_day", False)
        )
        stamp_field = self._format_datetime(
            "DTSTAMP", payload.get("last_modified"), False
        )
        created_field = self._format_datetime("CREATED", payload.get("created"), False)
        modified_field = self._format_datetime(
            "LAST-MODIFIED", payload.get("last_modified"), False
        )

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

    def _write_with_lock(self, path: Path, transform: Callable[[str], str]) -> str:
        """Write to the ICS file with exclusive locking."""
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
        """Insert a VEVENT block before END:VCALENDAR."""
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
        """Replace an existing VEVENT block."""
        index = text.find(block)
        if index == -1:
            raise EventNotFoundError("Unable to locate calendar event block")
        return text[:index] + replacement + text[index + len(block):]

    def _locate_event_block(
        self,
        text: str,
        event_id: str,
        calendar_name: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find a VEVENT block by event ID."""
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

    def _parse_block(self, block: str, calendar_name: str) -> Dict[str, Any]:
        """Parse a single VEVENT block into a payload dict."""
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

    def _load_updated_payload(
        self,
        path: Path,
        event_id: str,
        calendar_name: str,
    ) -> Dict[str, Any]:
        """Load the payload for an event after update."""
        text = path.read_text(encoding="utf-8")
        block, payload = self._locate_event_block(text, event_id, calendar_name)
        if block is None or payload is None:
            raise EventNotFoundError(
                f"Calendar event '{event_id}' was not found after update"
            )
        return payload

    def _block_matches(self, block: str, event_id: str) -> bool:
        """Check if a VEVENT block matches the given event ID."""
        for line in block.splitlines():
            if line.upper().startswith("UID"):
                _, _, value = line.partition(":")
                if value.strip() == event_id:
                    return True
        return False

    def _empty_calendar(self) -> str:
        """Return an empty VCALENDAR structure."""
        return "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//ATLAS//EN\nEND:VCALENDAR\n"

    @contextlib.contextmanager
    def _exclusive_lock(self, handle: Any) -> Iterable[None]:
        """Acquire exclusive lock on file handle."""
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Fallback for non-POSIX systems
            yield

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _in_range(event: CalendarEvent, start: _dt.datetime, end: _dt.datetime) -> bool:
        """Check if event overlaps with the given range."""
        return not (event.end < start or event.start > end)

    @staticmethod
    def _coerce_optional_str(value: Any) -> Optional[str]:
        """Coerce value to optional string."""
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0]
        text = str(value).strip()
        return text or None

    def _coerce_timezone(self, tzid: str) -> Optional[ZoneInfo]:
        """Parse timezone identifier."""
        try:
            return ZoneInfo(tzid)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone '%s' in calendar entry", tzid)
            return None

    def _normalize_write_payload(
        self,
        payload: Mapping[str, Any],
        *,
        default_id: str,
    ) -> Dict[str, Any]:
        """Normalize event payload for writing."""
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
            attendees.append({
                "email": attendee.get("email"),
                "name": attendee.get("name"),
                "role": attendee.get("role"),
                "status": attendee.get("status"),
            })

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

    def _merge_payload(
        self,
        existing: Mapping[str, Any],
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge existing payload with updates."""
        merged = dict(existing)
        for key in ("title", "start", "end", "all_day", "location", "description", "attendees"):
            if key in updates and updates[key] is not None:
                merged[key] = updates[key]

        merged.setdefault(
            "created",
            existing.get("created") or _dt.datetime.now(tz=_dt.timezone.utc),
        )
        merged["last_modified"] = (
            updates.get("last_modified") or _dt.datetime.now(tz=_dt.timezone.utc)
        )

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
            normalized_attendees.append({
                "email": attendee.get("email"),
                "name": attendee.get("name"),
                "role": attendee.get("role"),
                "status": attendee.get("status"),
            })
        merged["attendees"] = normalized_attendees

        return merged

    def _payload_to_event(
        self,
        payload: Mapping[str, Any],
        calendar_name: str,
    ) -> CalendarEvent:
        """Convert payload dict to CalendarEvent."""
        raw_payload = payload.get("raw")
        raw: Mapping[str, Any] = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}

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
            raw=raw,
        )

    @staticmethod
    def _is_event_not_found(exc: BaseException) -> bool:
        """Check if exception chain contains EventNotFoundError."""
        current: Optional[BaseException] = exc
        while current is not None:
            if isinstance(current, EventNotFoundError):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _format_datetime(
        self,
        field: str,
        value: Optional[_dt.datetime],
        all_day: bool,
    ) -> str:
        """Format datetime for ICS output."""
        if value is None:
            value = _dt.datetime.now(tz=_dt.timezone.utc)
        if all_day:
            return f"{field};VALUE=DATE:{value.date().strftime('%Y%m%d')}"
        value = value.astimezone(_dt.timezone.utc)
        return f"{field}:{value.strftime('%Y%m%dT%H%M%SZ')}"

    @staticmethod
    def _escape_text(value: str) -> str:
        """Escape text for ICS format."""
        return (
            value.replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )

    def _format_attendee(self, attendee: Mapping[str, Optional[str]]) -> Optional[str]:
        """Format attendee for ICS output."""
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
        """Parse attendees from ICS data."""
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
        """Parse a single ATTENDEE line."""
        if not line:
            return None
        header, _, tail = line.partition(":")
        if not tail:
            return None
        email = tail.strip()
        if email.lower().startswith("mailto:"):
            email = email[7:]
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
        """Parse ICS timestamp value."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return _dt.datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=_dt.timezone.utc
            )
        except ValueError:
            return None


__all__ = ["ICSCalendarBackend"]
