"""Async interface for Devian 12 calendar access.

This module provides a normalized wrapper around the Devian 12 calendar
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
allowing installations to point at a custom Devian 12 calendar path or
account identifier without code changes.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # ConfigManager is optional in some test contexts
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - exercised in environments without the manager
    ConfigManager = None  # type: ignore

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Devian12CalendarError(RuntimeError):
    """Base exception for Devian 12 calendar failures."""


class CalendarBackendError(Devian12CalendarError):
    """Raised when the underlying calendar backend cannot be accessed."""


class EventNotFoundError(Devian12CalendarError):
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


class ICSCalendarBackend(CalendarBackend):
    """Read-only backend for Devian 12 local ICS stores."""

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

        return CalendarEvent(
            id=uid,
            title=summary,
            start=start,
            end=end,
            all_day=all_day,
            location=self._coerce_optional_str(location_value),
            description=self._coerce_optional_str(description_value),
            calendar=calendar_name,
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


# ---------------------------------------------------------------------------
# Tool facade
# ---------------------------------------------------------------------------


class Devian12CalendarTool:
    """Facade responsible for resolving configuration and executing operations."""

    DEFAULT_LOOKAHEAD_DAYS = 30

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
            raise CalendarBackendError("Failed to list Devian 12 calendar events") from exc

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
            raise CalendarBackendError("Failed to retrieve Devian 12 calendar event") from exc
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
            raise CalendarBackendError("Failed to search Devian 12 calendar events") from exc

        normalized = [self._to_dict(event) for event in events]
        if limit is not None and limit >= 0:
            return normalized[:limit]
        return normalized

    async def run(self, operation: str, **kwargs: Any) -> Any:
        op = (operation or "").strip().lower()
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

        raise ValueError(f"Unsupported Devian 12 calendar operation '{operation}'")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _build_backend(self) -> CalendarBackend:
        calendar_paths = self._resolve_calendar_paths()
        if not calendar_paths:
            logger.info("Devian 12 calendar paths not configured; using null backend")
            return NullCalendarBackend()

        timezone = self._resolve_timezone()
        return ICSCalendarBackend(calendar_paths, timezone)

    def _resolve_calendar_paths(self) -> List[Path]:
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

    def _coerce_timezone(self, tzid: str) -> Optional[ZoneInfo]:
        try:
            return ZoneInfo(tzid)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown timezone '%s' in configuration", tzid)
            return None

    @staticmethod
    def _to_dict(event: CalendarEvent) -> Dict[str, Any]:
        return event.to_dict()


async def devian12_calendar(operation: str, **kwargs: Any) -> Any:
    """Convenience entry point compatible with the tool manifest."""

    tool = Devian12CalendarTool()
    return await tool.run(operation, **kwargs)


__all__ = [
    "CalendarBackend",
    "CalendarBackendError",
    "CalendarEvent",
    "Devian12CalendarError",
    "Devian12CalendarTool",
    "EventNotFoundError",
    "devian12_calendar",
]

