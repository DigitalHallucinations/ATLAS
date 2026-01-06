"""D-Bus calendar backend for Debian 12.

This module provides a calendar backend that communicates with the Debian 12
D-Bus calendar service for integration with desktop calendar applications.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from uuid import uuid4
from zoneinfo import ZoneInfo

from .base import (
    CalendarBackend,
    CalendarBackendError,
    CalendarEvent,
    EventNotFoundError,
)

if TYPE_CHECKING:
    from ..config import CalendarConfig

logger = logging.getLogger(__name__)


class DBusCalendarBackend(CalendarBackend):
    """Backend that communicates with the Debian 12 DBus calendar service.

    This backend wraps an existing DBus client to provide calendar operations
    through the system's calendar service.

    Example:
        client = get_dbus_calendar_client()
        backend = DBusCalendarBackend(
            client=client,
            default_timezone=ZoneInfo("America/New_York"),
        )
        events = await backend.list_events(start, end)
    """

    def __init__(
        self,
        client: Any,
        default_timezone: ZoneInfo,
        *,
        calendar_name: Optional[str] = None,
    ) -> None:
        """Initialize the DBus backend.

        Args:
            client: DBus calendar client instance.
            default_timezone: Default timezone for events without explicit timezone.
            calendar_name: Optional name override for this calendar.
        """
        self._client = client
        self._default_tz = default_timezone
        self._calendar_name = calendar_name

    @classmethod
    def from_config(
        cls,
        config: "CalendarConfig",
        client: Any,
    ) -> "DBusCalendarBackend":
        """Create a DBusCalendarBackend from configuration.

        Args:
            config: Calendar configuration.
            client: DBus calendar client instance.

        Returns:
            Configured DBusCalendarBackend instance.
        """
        timezone = ZoneInfo("UTC")
        if config.timezone:
            try:
                timezone = ZoneInfo(config.timezone)
            except Exception:
                logger.warning("Invalid timezone '%s', using UTC", config.timezone)

        return cls(
            client=client,
            default_timezone=timezone,
            calendar_name=config.name,
        )

    @property
    def name(self) -> str:
        return self._calendar_name or "dbus"

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        try:
            raw_events = await self._client.list_events(
                self._serialize_datetime(start),
                self._serialize_datetime(end),
                calendar=calendar,
            )
        except Exception as exc:
            raise CalendarBackendError(
                "Failed to list Debian 12 DBus calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        if not raw_events:
            return events

        for entry in raw_events:
            if not isinstance(entry, Mapping):
                logger.debug("Skipping non-mapping DBus calendar entry: %r", entry)
                continue
            try:
                normalized = self._coerce_backend_payload(entry, calendar)
            except CalendarBackendError as exc:
                logger.warning("Skipping malformed DBus calendar event: %s", exc)
                continue
            calendar_name = normalized.get("calendar") or self._default_calendar(
                calendar
            )
            events.append(self._payload_to_event(normalized, calendar_name))
        return events

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        try:
            payload = await self._client.get_event(event_id, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(event_id) from exc
            raise CalendarBackendError(
                "Failed to fetch Debian 12 DBus calendar event"
            ) from exc

        if not isinstance(payload, Mapping):
            raise EventNotFoundError(event_id)

        normalized = self._coerce_backend_payload(payload, calendar)
        calendar_name = normalized.get("calendar") or self._default_calendar(calendar)
        return self._payload_to_event(normalized, calendar_name)

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        try:
            raw_events = await self._client.search_events(
                query,
                self._serialize_datetime(start),
                self._serialize_datetime(end),
                calendar=calendar,
            )
        except Exception as exc:
            raise CalendarBackendError(
                "Failed to search Debian 12 DBus calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        if not raw_events:
            return events

        for entry in raw_events:
            if not isinstance(entry, Mapping):
                continue
            try:
                normalized = self._coerce_backend_payload(entry, calendar)
            except CalendarBackendError as exc:
                logger.warning("Skipping malformed DBus calendar event: %s", exc)
                continue
            calendar_name = normalized.get("calendar") or self._default_calendar(
                calendar
            )
            events.append(self._payload_to_event(normalized, calendar_name))
        return events

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        event_id = str(payload.get("id") or uuid4())
        normalized = self._normalize_write_payload(payload, default_id=event_id)
        request_payload = self._serialize_payload(normalized)

        try:
            response = await self._client.create_event(
                request_payload, calendar=calendar
            )
        except Exception as exc:
            raise CalendarBackendError(
                "Failed to create Debian 12 DBus calendar event"
            ) from exc

        if isinstance(response, Mapping):
            normalized_response = self._coerce_backend_payload(response, calendar)
            calendar_name = normalized_response.get(
                "calendar"
            ) or self._default_calendar(calendar)
            return self._payload_to_event(normalized_response, calendar_name)

        calendar_name = self._default_calendar(calendar)
        fallback = dict(normalized)
        fallback["calendar"] = calendar_name
        fallback["raw"] = dict(request_payload)
        return self._payload_to_event(fallback, calendar_name)

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        if not event_id:
            raise ValueError("event_id is required for DBus updates")

        try:
            existing = await self._client.get_event(event_id, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(event_id) from exc
            raise CalendarBackendError(
                "Failed to fetch Debian 12 DBus calendar event for update"
            ) from exc

        if not isinstance(existing, Mapping):
            raise EventNotFoundError(event_id)

        current_payload = self._coerce_backend_payload(existing, calendar)
        merged = self._merge_payload(current_payload, payload)
        request_payload = self._serialize_payload(merged)

        try:
            response = await self._client.update_event(
                event_id, request_payload, calendar=calendar
            )
        except EventNotFoundError:
            raise
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(event_id) from exc
            raise CalendarBackendError(
                "Failed to update Debian 12 DBus calendar event"
            ) from exc

        if isinstance(response, Mapping):
            normalized_response = self._coerce_backend_payload(response, calendar)
            calendar_name = normalized_response.get(
                "calendar"
            ) or self._default_calendar(calendar)
            return self._payload_to_event(normalized_response, calendar_name)

        calendar_name = current_payload.get("calendar") or self._default_calendar(
            calendar
        )
        merged["calendar"] = calendar_name
        merged["raw"] = dict(request_payload)
        return self._payload_to_event(merged, calendar_name)

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        if not event_id:
            raise ValueError("event_id is required for DBus deletion")

        try:
            await self._client.delete_event(event_id, calendar=calendar)
        except EventNotFoundError:
            raise
        except Exception as exc:
            if self._is_event_not_found(exc):
                raise EventNotFoundError(event_id) from exc
            raise CalendarBackendError(
                "Failed to delete Debian 12 DBus calendar event"
            ) from exc

    # -------------------------------------------------------------------------
    # Payload Coercion and Normalization
    # -------------------------------------------------------------------------

    def _coerce_backend_payload(
        self,
        payload: Mapping[str, Any],
        calendar_hint: Optional[str],
    ) -> Dict[str, Any]:
        """Coerce raw backend payload to normalized format."""
        event_id = self._coerce_optional_str(
            payload.get("id") or payload.get("uid") or payload.get("event_id")
        )
        if not event_id:
            raise CalendarBackendError(
                "DBus calendar payload is missing an identifier"
            )

        title = self._coerce_optional_str(payload.get("title")) or "Untitled event"
        start = self._parse_backend_datetime(payload.get("start"))
        if start is None:
            raise CalendarBackendError("DBus calendar payload is missing a start time")

        all_day_value = payload.get("all_day")
        if isinstance(all_day_value, str):
            lowered = all_day_value.strip().lower()
            all_day = lowered in {"true", "1", "yes", "on"}
        else:
            all_day = bool(all_day_value)

        end = self._parse_backend_datetime(payload.get("end"))
        if end is None:
            end = start + (
                _dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1)
            )

        calendar_name = (
            self._coerce_optional_str(payload.get("calendar"))
            or self._coerce_optional_str(payload.get("calendar_id"))
            or self._default_calendar(calendar_hint)
        )

        normalized: Dict[str, Any] = {
            "id": event_id,
            "title": title,
            "start": start,
            "end": end,
            "all_day": all_day,
            "location": self._coerce_optional_str(payload.get("location")),
            "description": self._coerce_optional_str(payload.get("description")),
            "attendees": self._normalize_backend_attendees(payload.get("attendees")),
            "created": self._parse_backend_timestamp(payload.get("created")),
            "last_modified": self._parse_backend_timestamp(payload.get("last_modified")),
            "calendar": calendar_name,
            "raw": dict(payload),
        }
        return normalized

    def _normalize_backend_attendees(
        self,
        value: Any,
    ) -> List[Dict[str, Optional[str]]]:
        """Normalize attendees from backend format."""
        if value is None:
            return []
        if isinstance(value, Mapping):
            entries: Iterable[Any] = [value]
        elif isinstance(value, (str, bytes)):
            entries = [value]
        else:
            entries = value

        attendees: List[Dict[str, Optional[str]]] = []
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
                email = email[7:]

            attendees.append(
                {
                    "email": email,
                    "name": name,
                    "role": role,
                    "status": status,
                }
            )

        return attendees

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
            end = start + (
                _dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1)
            )

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

    def _merge_payload(
        self,
        existing: Mapping[str, Any],
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge existing payload with updates."""
        merged = dict(existing)
        for key in (
            "title",
            "start",
            "end",
            "all_day",
            "location",
            "description",
            "attendees",
        ):
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

    def _payload_to_event(
        self,
        payload: Mapping[str, Any],
        calendar_name: str,
    ) -> CalendarEvent:
        """Convert payload dict to CalendarEvent."""
        raw_payload = payload.get("raw")
        raw: Mapping[str, Any] = (
            dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        )

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

    # -------------------------------------------------------------------------
    # Datetime Helpers
    # -------------------------------------------------------------------------

    def _parse_backend_datetime(self, value: Any) -> Optional[_dt.datetime]:
        """Parse datetime from backend response."""
        if isinstance(value, _dt.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=self._default_tz)
            return value
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return _dt.datetime.fromtimestamp(float(value), tz=self._default_tz)

        text = str(value).strip()
        if not text:
            return None

        parsed: Optional[_dt.datetime]
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
            parsed = parsed.replace(tzinfo=self._default_tz)
        return parsed

    def _parse_backend_timestamp(self, value: Any) -> Optional[_dt.datetime]:
        """Parse timestamp as UTC datetime."""
        parsed = self._parse_backend_datetime(value)
        if parsed is None:
            return None
        return parsed.astimezone(_dt.timezone.utc)

    def _serialize_datetime(self, value: _dt.datetime) -> str:
        """Serialize datetime for DBus client."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=self._default_tz)
        return value.astimezone(_dt.timezone.utc).isoformat()

    def _serialize_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Serialize payload for DBus client."""
        serialized: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, _dt.datetime):
                serialized[key] = self._serialize_datetime(value)
            elif isinstance(value, list):
                serialized[key] = [
                    dict(item) if isinstance(item, Mapping) else item for item in value
                ]
            elif isinstance(value, Mapping):
                serialized[key] = dict(value)
            else:
                serialized[key] = value
        return serialized

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _default_calendar(calendar_hint: Optional[str]) -> str:
        """Return default calendar name."""
        if calendar_hint and str(calendar_hint).strip():
            return str(calendar_hint).strip()
        return "default"

    @staticmethod
    def _coerce_optional_str(value: Any) -> Optional[str]:
        """Coerce value to optional string."""
        if value is None:
            return None
        if isinstance(value, list):
            value = value[0]
        text = str(value).strip()
        return text or None

    @staticmethod
    def _is_event_not_found(exc: BaseException) -> bool:
        """Check if exception chain contains EventNotFoundError."""
        current: Optional[BaseException] = exc
        while current is not None:
            if isinstance(current, EventNotFoundError):
                return True
            current = current.__cause__ or current.__context__
        return False


__all__ = ["DBusCalendarBackend"]
