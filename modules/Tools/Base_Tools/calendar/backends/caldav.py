"""CalDAV calendar backend.

This module provides a CalendarBackend implementation that connects to
CalDAV servers such as Nextcloud, Fastmail, iCloud, and other standards-
compliant calendar services.

Requirements:
    caldav>=1.3.0

Example configuration:
    calendars:
      sources:
        nextcloud:
          type: caldav
          url: https://cloud.example.com/remote.php/dav
          username: user@example.com
          password_key: nextcloud_calendar_password  # Key in secrets store
          calendar_id: personal  # Optional: specific calendar path
          write_enabled: true
          sync_mode: on-demand
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
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
    AuthenticationError,
    ConnectionError,
)

if TYPE_CHECKING:
    from ..config import CalendarConfig

logger = logging.getLogger(__name__)


# Try to import caldav, but make it optional
try:
    import caldav
    from caldav.lib.error import AuthorizationError, NotFoundError
    CALDAV_AVAILABLE = True
except ImportError:
    caldav = None  # type: ignore
    AuthorizationError = Exception  # type: ignore
    NotFoundError = Exception  # type: ignore
    CALDAV_AVAILABLE = False


@dataclass
class CalDAVConnectionInfo:
    """Connection information for CalDAV server."""

    url: str
    username: str
    password: str
    calendar_path: Optional[str] = None
    verify_ssl: bool = True


class CalDAVCalendarBackend(CalendarBackend):
    """Calendar backend for CalDAV servers.

    Supports any CalDAV-compliant server including:
    - Nextcloud
    - Fastmail
    - iCloud (Apple)
    - Radicale
    - Baïkal
    - Google Calendar (via CalDAV)

    The backend uses the caldav library to communicate with the server
    and handles event parsing/serialization via iCalendar format.
    """

    def __init__(
        self,
        connection_info: CalDAVConnectionInfo,
        timezone: ZoneInfo,
        calendar_name: str = "caldav",
    ) -> None:
        """Initialize CalDAV backend.

        Args:
            connection_info: Server connection details.
            timezone: Default timezone for events.
            calendar_name: Name identifier for this calendar.
        """
        if not CALDAV_AVAILABLE:
            raise ImportError(
                "caldav package is required for CalDAV support. "
                "Install with: pip install caldav"
            )

        self._connection = connection_info
        self._timezone = timezone
        self._calendar_name = calendar_name
        self._client: Optional[Any] = None
        self._calendar: Optional[Any] = None

    @classmethod
    def from_config(cls, config: "CalendarConfig") -> "CalDAVCalendarBackend":
        """Create backend from configuration.

        Args:
            config: Calendar configuration object.

        Returns:
            Configured CalDAVCalendarBackend instance.
        """
        if not CALDAV_AVAILABLE:
            raise ImportError(
                "caldav package is required for CalDAV support. "
                "Install with: pip install caldav"
            )

        # Get password from secrets or config
        password = cls._resolve_password(config)

        connection_info = CalDAVConnectionInfo(
            url=config.url or "",
            username=config.username or "",
            password=password,
            calendar_path=config.calendar_id,
            verify_ssl=getattr(config, "verify_ssl", True),
        )

        timezone = ZoneInfo(config.timezone) if config.timezone else ZoneInfo("UTC")

        return cls(
            connection_info=connection_info,
            timezone=timezone,
            calendar_name=config.name,
        )

    @staticmethod
    def _resolve_password(config: "CalendarConfig") -> str:
        """Resolve password from secrets store or config."""
        # Try password_key from secrets store first
        password_key = getattr(config, "password_key", None)
        if password_key:
            try:
                from core.services.secrets import SecretService
                secrets = SecretService.get_instance()
                password = secrets.get_secret(password_key)
                if password:
                    return password
            except ImportError:
                logger.debug("SecretService not available")
            except Exception as e:
                logger.warning("Failed to retrieve password from secrets: %s", e)

        # Fall back to direct password (not recommended)
        return getattr(config, "password", "") or ""

    def _ensure_connected(self) -> None:
        """Ensure we have an active connection to the CalDAV server."""
        if self._client is not None and self._calendar is not None:
            return

        try:
            self._client = caldav.DAVClient(
                url=self._connection.url,
                username=self._connection.username,
                password=self._connection.password,
                ssl_verify_cert=self._connection.verify_ssl,
            )

            principal = self._client.principal()

            if self._connection.calendar_path:
                # Try to find specific calendar
                calendars = principal.calendars()
                for cal in calendars:
                    if (
                        self._connection.calendar_path in str(cal.url)
                        or cal.name == self._connection.calendar_path
                    ):
                        self._calendar = cal
                        break

                if self._calendar is None:
                    # Try to get by URL path
                    try:
                        self._calendar = self._client.calendar(
                            url=self._connection.calendar_path
                        )
                    except Exception:
                        logger.warning(
                            "Calendar '%s' not found, using first available",
                            self._connection.calendar_path,
                        )
                        if calendars:
                            self._calendar = calendars[0]

            if self._calendar is None:
                # Use first available calendar
                calendars = principal.calendars()
                if calendars:
                    self._calendar = calendars[0]
                else:
                    raise CalendarBackendError("No calendars found on CalDAV server")

            logger.info(
                "Connected to CalDAV calendar: %s",
                getattr(self._calendar, "name", "unknown"),
            )

        except AuthorizationError as e:
            raise AuthenticationError(f"CalDAV authentication failed: {e}") from e
        except Exception as e:
            self._client = None
            self._calendar = None
            raise ConnectionError(f"Failed to connect to CalDAV server: {e}") from e

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        """List events within a time range."""
        self._ensure_connected()

        try:
            # CalDAV search by date range
            events_raw = self._calendar.search(
                start=start,
                end=end,
                event=True,
                expand=True,
            )

            events: List[CalendarEvent] = []
            for event_obj in events_raw:
                try:
                    parsed = self._parse_caldav_event(event_obj)
                    if parsed:
                        events.append(parsed)
                except Exception as e:
                    logger.warning("Failed to parse CalDAV event: %s", e)
                    continue

            # Sort by start time
            events.sort(key=lambda e: e.start)
            return events

        except Exception as e:
            raise CalendarBackendError(f"Failed to list CalDAV events: {e}") from e

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Get a specific event by ID."""
        self._ensure_connected()

        try:
            # CalDAV uses href/URL for event lookup
            event_obj = self._calendar.event_by_uid(event_id)
            parsed = self._parse_caldav_event(event_obj)
            if parsed is None:
                raise EventNotFoundError(f"Event {event_id} not found")
            return parsed

        except NotFoundError:
            raise EventNotFoundError(f"Event {event_id} not found")
        except EventNotFoundError:
            raise
        except Exception as e:
            raise CalendarBackendError(f"Failed to get CalDAV event: {e}") from e

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        """Search for events matching a query string."""
        # CalDAV doesn't have native text search, so we list and filter
        all_events = await self.list_events(start, end, calendar)

        if not query:
            return all_events

        query_lower = query.lower()
        matching: List[CalendarEvent] = []

        for event in all_events:
            if (
                query_lower in event.title.lower()
                or (event.description and query_lower in event.description.lower())
                or (event.location and query_lower in event.location.lower())
            ):
                matching.append(event)

        return matching

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Create a new event."""
        self._ensure_connected()

        try:
            # Build iCalendar VEVENT
            event_id = str(payload.get("id") or uuid4())
            ical_data = self._build_ical_event(event_id, payload)

            # Save to CalDAV server
            event_obj = self._calendar.save_event(ical_data)

            # Parse back the created event
            parsed = self._parse_caldav_event(event_obj)
            if parsed is None:
                # Return based on payload if parse fails
                return self._payload_to_event(event_id, payload)
            return parsed

        except Exception as e:
            raise CalendarBackendError(f"Failed to create CalDAV event: {e}") from e

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Update an existing event."""
        self._ensure_connected()

        try:
            # Get existing event
            event_obj = self._calendar.event_by_uid(event_id)

            # Get current values and merge with updates
            existing = self._parse_caldav_event(event_obj)
            if existing is None:
                raise EventNotFoundError(f"Event {event_id} not found")

            merged_payload = self._merge_event_payload(existing, payload)

            # Build updated iCalendar data
            ical_data = self._build_ical_event(event_id, merged_payload)

            # Delete old and create new (atomic update)
            event_obj.delete()
            new_event = self._calendar.save_event(ical_data)

            parsed = self._parse_caldav_event(new_event)
            if parsed is None:
                return self._payload_to_event(event_id, merged_payload)
            return parsed

        except NotFoundError:
            raise EventNotFoundError(f"Event {event_id} not found")
        except EventNotFoundError:
            raise
        except Exception as e:
            raise CalendarBackendError(f"Failed to update CalDAV event: {e}") from e

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        """Delete an event."""
        self._ensure_connected()

        try:
            event_obj = self._calendar.event_by_uid(event_id)
            event_obj.delete()
            logger.info("Deleted CalDAV event: %s", event_id)

        except NotFoundError:
            raise EventNotFoundError(f"Event {event_id} not found")
        except Exception as e:
            raise CalendarBackendError(f"Failed to delete CalDAV event: {e}") from e

    def _parse_caldav_event(self, event_obj: Any) -> Optional[CalendarEvent]:
        """Parse a caldav event object into CalendarEvent."""
        try:
            # Get the iCalendar data
            ical = event_obj.icalendar_component

            # Extract UID
            uid = str(ical.get("uid", ""))
            if not uid:
                uid = str(uuid4())

            # Extract summary/title
            title = str(ical.get("summary", "Untitled event"))

            # Extract start/end times
            dtstart = ical.get("dtstart")
            dtend = ical.get("dtend")

            if dtstart is None:
                return None

            start = dtstart.dt
            if isinstance(start, _dt.date) and not isinstance(start, _dt.datetime):
                # All-day event
                start = _dt.datetime.combine(start, _dt.time.min, tzinfo=self._timezone)
                all_day = True
            else:
                if start.tzinfo is None:
                    start = start.replace(tzinfo=self._timezone)
                all_day = False

            if dtend is not None:
                end = dtend.dt
                if isinstance(end, _dt.date) and not isinstance(end, _dt.datetime):
                    end = _dt.datetime.combine(end, _dt.time.min, tzinfo=self._timezone)
                elif end.tzinfo is None:
                    end = end.replace(tzinfo=self._timezone)
            else:
                end = start + _dt.timedelta(hours=1)

            # Extract optional fields
            location = str(ical.get("location", "")) or None
            description = str(ical.get("description", "")) or None

            # Extract attendees
            attendees: List[Dict[str, Optional[str]]] = []
            for attendee in ical.get("attendee", []):
                email = str(attendee).replace("mailto:", "")
                name = attendee.params.get("cn", [None])[0] if hasattr(attendee, "params") else None
                role = attendee.params.get("role", [None])[0] if hasattr(attendee, "params") else None
                status = attendee.params.get("partstat", [None])[0] if hasattr(attendee, "params") else None
                attendees.append({
                    "email": email,
                    "name": name,
                    "role": role,
                    "status": status,
                })

            return CalendarEvent(
                id=uid,
                title=title,
                start=start,
                end=end,
                all_day=all_day,
                location=location,
                description=description,
                calendar=self._calendar_name,
                attendees=attendees,
                raw={"ical": str(event_obj.data)},
            )

        except Exception as e:
            logger.warning("Failed to parse CalDAV event: %s", e)
            return None

    def _build_ical_event(
        self,
        uid: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Build iCalendar VEVENT string from payload."""
        now = _dt.datetime.now(_dt.timezone.utc)
        dtstamp = now.strftime("%Y%m%dT%H%M%SZ")

        title = str(payload.get("title", "Untitled event"))
        start = payload.get("start")
        end = payload.get("end")
        all_day = bool(payload.get("all_day", False))
        location = payload.get("location", "")
        description = payload.get("description", "")

        # Format dates
        if all_day:
            if isinstance(start, _dt.datetime):
                dtstart = start.strftime("%Y%m%d")
            else:
                dtstart = start.strftime("%Y%m%d") if hasattr(start, "strftime") else str(start)

            if end and isinstance(end, _dt.datetime):
                dtend = end.strftime("%Y%m%d")
            elif end and hasattr(end, "strftime"):
                dtend = end.strftime("%Y%m%d")
            else:
                dtend = dtstart
            dtstart_line = f"DTSTART;VALUE=DATE:{dtstart}"
            dtend_line = f"DTEND;VALUE=DATE:{dtend}"
        else:
            if isinstance(start, _dt.datetime):
                dtstart = start.strftime("%Y%m%dT%H%M%SZ") if start.tzinfo else start.strftime("%Y%m%dT%H%M%S")
            else:
                dtstart = str(start)

            if end and isinstance(end, _dt.datetime):
                dtend = end.strftime("%Y%m%dT%H%M%SZ") if end.tzinfo else end.strftime("%Y%m%dT%H%M%S")
            elif end:
                dtend = str(end)
            else:
                dtend = dtstart
            dtstart_line = f"DTSTART:{dtstart}"
            dtend_line = f"DTEND:{dtend}"

        # Build VEVENT
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//ATLAS//CalDAV Backend//EN",
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            dtstart_line,
            dtend_line,
            f"SUMMARY:{self._escape_ical(title)}",
        ]

        if location:
            lines.append(f"LOCATION:{self._escape_ical(str(location))}")
        if description:
            lines.append(f"DESCRIPTION:{self._escape_ical(str(description))}")

        # Add attendees
        attendees = payload.get("attendees", [])
        for attendee in attendees:
            email = attendee.get("email", "")
            if email:
                cn = attendee.get("name", "")
                if cn:
                    lines.append(f"ATTENDEE;CN={self._escape_ical(cn)}:mailto:{email}")
                else:
                    lines.append(f"ATTENDEE:mailto:{email}")

        lines.extend([
            "END:VEVENT",
            "END:VCALENDAR",
        ])

        return "\r\n".join(lines)

    @staticmethod
    def _escape_ical(text: str) -> str:
        """Escape text for iCalendar format."""
        return (
            text.replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )

    def _merge_event_payload(
        self,
        existing: CalendarEvent,
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge existing event with update payload."""
        return {
            "title": updates.get("title", existing.title),
            "start": updates.get("start", existing.start),
            "end": updates.get("end", existing.end),
            "all_day": updates.get("all_day", existing.all_day),
            "location": updates.get("location", existing.location),
            "description": updates.get("description", existing.description),
            "attendees": updates.get("attendees", existing.attendees),
        }

    def _payload_to_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
    ) -> CalendarEvent:
        """Convert payload to CalendarEvent."""
        return CalendarEvent(
            id=event_id,
            title=str(payload.get("title", "Untitled event")),
            start=payload.get("start", _dt.datetime.now(_dt.timezone.utc)),
            end=payload.get("end", _dt.datetime.now(_dt.timezone.utc)),
            all_day=bool(payload.get("all_day", False)),
            location=payload.get("location"),
            description=payload.get("description"),
            calendar=self._calendar_name,
            attendees=payload.get("attendees", []),
            raw={},
        )
