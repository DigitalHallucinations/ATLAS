"""Google Calendar backend.

This module provides a calendar backend that integrates with Google Calendar
API using OAuth2 authentication.

Requirements:
    - google-auth
    - google-auth-oauthlib
    - google-api-python-client

Example:
    backend = GoogleCalendarBackend.from_config(config)
    await backend.connect()
    events = await backend.list_events(start, end)
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
from pathlib import Path
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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import (
    CalendarBackend,
    CalendarBackendError,
    CalendarEvent,
    EventNotFoundError,
    AuthenticationError,
    RateLimitError,
    ConnectionError as CalendarConnectionError,
)

if TYPE_CHECKING:
    from ..config import CalendarConfig

logger = logging.getLogger(__name__)

# Google Calendar API scopes
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]


class GoogleCalendarBackend(CalendarBackend):
    """Backend for Google Calendar API integration.

    Supports OAuth2 authentication for accessing Google Calendar accounts.
    Credentials are stored locally for reuse across sessions.

    Example:
        backend = GoogleCalendarBackend(
            credentials_path=Path("~/.config/atlas/google_creds.json"),
            token_path=Path("~/.config/atlas/google_token.json"),
            calendar_id="primary",
        )
        await backend.connect()
        events = await backend.list_events(start, end)
    """

    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        calendar_id: str = "primary",
        *,
        default_timezone: Optional[ZoneInfo] = None,
        calendar_name: Optional[str] = None,
    ) -> None:
        """Initialize the Google Calendar backend.

        Args:
            credentials_path: Path to OAuth2 client credentials JSON file.
            token_path: Path to store/load cached OAuth2 tokens.
            calendar_id: Google Calendar ID (default: "primary").
            default_timezone: Default timezone for events.
            calendar_name: Optional name override for this calendar.
        """
        self._credentials_path = Path(credentials_path).expanduser()
        self._token_path = Path(token_path).expanduser()
        self._calendar_id = calendar_id
        self._default_tz = default_timezone or ZoneInfo("UTC")
        self._calendar_name = calendar_name or "google"
        self._service: Optional[Any] = None
        self._credentials: Optional[Any] = None

    @classmethod
    def from_config(cls, config: "CalendarConfig") -> "GoogleCalendarBackend":
        """Create a GoogleCalendarBackend from configuration.

        Args:
            config: Calendar configuration.

        Returns:
            Configured GoogleCalendarBackend instance.

        Configuration options used:
            - credentials_path: Path to OAuth2 client credentials
            - token_path: Path to store cached tokens
            - calendar_id: Google Calendar ID (default: "primary")
            - timezone: Default timezone
        """
        creds_path = config.credentials_path or Path(
            "~/.config/atlas/google_credentials.json"
        )
        token_path = config.token_path or Path(
            "~/.config/atlas/google_token.json"
        )
        calendar_id = config.calendar_id or "primary"

        timezone: Optional[ZoneInfo] = None
        if config.timezone:
            try:
                timezone = ZoneInfo(config.timezone)
            except (ZoneInfoNotFoundError, ValueError):
                logger.warning("Invalid timezone '%s', using UTC", config.timezone)

        return cls(
            credentials_path=creds_path,
            token_path=token_path,
            calendar_id=calendar_id,
            default_timezone=timezone,
            calendar_name=config.name,
        )

    @property
    def name(self) -> str:
        return self._calendar_name

    @property
    def is_connected(self) -> bool:
        """Check if the backend is authenticated and connected."""
        return self._service is not None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate and connect to Google Calendar API.

        Raises:
            AuthenticationError: If authentication fails.
        """
        try:
            await asyncio.to_thread(self._authenticate)
        except Exception as exc:
            raise AuthenticationError(
                f"Failed to authenticate with Google Calendar: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Disconnect from Google Calendar API."""
        self._service = None
        self._credentials = None

    def _authenticate(self) -> None:
        """Perform OAuth2 authentication (blocking)."""
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise AuthenticationError(
                "Google Calendar dependencies not installed. "
                "Install with: pip install google-auth google-auth-oauthlib "
                "google-api-python-client"
            ) from exc

        creds: Optional[Credentials] = None

        # Load existing token
        if self._token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(self._token_path), SCOPES
                )
            except Exception as exc:
                logger.warning("Failed to load cached token: %s", exc)
                creds = None

        # Refresh or obtain new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as exc:
                    logger.warning("Failed to refresh token: %s", exc)
                    creds = None

            if not creds:
                if not self._credentials_path.exists():
                    raise AuthenticationError(
                        f"OAuth2 credentials file not found: {self._credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for future use
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._token_path, "w") as token_file:
                token_file.write(creds.to_json())

        self._credentials = creds
        self._service = build("calendar", "v3", credentials=creds)

    def _ensure_connected(self) -> Any:
        """Ensure the service is connected."""
        if self._service is None:
            raise CalendarConnectionError(
                "Not connected to Google Calendar. Call connect() first."
            )
        return self._service

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        service = self._ensure_connected()

        calendar_id = calendar or self._calendar_id
        time_min = self._format_datetime(start)
        time_max = self._format_datetime(end)

        try:
            result = await asyncio.to_thread(
                lambda: service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                )
                .execute()
            )
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to list Google Calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        for item in result.get("items", []):
            event = self._parse_event(item, calendar_id)
            if event:
                events.append(event)

        return events

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        service = self._ensure_connected()
        calendar_id = calendar or self._calendar_id

        try:
            result = await asyncio.to_thread(
                lambda: service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise EventNotFoundError(
                    f"Google Calendar event '{event_id}' not found"
                ) from exc
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to get Google Calendar event"
            ) from exc

        event = self._parse_event(result, calendar_id)
        if not event:
            raise EventNotFoundError(
                f"Google Calendar event '{event_id}' could not be parsed"
            )
        return event

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        service = self._ensure_connected()

        calendar_id = calendar or self._calendar_id
        time_min = self._format_datetime(start)
        time_max = self._format_datetime(end)

        try:
            result = await asyncio.to_thread(
                lambda: service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    q=query,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                )
                .execute()
            )
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to search Google Calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        for item in result.get("items", []):
            event = self._parse_event(item, calendar_id)
            if event:
                events.append(event)

        return events

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        service = self._ensure_connected()
        calendar_id = calendar or self._calendar_id

        body = self._build_event_body(payload)

        try:
            result = await asyncio.to_thread(
                lambda: service.events()
                .insert(calendarId=calendar_id, body=body)
                .execute()
            )
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to create Google Calendar event"
            ) from exc

        event = self._parse_event(result, calendar_id)
        if not event:
            raise CalendarBackendError(
                "Failed to parse created Google Calendar event"
            )
        return event

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        service = self._ensure_connected()
        calendar_id = calendar or self._calendar_id

        # Get existing event
        try:
            existing = await asyncio.to_thread(
                lambda: service.events()
                .get(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise EventNotFoundError(
                    f"Google Calendar event '{event_id}' not found"
                ) from exc
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to fetch Google Calendar event for update"
            ) from exc

        # Merge updates
        body = self._merge_event_body(existing, payload)

        try:
            result = await asyncio.to_thread(
                lambda: service.events()
                .update(calendarId=calendar_id, eventId=event_id, body=body)
                .execute()
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise EventNotFoundError(
                    f"Google Calendar event '{event_id}' not found"
                ) from exc
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to update Google Calendar event"
            ) from exc

        event = self._parse_event(result, calendar_id)
        if not event:
            raise CalendarBackendError(
                "Failed to parse updated Google Calendar event"
            )
        return event

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        service = self._ensure_connected()
        calendar_id = calendar or self._calendar_id

        try:
            await asyncio.to_thread(
                lambda: service.events()
                .delete(calendarId=calendar_id, eventId=event_id)
                .execute()
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise EventNotFoundError(
                    f"Google Calendar event '{event_id}' not found"
                ) from exc
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to delete Google Calendar event"
            ) from exc

    # -------------------------------------------------------------------------
    # Event Parsing
    # -------------------------------------------------------------------------

    def _parse_event(
        self,
        item: Mapping[str, Any],
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Parse Google Calendar event into CalendarEvent."""
        event_id = item.get("id")
        if not event_id:
            return None

        summary = item.get("summary", "Untitled event")

        # Parse start/end (can be date or dateTime)
        start_data = item.get("start", {})
        end_data = item.get("end", {})

        start, all_day = self._parse_google_datetime(start_data)
        end, _ = self._parse_google_datetime(end_data)

        if not start:
            return None
        if not end:
            end = start + (_dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1))

        # Parse attendees
        attendees: List[Dict[str, Optional[str]]] = []
        for attendee in item.get("attendees", []):
            attendees.append({
                "email": attendee.get("email"),
                "name": attendee.get("displayName"),
                "role": "organizer" if attendee.get("organizer") else None,
                "status": attendee.get("responseStatus"),
            })

        return CalendarEvent(
            id=event_id,
            title=summary,
            start=start,
            end=end,
            all_day=all_day,
            location=item.get("location"),
            description=item.get("description"),
            calendar=calendar_id,
            attendees=attendees,
            raw=dict(item),
        )

    def _parse_google_datetime(
        self,
        data: Mapping[str, Any],
    ) -> tuple[Optional[_dt.datetime], bool]:
        """Parse Google Calendar dateTime or date field."""
        if not data:
            return None, False

        # Check for all-day event (date field)
        date_str = data.get("date")
        if date_str:
            try:
                parsed_date = _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                return (
                    _dt.datetime.combine(
                        parsed_date, _dt.time.min, tzinfo=self._default_tz
                    ),
                    True,
                )
            except ValueError:
                return None, False

        # Regular dateTime
        datetime_str = data.get("dateTime")
        if datetime_str:
            try:
                parsed = _dt.datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                return parsed, False
            except ValueError:
                return None, False

        return None, False

    # -------------------------------------------------------------------------
    # Event Building
    # -------------------------------------------------------------------------

    def _build_event_body(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Build Google Calendar event body from payload."""
        body: Dict[str, Any] = {}

        # Summary/title
        title = payload.get("title")
        if title:
            body["summary"] = str(title)

        # Start/end times
        start = payload.get("start")
        end = payload.get("end")
        all_day = bool(payload.get("all_day", False))

        if isinstance(start, _dt.datetime):
            body["start"] = self._format_google_datetime(start, all_day)
        if isinstance(end, _dt.datetime):
            body["end"] = self._format_google_datetime(end, all_day)

        # Location and description
        location = payload.get("location")
        if location:
            body["location"] = str(location)
        description = payload.get("description")
        if description:
            body["description"] = str(description)

        # Attendees
        attendees = payload.get("attendees", [])
        if attendees:
            body["attendees"] = [
                {"email": a.get("email"), "displayName": a.get("name")}
                for a in attendees
                if a.get("email")
            ]

        return body

    def _merge_event_body(
        self,
        existing: Mapping[str, Any],
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge existing event with updates."""
        body = dict(existing)

        if "title" in updates:
            body["summary"] = updates["title"]
        if "location" in updates:
            body["location"] = updates["location"]
        if "description" in updates:
            body["description"] = updates["description"]

        all_day = bool(updates.get("all_day", existing.get("start", {}).get("date")))

        if "start" in updates and isinstance(updates["start"], _dt.datetime):
            body["start"] = self._format_google_datetime(updates["start"], all_day)
        if "end" in updates and isinstance(updates["end"], _dt.datetime):
            body["end"] = self._format_google_datetime(updates["end"], all_day)

        if "attendees" in updates:
            body["attendees"] = [
                {"email": a.get("email"), "displayName": a.get("name")}
                for a in updates["attendees"]
                if a.get("email")
            ]

        return body

    def _format_google_datetime(
        self,
        dt: _dt.datetime,
        all_day: bool,
    ) -> Dict[str, str]:
        """Format datetime for Google Calendar API."""
        if all_day:
            return {"date": dt.date().isoformat()}
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._default_tz)
        return {"dateTime": dt.isoformat(), "timeZone": str(self._default_tz)}

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _format_datetime(self, dt: _dt.datetime) -> str:
        """Format datetime as ISO string for API."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._default_tz)
        return dt.isoformat()

    def _handle_api_error(self, exc: Exception) -> None:
        """Handle Google API errors and raise appropriate exceptions."""
        try:
            from googleapiclient.errors import HttpError
        except ImportError:
            return

        if isinstance(exc, HttpError):
            status = exc.resp.status
            if status == 401:
                raise AuthenticationError(
                    "Google Calendar authentication expired or invalid"
                ) from exc
            elif status == 429:
                raise RateLimitError(
                    "Google Calendar API rate limit exceeded"
                ) from exc
            elif status == 404:
                pass  # Let caller handle not found

    def _is_not_found_error(self, exc: Exception) -> bool:
        """Check if exception is a 404 not found error."""
        try:
            from googleapiclient.errors import HttpError
        except ImportError:
            return False

        if isinstance(exc, HttpError):
            return exc.resp.status == 404
        return False


__all__ = ["GoogleCalendarBackend"]
