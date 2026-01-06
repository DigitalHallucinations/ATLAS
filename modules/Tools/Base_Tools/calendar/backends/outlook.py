"""Microsoft Outlook/Office 365 Calendar backend.

This module provides a calendar backend that integrates with Microsoft Graph API
for accessing Outlook/Office 365 calendars using OAuth2 authentication.

Requirements:
    - msal
    - requests (or httpx for async)

Example:
    backend = OutlookCalendarBackend.from_config(config)
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

# Microsoft Graph API endpoints
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
AUTHORITY_URL = "https://login.microsoftonline.com"

# Default scopes for calendar access
DEFAULT_SCOPES = [
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/User.Read",
]


class OutlookCalendarBackend(CalendarBackend):
    """Backend for Microsoft Outlook/Office 365 Calendar via Graph API.

    Supports OAuth2 authentication using MSAL (Microsoft Authentication Library)
    for accessing Microsoft 365 calendars.

    Example:
        backend = OutlookCalendarBackend(
            client_id="your-app-client-id",
            tenant_id="your-tenant-id",
            token_cache_path=Path("~/.config/atlas/outlook_token.json"),
            calendar_id="primary",
        )
        await backend.connect()
        events = await backend.list_events(start, end)
    """

    def __init__(
        self,
        client_id: str,
        tenant_id: str = "common",
        client_secret: Optional[str] = None,
        token_cache_path: Optional[Path] = None,
        calendar_id: Optional[str] = None,
        *,
        default_timezone: Optional[ZoneInfo] = None,
        calendar_name: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Initialize the Outlook Calendar backend.

        Args:
            client_id: Azure AD application (client) ID.
            tenant_id: Azure AD tenant ID or "common" for multi-tenant.
            client_secret: Optional client secret for confidential apps.
            token_cache_path: Path to store/load cached tokens.
            calendar_id: Outlook Calendar ID (default: user's default calendar).
            default_timezone: Default timezone for events.
            calendar_name: Optional name override for this calendar.
            scopes: OAuth2 scopes (defaults to calendar read/write).
        """
        self._client_id = client_id
        self._tenant_id = tenant_id
        self._client_secret = client_secret
        self._token_cache_path = (
            Path(token_cache_path).expanduser() if token_cache_path else None
        )
        self._calendar_id = calendar_id
        self._default_tz = default_timezone or ZoneInfo("UTC")
        self._calendar_name = calendar_name or "outlook"
        self._scopes = scopes or DEFAULT_SCOPES

        self._msal_app: Optional[Any] = None
        self._access_token: Optional[str] = None
        self._session: Optional[Any] = None

    @classmethod
    def from_config(cls, config: "CalendarConfig") -> "OutlookCalendarBackend":
        """Create an OutlookCalendarBackend from configuration.

        Args:
            config: Calendar configuration.

        Returns:
            Configured OutlookCalendarBackend instance.

        Configuration options used:
            - client_id: Azure AD application ID (required)
            - tenant_id: Azure AD tenant ID (default: "common")
            - client_secret: Optional client secret
            - token_path: Path to store cached tokens
            - calendar_id: Outlook Calendar ID
            - timezone: Default timezone
        """
        client_id = config.client_id
        if not client_id:
            raise ValueError("client_id is required for Outlook Calendar backend")

        tenant_id = config.tenant_id or "common"
        token_path = config.token_path or Path("~/.config/atlas/outlook_token.json")
        calendar_id = config.calendar_id

        timezone: Optional[ZoneInfo] = None
        if config.timezone:
            try:
                timezone = ZoneInfo(config.timezone)
            except (ZoneInfoNotFoundError, ValueError):
                logger.warning("Invalid timezone '%s', using UTC", config.timezone)

        return cls(
            client_id=client_id,
            tenant_id=tenant_id,
            client_secret=config.client_secret,
            token_cache_path=token_path,
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
        return self._access_token is not None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate and connect to Microsoft Graph API.

        Raises:
            AuthenticationError: If authentication fails.
        """
        try:
            await asyncio.to_thread(self._authenticate)
        except Exception as exc:
            raise AuthenticationError(
                f"Failed to authenticate with Outlook Calendar: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Disconnect from Microsoft Graph API."""
        self._access_token = None
        if self._session:
            try:
                await asyncio.to_thread(self._session.close)
            except Exception:
                pass
            self._session = None

    def _authenticate(self) -> None:
        """Perform OAuth2 authentication (blocking)."""
        try:
            import msal
        except ImportError as exc:
            raise AuthenticationError(
                "Microsoft authentication dependencies not installed. "
                "Install with: pip install msal"
            ) from exc

        authority = f"{AUTHORITY_URL}/{self._tenant_id}"

        # Load token cache
        cache = msal.SerializableTokenCache()
        if self._token_cache_path and self._token_cache_path.exists():
            try:
                cache.deserialize(self._token_cache_path.read_text())
            except Exception as exc:
                logger.warning("Failed to load token cache: %s", exc)

        # Create MSAL application
        if self._client_secret:
            self._msal_app = msal.ConfidentialClientApplication(
                self._client_id,
                authority=authority,
                client_credential=self._client_secret,
                token_cache=cache,
            )
        else:
            self._msal_app = msal.PublicClientApplication(
                self._client_id,
                authority=authority,
                token_cache=cache,
            )

        # Try to get token from cache
        accounts = self._msal_app.get_accounts()
        result = None
        if accounts:
            result = self._msal_app.acquire_token_silent(
                self._scopes, account=accounts[0]
            )

        # If no cached token, acquire interactively
        if not result:
            if self._client_secret:
                # For confidential apps, use client credentials flow
                result = self._msal_app.acquire_token_for_client(scopes=self._scopes)
            else:
                # For public apps, use device code flow
                flow = self._msal_app.initiate_device_flow(scopes=self._scopes)
                if "user_code" not in flow:
                    raise AuthenticationError(
                        f"Failed to initiate device flow: {flow.get('error_description', 'Unknown error')}"
                    )
                logger.info(
                    "To sign in, visit %s and enter code: %s",
                    flow["verification_uri"],
                    flow["user_code"],
                )
                result = self._msal_app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            error = result.get("error_description", result.get("error", "Unknown error"))
            raise AuthenticationError(f"Failed to acquire token: {error}")

        self._access_token = result["access_token"]

        # Save token cache
        if self._token_cache_path and cache.has_state_changed:
            self._token_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_cache_path.write_text(cache.serialize())

    def _ensure_connected(self) -> str:
        """Ensure the backend is connected and return access token."""
        if self._access_token is None:
            raise CalendarConnectionError(
                "Not connected to Outlook Calendar. Call connect() first."
            )
        return self._access_token

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        token = self._ensure_connected()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": f'outlook.timezone="{self._default_tz}"',
        }

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
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/calendarView"

        params = {
            "startDateTime": self._format_datetime(start),
            "endDateTime": self._format_datetime(end),
            "$orderby": "start/dateTime",
            "$top": 1000,
        }

        try:
            response = await asyncio.to_thread(
                lambda: requests.get(url, headers=self._get_headers(), params=params)
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to list Outlook Calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        for item in data.get("value", []):
            event = self._parse_event(item, calendar or "default")
            if event:
                events.append(event)

        return events

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        try:
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/events/{event_id}"

        try:
            response = await asyncio.to_thread(
                lambda: requests.get(url, headers=self._get_headers())
            )
            if response.status_code == 404:
                raise EventNotFoundError(
                    f"Outlook Calendar event '{event_id}' not found"
                )
            response.raise_for_status()
            data = response.json()
        except EventNotFoundError:
            raise
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to get Outlook Calendar event"
            ) from exc

        event = self._parse_event(data, calendar or "default")
        if not event:
            raise EventNotFoundError(
                f"Outlook Calendar event '{event_id}' could not be parsed"
            )
        return event

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        try:
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/calendarView"

        # Microsoft Graph uses OData filter
        params = {
            "startDateTime": self._format_datetime(start),
            "endDateTime": self._format_datetime(end),
            "$filter": f"contains(subject,'{query}') or contains(bodyPreview,'{query}')",
            "$orderby": "start/dateTime",
            "$top": 1000,
        }

        try:
            response = await asyncio.to_thread(
                lambda: requests.get(url, headers=self._get_headers(), params=params)
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to search Outlook Calendar events"
            ) from exc

        events: List[CalendarEvent] = []
        for item in data.get("value", []):
            event = self._parse_event(item, calendar or "default")
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
        try:
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/events"

        body = self._build_event_body(payload)

        try:
            response = await asyncio.to_thread(
                lambda: requests.post(
                    url, headers=self._get_headers(), json=body
                )
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to create Outlook Calendar event"
            ) from exc

        event = self._parse_event(data, calendar or "default")
        if not event:
            raise CalendarBackendError(
                "Failed to parse created Outlook Calendar event"
            )
        return event

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        try:
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/events/{event_id}"

        body = self._build_event_body(payload)

        try:
            response = await asyncio.to_thread(
                lambda: requests.patch(
                    url, headers=self._get_headers(), json=body
                )
            )
            if response.status_code == 404:
                raise EventNotFoundError(
                    f"Outlook Calendar event '{event_id}' not found"
                )
            response.raise_for_status()
            data = response.json()
        except EventNotFoundError:
            raise
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to update Outlook Calendar event"
            ) from exc

        event = self._parse_event(data, calendar or "default")
        if not event:
            raise CalendarBackendError(
                "Failed to parse updated Outlook Calendar event"
            )
        return event

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        try:
            import requests
        except ImportError as exc:
            raise CalendarBackendError(
                "requests library required for Outlook Calendar"
            ) from exc

        calendar_path = self._calendar_path(calendar)
        url = f"{GRAPH_BASE_URL}{calendar_path}/events/{event_id}"

        try:
            response = await asyncio.to_thread(
                lambda: requests.delete(url, headers=self._get_headers())
            )
            if response.status_code == 404:
                raise EventNotFoundError(
                    f"Outlook Calendar event '{event_id}' not found"
                )
            response.raise_for_status()
        except EventNotFoundError:
            raise
        except Exception as exc:
            self._handle_api_error(exc)
            raise CalendarBackendError(
                "Failed to delete Outlook Calendar event"
            ) from exc

    # -------------------------------------------------------------------------
    # Event Parsing
    # -------------------------------------------------------------------------

    def _parse_event(
        self,
        item: Mapping[str, Any],
        calendar_id: str,
    ) -> Optional[CalendarEvent]:
        """Parse Microsoft Graph event into CalendarEvent."""
        event_id = item.get("id")
        if not event_id:
            return None

        subject = item.get("subject", "Untitled event")

        # Parse start/end
        start_data = item.get("start", {})
        end_data = item.get("end", {})

        start = self._parse_outlook_datetime(start_data)
        end = self._parse_outlook_datetime(end_data)
        all_day = item.get("isAllDay", False)

        if not start:
            return None
        if not end:
            end = start + (_dt.timedelta(days=1) if all_day else _dt.timedelta(hours=1))

        # Parse location
        location_data = item.get("location", {})
        location = location_data.get("displayName") if location_data else None

        # Parse body/description
        body_data = item.get("body", {})
        description = None
        if body_data:
            content = body_data.get("content", "")
            content_type = body_data.get("contentType", "text")
            if content_type == "text":
                description = content
            else:
                # Strip HTML tags for basic extraction
                import re
                description = re.sub(r"<[^>]+>", "", content).strip()

        # Parse attendees
        attendees: List[Dict[str, Optional[str]]] = []
        for attendee in item.get("attendees", []):
            email_addr = attendee.get("emailAddress", {})
            attendees.append({
                "email": email_addr.get("address"),
                "name": email_addr.get("name"),
                "role": attendee.get("type"),
                "status": attendee.get("status", {}).get("response"),
            })

        return CalendarEvent(
            id=event_id,
            title=subject,
            start=start,
            end=end,
            all_day=all_day,
            location=location,
            description=description,
            calendar=calendar_id,
            attendees=attendees,
            raw=dict(item),
        )

    def _parse_outlook_datetime(
        self,
        data: Mapping[str, Any],
    ) -> Optional[_dt.datetime]:
        """Parse Microsoft Graph dateTime field."""
        if not data:
            return None

        datetime_str = data.get("dateTime")
        if not datetime_str:
            return None

        timezone_str = data.get("timeZone", "UTC")

        try:
            # Parse the datetime string (no timezone info in the string)
            parsed = _dt.datetime.fromisoformat(datetime_str.replace("Z", ""))
            # Apply the specified timezone
            try:
                tz = ZoneInfo(timezone_str)
            except (ZoneInfoNotFoundError, ValueError):
                tz = self._default_tz
            return parsed.replace(tzinfo=tz)
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    # Event Building
    # -------------------------------------------------------------------------

    def _build_event_body(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Build Microsoft Graph event body from payload."""
        body: Dict[str, Any] = {}

        # Subject/title
        title = payload.get("title")
        if title:
            body["subject"] = str(title)

        # Start/end times
        start = payload.get("start")
        end = payload.get("end")
        all_day = bool(payload.get("all_day", False))

        if isinstance(start, _dt.datetime):
            body["start"] = self._format_outlook_datetime(start)
        if isinstance(end, _dt.datetime):
            body["end"] = self._format_outlook_datetime(end)

        body["isAllDay"] = all_day

        # Location
        location = payload.get("location")
        if location:
            body["location"] = {"displayName": str(location)}

        # Description/body
        description = payload.get("description")
        if description:
            body["body"] = {
                "contentType": "text",
                "content": str(description),
            }

        # Attendees
        attendees = payload.get("attendees", [])
        if attendees:
            body["attendees"] = [
                {
                    "emailAddress": {
                        "address": a.get("email"),
                        "name": a.get("name"),
                    },
                    "type": a.get("role", "required"),
                }
                for a in attendees
                if a.get("email")
            ]

        return body

    def _format_outlook_datetime(self, dt: _dt.datetime) -> Dict[str, str]:
        """Format datetime for Microsoft Graph API."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._default_tz)
        return {
            "dateTime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": str(self._default_tz),
        }

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _calendar_path(self, calendar_id: Optional[str]) -> str:
        """Build API path for calendar operations."""
        cal_id = calendar_id or self._calendar_id
        if cal_id:
            return f"/me/calendars/{cal_id}"
        return "/me/calendar"

    def _format_datetime(self, dt: _dt.datetime) -> str:
        """Format datetime as ISO string for API."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._default_tz)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _handle_api_error(self, exc: Exception) -> None:
        """Handle Microsoft Graph API errors."""
        try:
            import requests
        except ImportError:
            return

        if isinstance(exc, requests.HTTPError):
            response = exc.response
            if response is not None:
                status = response.status_code
                if status == 401:
                    raise AuthenticationError(
                        "Outlook Calendar authentication expired or invalid"
                    ) from exc
                elif status == 429:
                    raise RateLimitError(
                        "Microsoft Graph API rate limit exceeded"
                    ) from exc


__all__ = ["OutlookCalendarBackend"]
