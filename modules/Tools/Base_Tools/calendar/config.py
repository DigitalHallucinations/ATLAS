"""Calendar configuration models.

This module defines the configuration dataclasses for the multi-calendar
system, including per-calendar settings and global calendar configuration.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class CalendarBackendType(str, Enum):
    """Supported calendar backend types."""

    ICS = "ics"
    DBUS = "dbus"
    GOOGLE = "google"
    OUTLOOK = "outlook"
    CALDAV = "caldav"
    APPLE = "apple"


class SyncMode(str, Enum):
    """Calendar synchronization modes."""

    REALTIME = "realtime"      # Push notifications / webhooks where supported
    ON_DEMAND = "on-demand"    # Fetch fresh data on each query
    READ_ONLY = "read-only"    # Fetch but never write
    DAILY = "daily"            # Cache for 24h, background refresh
    MANUAL = "manual"          # Only sync when user triggers


@dataclass(slots=True)
class CalendarConfig:
    """Configuration for a single calendar source.

    Attributes:
        name: Unique identifier for this calendar (e.g., "personal", "work").
        backend_type: The type of backend (ics, google, caldav, etc.).
        write_enabled: Whether write operations are allowed.
        sync_mode: How the calendar should be synchronized.
        color: Optional hex color for UI display (e.g., "#4285f4").
        display_name: Human-readable name for UI display.
        priority: Order priority for listing (lower = first).

        # ICS-specific
        path: Local file path for ICS calendars.
        url: Remote URL for ICS or CalDAV calendars.

        # Account-based (Google, Outlook)
        account: Account email or identifier.
        calendar_id: Specific calendar ID within the account.

        # OAuth2 (Google, Outlook)
        credentials_path: Path to OAuth2 client credentials file.
        token_path: Path to store cached OAuth2 tokens.
        client_id: OAuth2 client/application ID.
        client_secret: OAuth2 client secret (optional for public apps).
        tenant_id: Azure AD tenant ID for Outlook (default: "common").

        # CalDAV-specific
        username: Username for CalDAV authentication.
        password_key: Key in secrets store for password (not the password itself).

        # General
        timezone: Timezone override for this calendar.
        extra: Additional backend-specific configuration.
    """

    name: str
    backend_type: CalendarBackendType
    write_enabled: bool = True
    sync_mode: SyncMode = SyncMode.ON_DEMAND
    color: Optional[str] = None
    display_name: Optional[str] = None
    priority: int = 100

    # ICS-specific
    path: Optional[Path] = None
    url: Optional[str] = None

    # Account-based
    account: Optional[str] = None
    calendar_id: Optional[str] = None

    # OAuth2
    credentials_path: Optional[Path] = None
    token_path: Optional[Path] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None

    # CalDAV-specific
    username: Optional[str] = None
    password_key: Optional[str] = None

    # General
    timezone: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.backend_type, str):
            self.backend_type = CalendarBackendType(self.backend_type.lower())
        if isinstance(self.sync_mode, str):
            self.sync_mode = SyncMode(self.sync_mode.lower().replace("_", "-"))
        if isinstance(self.path, str):
            self.path = Path(self.path).expanduser()
        if isinstance(self.credentials_path, str):
            self.credentials_path = Path(self.credentials_path).expanduser()
        if isinstance(self.token_path, str):
            self.token_path = Path(self.token_path).expanduser()

    @property
    def is_read_only(self) -> bool:
        """Return True if this calendar should not accept writes."""
        return not self.write_enabled or self.sync_mode == SyncMode.READ_ONLY

    @property
    def effective_display_name(self) -> str:
        """Return the display name, falling back to the calendar name."""
        return self.display_name or self.name.replace("_", " ").title()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "backend_type": self.backend_type.value,
            "write_enabled": self.write_enabled,
            "sync_mode": self.sync_mode.value,
            "priority": self.priority,
        }
        if self.color:
            result["color"] = self.color
        if self.display_name:
            result["display_name"] = self.display_name
        if self.path:
            result["path"] = str(self.path)
        if self.url:
            result["url"] = self.url
        if self.account:
            result["account"] = self.account
        if self.calendar_id:
            result["calendar_id"] = self.calendar_id
        if self.credentials_path:
            result["credentials_path"] = str(self.credentials_path)
        if self.token_path:
            result["token_path"] = str(self.token_path)
        if self.client_id:
            result["client_id"] = self.client_id
        # Note: client_secret intentionally not serialized for security
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        if self.username:
            result["username"] = self.username
        if self.password_key:
            result["password_key"] = self.password_key
        if self.timezone:
            result["timezone"] = self.timezone
        if self.extra:
            result["extra"] = dict(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalendarConfig":
        """Create a CalendarConfig from a dictionary."""
        return cls(
            name=str(data.get("name", "")),
            backend_type=data.get("backend_type", data.get("type", "ics")),
            write_enabled=bool(data.get("write_enabled", True)),
            sync_mode=data.get("sync_mode", "on-demand"),
            color=data.get("color"),
            display_name=data.get("display_name"),
            priority=int(data.get("priority", 100)),
            path=data.get("path"),
            url=data.get("url"),
            account=data.get("account"),
            calendar_id=data.get("calendar_id"),
            credentials_path=data.get("credentials_path"),
            token_path=data.get("token_path"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            tenant_id=data.get("tenant_id"),
            username=data.get("username"),
            password_key=data.get("password_key"),
            timezone=data.get("timezone"),
            extra=dict(data.get("extra", {})),
        )


@dataclass(slots=True)
class CalendarInfo:
    """Summary information about a configured calendar.

    Used for listing available calendars without exposing full config.
    """

    name: str
    display_name: str
    backend_type: CalendarBackendType
    color: Optional[str]
    write_enabled: bool
    sync_mode: SyncMode
    is_connected: bool = True
    last_sync: Optional[_dt.datetime] = None
    event_count: Optional[int] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        result: Dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "backend_type": self.backend_type.value,
            "write_enabled": self.write_enabled,
            "sync_mode": self.sync_mode.value,
            "is_connected": self.is_connected,
        }
        if self.color:
            result["color"] = self.color
        if self.last_sync:
            result["last_sync"] = self.last_sync.isoformat()
        if self.event_count is not None:
            result["event_count"] = self.event_count
        if self.error_message:
            result["error_message"] = self.error_message
        return result


@dataclass(slots=True)
class CalendarsGlobalConfig:
    """Global configuration for the calendar system.

    Attributes:
        default_calendar: Name of the default calendar for write operations.
        calendars: List of configured calendar sources.
        merge_duplicates: Whether to detect and merge duplicate events.
        parallel_queries: Whether to query backends in parallel.
        query_timeout: Timeout in seconds for backend queries.
    """

    default_calendar: Optional[str] = None
    calendars: List[CalendarConfig] = field(default_factory=list)
    merge_duplicates: bool = False
    parallel_queries: bool = True
    query_timeout: float = 30.0

    def get_calendar(self, name: str) -> Optional[CalendarConfig]:
        """Get a calendar configuration by name."""
        for cal in self.calendars:
            if cal.name == name:
                return cal
        return None

    def get_default_calendar(self) -> Optional[CalendarConfig]:
        """Get the default calendar configuration."""
        if self.default_calendar:
            return self.get_calendar(self.default_calendar)
        if self.calendars:
            # Return first writable calendar as fallback
            for cal in sorted(self.calendars, key=lambda c: c.priority):
                if cal.write_enabled:
                    return cal
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalendarsGlobalConfig":
        """Create from configuration dictionary."""
        sources = data.get("sources", {})
        calendars: List[CalendarConfig] = []

        if isinstance(sources, Mapping):
            for name, config in sources.items():
                if isinstance(config, Mapping):
                    cal_data = dict(config)
                    cal_data["name"] = name
                    calendars.append(CalendarConfig.from_dict(cal_data))

        return cls(
            default_calendar=data.get("default_calendar"),
            calendars=calendars,
            merge_duplicates=bool(data.get("merge_duplicates", False)),
            parallel_queries=bool(data.get("parallel_queries", True)),
            query_timeout=float(data.get("query_timeout", 30.0)),
        )


__all__ = [
    "CalendarBackendType",
    "CalendarConfig",
    "CalendarInfo",
    "CalendarsGlobalConfig",
    "SyncMode",
]
