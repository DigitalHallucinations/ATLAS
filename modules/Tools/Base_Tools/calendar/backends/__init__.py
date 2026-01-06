"""Calendar backend implementations.

This package provides concrete calendar backend implementations for various
calendar providers including local ICS files, D-Bus (Debian 12), Google,
Outlook, CalDAV, and Apple Calendar.

Each backend inherits from the abstract CalendarBackend class and implements
the standard interface for listing, searching, creating, updating, and
deleting calendar events.

Example:
    from modules.Tools.Base_Tools.calendar.backends import (
        ICSCalendarBackend,
        NullCalendarBackend,
    )

    # Create ICS backend
    backend = ICSCalendarBackend.from_config(calendar_config)

    # Use null backend for testing
    null_backend = NullCalendarBackend()
"""

from .base import (
    CalendarBackend,
    CalendarBackendError,
    CalendarEvent,
    EventNotFoundError,
    AuthenticationError,
    RateLimitError,
    ConnectionError,
    NullCalendarBackend,
)
from .ics import ICSCalendarBackend
from .dbus import DBusCalendarBackend
from .google import GoogleCalendarBackend
from .outlook import OutlookCalendarBackend
from .caldav import CalDAVCalendarBackend

__all__ = [
    # Base classes and types
    "CalendarBackend",
    "CalendarBackendError",
    "CalendarEvent",
    "EventNotFoundError",
    "AuthenticationError",
    "RateLimitError",
    "ConnectionError",
    "NullCalendarBackend",
    # Backend implementations
    "ICSCalendarBackend",
    "DBusCalendarBackend",
    "GoogleCalendarBackend",
    "OutlookCalendarBackend",
    "CalDAVCalendarBackend",
]
