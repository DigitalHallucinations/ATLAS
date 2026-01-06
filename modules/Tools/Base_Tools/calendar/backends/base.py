"""Base calendar backend classes and shared utilities.

This module provides the abstract CalendarBackend interface and shared
helper classes used by all backend implementations.
"""

from __future__ import annotations

import datetime as _dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class CalendarBackendError(RuntimeError):
    """Base exception for calendar backend failures."""


class EventNotFoundError(CalendarBackendError):
    """Raised when an event identifier cannot be resolved."""


class AuthenticationError(CalendarBackendError):
    """Raised when authentication fails."""


class RateLimitError(CalendarBackendError):
    """Raised when API rate limits are exceeded."""


class ConnectionError(CalendarBackendError):
    """Raised when connection to the calendar service fails."""


# -----------------------------------------------------------------------------
# Data Model
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CalendarEvent:
    """Normalized calendar event representation.

    This is the common event model used across all backends. Each backend
    is responsible for converting its native event format to this structure.

    Attributes:
        id: Unique identifier within the backend.
        title: Event title/summary.
        start: Start datetime (timezone-aware).
        end: End datetime (timezone-aware).
        all_day: Whether this is an all-day event.
        location: Optional location string.
        description: Optional event description.
        calendar: Name of the calendar this event belongs to.
        attendees: List of attendee dictionaries with email, name, role, status.
        raw: Original backend-specific data for debugging.
    """

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

    @property
    def global_id(self) -> str:
        """Return a globally unique ID combining calendar and event ID."""
        if self.calendar:
            return f"{self.calendar}:{self.id}"
        return self.id

    @property
    def duration(self) -> _dt.timedelta:
        """Return the event duration."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event into a JSON-friendly dict."""
        return {
            "id": self.id,
            "global_id": self.global_id,
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


# -----------------------------------------------------------------------------
# Abstract Backend Interface
# -----------------------------------------------------------------------------


class CalendarBackend(ABC):
    """Abstract base class for calendar backends.

    All calendar backends (ICS, Google, Outlook, CalDAV, etc.) must implement
    this interface. The interface supports both read and write operations.

    Backends should:
    - Handle authentication internally
    - Convert native event formats to CalendarEvent
    - Raise appropriate exceptions for errors
    - Be async-compatible for I/O operations
    """

    @property
    def name(self) -> str:
        """Return the backend name for logging/identification."""
        return self.__class__.__name__

    @property
    def supports_write(self) -> bool:
        """Return True if this backend supports write operations."""
        return True

    async def initialize(self) -> None:
        """Initialize the backend (authenticate, connect, etc.).

        Called once when the backend is first registered. Override to
        perform any necessary setup.
        """
        pass

    async def shutdown(self) -> None:
        """Shutdown the backend and release resources.

        Called when the application is shutting down. Override to
        perform cleanup.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the backend is healthy and connected.

        Returns:
            True if the backend is operational.
        """
        return True

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        """List events within a time range.

        Args:
            start: Start of the time range (inclusive).
            end: End of the time range (exclusive).
            calendar: Optional calendar filter (backend-specific).

        Returns:
            List of events within the range, sorted by start time.
        """
        ...

    @abstractmethod
    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Get a single event by ID.

        Args:
            event_id: The event identifier.
            calendar: Optional calendar hint (backend-specific).

        Returns:
            The requested CalendarEvent.

        Raises:
            EventNotFoundError: If the event doesn't exist.
        """
        ...

    @abstractmethod
    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        """Search events by query string.

        Args:
            query: Search query (matched against title, description, location).
            start: Start of the time range.
            end: End of the time range.
            calendar: Optional calendar filter.

        Returns:
            List of matching events.
        """
        ...

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Create a new event.

        Args:
            payload: Event data including title, start, end, etc.
            calendar: Target calendar (backend-specific).

        Returns:
            The created CalendarEvent with assigned ID.

        Raises:
            CalendarBackendError: If creation fails.
        """
        ...

    @abstractmethod
    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        """Update an existing event.

        Args:
            event_id: The event identifier.
            payload: Updated event data.
            calendar: Calendar containing the event.

        Returns:
            The updated CalendarEvent.

        Raises:
            EventNotFoundError: If the event doesn't exist.
            CalendarBackendError: If update fails.
        """
        ...

    @abstractmethod
    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        """Delete an event.

        Args:
            event_id: The event identifier.
            calendar: Calendar containing the event.

        Raises:
            EventNotFoundError: If the event doesn't exist.
            CalendarBackendError: If deletion fails.
        """
        ...


# -----------------------------------------------------------------------------
# Null Backend (No-op fallback)
# -----------------------------------------------------------------------------


class NullCalendarBackend(CalendarBackend):
    """Backend used when calendar access has not been configured.

    Returns empty results for read operations and raises errors for writes.
    """

    @property
    def supports_write(self) -> bool:
        return False

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        return []

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
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
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
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


__all__ = [
    # Exceptions
    "AuthenticationError",
    "CalendarBackendError",
    "ConnectionError",
    "EventNotFoundError",
    "RateLimitError",
    # Data model
    "CalendarEvent",
    # Backends
    "CalendarBackend",
    "NullCalendarBackend",
]
