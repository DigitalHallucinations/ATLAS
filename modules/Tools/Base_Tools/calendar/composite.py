"""Composite calendar backend.

This module provides the CompositeCalendarBackend which aggregates multiple
calendar backends into a single unified interface, supporting parallel queries
and intelligent write routing.
"""

from __future__ import annotations

import asyncio
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

from .config import CalendarConfig, SyncMode
from .registry import (
    CalendarNotFoundError,
    CalendarProviderRegistry,
    CalendarReadOnlyError,
)

if TYPE_CHECKING:
    from ..debian12_calendar import CalendarBackend, CalendarEvent

logger = logging.getLogger(__name__)


class CompositeCalendarError(RuntimeError):
    """Base exception for composite calendar operations."""


class NoWritableCalendarError(CompositeCalendarError):
    """Raised when a write operation has no target calendar."""


class CompositeCalendarBackend:
    """A calendar backend that aggregates multiple backends.

    This class implements the same interface as CalendarBackend but routes
    operations to multiple underlying backends via a CalendarProviderRegistry.

    For read operations (list_events, search_events):
        - Queries all backends in parallel (if enabled)
        - Merges and sorts results by start time
        - Annotates events with their source calendar

    For write operations (create_event, update_event, delete_event):
        - Routes to the specified calendar, or the default calendar
        - Validates that the target supports writes

    Example:
        registry = CalendarProviderRegistry()
        # ... register backends ...

        composite = CompositeCalendarBackend(registry)
        events = await composite.list_events(start, end)  # From all calendars
        await composite.create_event(payload, calendar="work")  # Routes to "work"
    """

    def __init__(
        self,
        registry: CalendarProviderRegistry,
        *,
        default_calendar: Optional[str] = None,
        parallel_queries: bool = True,
        query_timeout: float = 30.0,
    ) -> None:
        """Initialize the composite backend.

        Args:
            registry: The calendar provider registry.
            default_calendar: Override for default calendar name.
            parallel_queries: Whether to query backends in parallel.
            query_timeout: Timeout in seconds for backend queries.
        """
        self._registry = registry
        self._default_calendar_override = default_calendar
        self._parallel_queries = parallel_queries
        self._query_timeout = query_timeout

    @property
    def registry(self) -> CalendarProviderRegistry:
        """Return the underlying registry."""
        return self._registry

    @property
    def default_calendar(self) -> Optional[str]:
        """Return the default calendar name for write operations."""
        if self._default_calendar_override:
            return self._default_calendar_override
        return self._registry.default_calendar_name

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence["CalendarEvent"]:
        """List events from one or all calendars.

        Args:
            start: Start of the time range.
            end: End of the time range.
            calendar: Optional specific calendar name. If None, queries all.

        Returns:
            List of events sorted by start time.
        """
        if calendar:
            # Query single calendar
            backend = self._registry.get_backend(calendar)
            events = await self._query_with_timeout(
                backend.list_events(start, end, calendar=calendar),
                calendar,
            )
            return self._ensure_calendar_annotation(events, calendar)

        # Query all calendars
        backends = [
            (name, self._registry.get_backend(name))
            for name in self._registry.calendar_names
        ]

        if not backends:
            return []

        all_events: List["CalendarEvent"] = []

        if self._parallel_queries:
            tasks = [
                self._query_with_timeout(
                    backend.list_events(start, end),
                    name,
                )
                for name, backend in backends
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (name, _), result in zip(backends, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Error querying calendar '%s': %s",
                        name,
                        result,
                    )
                    self._registry.update_sync_status(
                        name,
                        is_connected=False,
                        error_message=str(result),
                    )
                else:
                    # Type narrowed: result is Sequence[CalendarEvent]
                    events: Sequence["CalendarEvent"] = result  # type: ignore[assignment]
                    all_events.extend(self._ensure_calendar_annotation(events, name))
                    self._registry.update_sync_status(
                        name,
                        is_connected=True,
                        last_sync=_dt.datetime.now(_dt.timezone.utc),
                    )
        else:
            for name, backend in backends:
                try:
                    events = await self._query_with_timeout(
                        backend.list_events(start, end),
                        name,
                    )
                    all_events.extend(self._ensure_calendar_annotation(events, name))
                    self._registry.update_sync_status(
                        name,
                        is_connected=True,
                        last_sync=_dt.datetime.now(_dt.timezone.utc),
                    )
                except Exception as exc:
                    logger.warning("Error querying calendar '%s': %s", name, exc)
                    self._registry.update_sync_status(
                        name,
                        is_connected=False,
                        error_message=str(exc),
                    )

        # Sort by start time
        all_events.sort(key=lambda e: e.start)
        return all_events

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> "CalendarEvent":
        """Get a single event by ID.

        Args:
            event_id: The event identifier.
            calendar: Optional calendar name. If None, searches all.

        Returns:
            The found CalendarEvent.

        Raises:
            EventNotFoundError: If the event is not found.
        """
        # Import here to avoid circular import
        from ..debian12_calendar import EventNotFoundError

        if calendar:
            backend = self._registry.get_backend(calendar)
            event = await backend.get_event(event_id, calendar=calendar)
            return self._annotate_event(event, calendar)

        # Search all calendars
        errors: List[Exception] = []

        for name in self._registry.calendar_names:
            try:
                backend = self._registry.get_backend(name)
                event = await backend.get_event(event_id, calendar=name)
                return self._annotate_event(event, name)
            except EventNotFoundError:
                continue
            except Exception as exc:
                errors.append(exc)
                logger.debug("Error searching calendar '%s' for event: %s", name, exc)

        raise EventNotFoundError(f"Event '{event_id}' not found in any calendar")

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence["CalendarEvent"]:
        """Search events by query string.

        Args:
            query: Search query.
            start: Start of time range.
            end: End of time range.
            calendar: Optional specific calendar. If None, searches all.

        Returns:
            List of matching events sorted by start time.
        """
        if calendar:
            backend = self._registry.get_backend(calendar)
            events = await self._query_with_timeout(
                backend.search_events(query, start, end, calendar=calendar),
                calendar,
            )
            return self._ensure_calendar_annotation(events, calendar)

        # Search all calendars
        backends = [
            (name, self._registry.get_backend(name))
            for name in self._registry.calendar_names
        ]

        if not backends:
            return []

        all_events: List["CalendarEvent"] = []

        if self._parallel_queries:
            tasks = [
                self._query_with_timeout(
                    backend.search_events(query, start, end),
                    name,
                )
                for name, backend in backends
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (name, _), result in zip(backends, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Error searching calendar '%s': %s",
                        name,
                        result,
                    )
                else:
                    # Type narrowed: result is Sequence[CalendarEvent]
                    events: Sequence["CalendarEvent"] = result  # type: ignore[assignment]
                    all_events.extend(self._ensure_calendar_annotation(events, name))
        else:
            for name, backend in backends:
                try:
                    events = await self._query_with_timeout(
                        backend.search_events(query, start, end),
                        name,
                    )
                    all_events.extend(self._ensure_calendar_annotation(events, name))
                except Exception as exc:
                    logger.warning("Error searching calendar '%s': %s", name, exc)

        all_events.sort(key=lambda e: e.start)
        return all_events

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> "CalendarEvent":
        """Create a new event.

        Args:
            payload: Event data.
            calendar: Target calendar name. Uses default if not specified.

        Returns:
            The created CalendarEvent.

        Raises:
            NoWritableCalendarError: If no writable calendar is available.
            CalendarReadOnlyError: If the target calendar is read-only.
        """
        target = self._resolve_write_target(calendar)
        backend = self._registry.get_writable_backend(target)
        event = await backend.create_event(payload, calendar=target)
        return self._annotate_event(event, target)

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> "CalendarEvent":
        """Update an existing event.

        Args:
            event_id: The event identifier.
            payload: Updated event data.
            calendar: Target calendar name. Required if event could be in multiple.

        Returns:
            The updated CalendarEvent.

        Raises:
            CalendarNotFoundError: If calendar not specified and can't be inferred.
            CalendarReadOnlyError: If the target calendar is read-only.
        """
        if calendar:
            target = calendar
        else:
            # Try to find which calendar has this event
            target = await self._find_event_calendar(event_id)
            if target is None:
                raise CalendarNotFoundError(
                    f"Cannot determine calendar for event '{event_id}'. "
                    "Please specify the calendar parameter."
                )

        backend = self._registry.get_writable_backend(target)
        event = await backend.update_event(event_id, payload, calendar=target)
        return self._annotate_event(event, target)

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        """Delete an event.

        Args:
            event_id: The event identifier.
            calendar: Target calendar name. Required if event could be in multiple.

        Raises:
            CalendarNotFoundError: If calendar not specified and can't be inferred.
            CalendarReadOnlyError: If the target calendar is read-only.
        """
        if calendar:
            target = calendar
        else:
            target = await self._find_event_calendar(event_id)
            if target is None:
                raise CalendarNotFoundError(
                    f"Cannot determine calendar for event '{event_id}'. "
                    "Please specify the calendar parameter."
                )

        backend = self._registry.get_writable_backend(target)
        await backend.delete_event(event_id, calendar=target)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_write_target(self, calendar: Optional[str]) -> str:
        """Resolve the target calendar for a write operation."""
        if calendar:
            # Validate it exists
            if calendar not in self._registry.calendar_names:
                raise CalendarNotFoundError(f"Calendar '{calendar}' is not registered")
            return calendar

        default = self.default_calendar
        if default:
            return default

        # Find first writable calendar
        writable = self._registry.list_writable_calendars()
        if writable:
            return writable[0].name

        raise NoWritableCalendarError(
            "No calendar specified and no writable calendars available"
        )

    async def _find_event_calendar(self, event_id: str) -> Optional[str]:
        """Try to find which calendar contains an event."""
        from ..debian12_calendar import EventNotFoundError

        for name in self._registry.calendar_names:
            try:
                backend = self._registry.get_backend(name)
                await backend.get_event(event_id, calendar=name)
                return name
            except EventNotFoundError:
                continue
            except Exception:
                continue
        return None

    async def _query_with_timeout(
        self,
        coro: Any,
        calendar_name: str,
    ) -> Sequence["CalendarEvent"]:
        """Execute a query coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=self._query_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timeout querying calendar '%s' (%.1fs)",
                calendar_name,
                self._query_timeout,
            )
            raise

    def _ensure_calendar_annotation(
        self,
        events: Sequence["CalendarEvent"],
        calendar_name: str,
    ) -> List["CalendarEvent"]:
        """Ensure all events have their calendar field set."""
        result: List["CalendarEvent"] = []
        for event in events:
            result.append(self._annotate_event(event, calendar_name))
        return result

    def _annotate_event(
        self,
        event: "CalendarEvent",
        calendar_name: str,
    ) -> "CalendarEvent":
        """Annotate an event with calendar metadata."""
        # Only set calendar if not already set
        if event.calendar is None:
            # Create a new event with the calendar field set
            # We need to import the class to create a new instance
            from ..debian12_calendar import CalendarEvent as CalendarEventClass

            return CalendarEventClass(
                id=event.id,
                title=event.title,
                start=event.start,
                end=event.end,
                all_day=event.all_day,
                location=event.location,
                description=event.description,
                calendar=calendar_name,
                attendees=list(event.attendees),
                raw=dict(event.raw),
            )
        return event


__all__ = [
    "CompositeCalendarBackend",
    "CompositeCalendarError",
    "NoWritableCalendarError",
]
