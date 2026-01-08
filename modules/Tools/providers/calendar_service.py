"""Calendar service provider that uses CalendarEventService.

This provider replaces debian12_local for calendar operations, using the
CalendarEventService from core.services.calendar instead of the legacy
debian12_calendar implementation.

Operations supported:
    - list: List upcoming events
    - detail: Get event by ID
    - search: Full-text search for events
    - create: Create a new event
    - update: Update an existing event
    - delete: Delete an event
    - list_calendars: List calendar categories
    - upcoming: Get upcoming events (agent convenience method)
    - availability: Check availability for a time range
    - find_free_time: Find available time slots
    - summary: Get calendar summary for a period
    - suggest_times: Get meeting time suggestions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from core.services.calendar import (
    CalendarEventService,
    CalendarEventCreate,
    CalendarEventUpdate,
    EventStatus,
    EventVisibility,
)
from core.services.common import Actor
from modules.calendar_store.dataclasses import BusyStatus

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry

if TYPE_CHECKING:
    from core.services.calendar import CalendarEvent


class CalendarServiceProvider(ToolProvider):
    """Tool provider that wraps CalendarEventService for agent use.
    
    This provider offers the same operations as the legacy debian12_calendar
    tool but uses the new service layer with proper permission checking,
    validation, and domain events.
    
    Additionally, it provides agent-focused operations like:
    - upcoming: Quick access to upcoming events
    - availability: Check if a time slot is free
    - find_free_time: Find available meeting slots
    - summary: Get a calendar overview
    - suggest_times: Get meeting time suggestions
    """
    
    # Operations that modify calendar data
    WRITE_OPERATIONS = frozenset({"create", "update", "delete"})
    
    def __init__(
        self,
        spec: ToolProviderSpec | None = None,
        *,
        tool_name: str = "calendar",
        fallback_callable: Any | None = None,
        service: CalendarEventService | None = None,
        default_actor: Actor | None = None,
    ) -> None:
        """Initialize the provider.
        
        Parameters
        ----------
        spec
            Tool provider specification (optional for standalone use)
        tool_name
            Name of the tool
        fallback_callable
            Fallback callable if provider fails
        service
            CalendarEventService instance. If not provided, will be
            lazily initialized from the service registry.
        default_actor
            Default actor to use when context doesn't provide one.
            Required for standalone usage.
        """
        # Create a default spec if not provided
        if spec is None:
            spec = ToolProviderSpec(name="calendar_service")
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        self._service = service
        self._default_actor = default_actor
    
    @property
    def service(self) -> CalendarEventService:
        """Get or create the calendar service."""
        if self._service is None:
            # Lazy initialization - caller should inject service
            raise RuntimeError(
                "CalendarEventService not configured. "
                "Inject service via constructor or set _service attribute."
            )
        return self._service
    
    async def call(self, **kwargs: Any) -> Any:
        """Execute a calendar operation.
        
        Parameters
        ----------
        operation : str
            The operation to perform: list, detail, search, create,
            update, delete, list_calendars, upcoming, availability,
            find_free_time, summary, suggest_times
        **kwargs
            Operation-specific parameters
            
        Returns
        -------
        Dict or List
            Operation result
        """
        operation = kwargs.pop("operation", "").strip().lower()
        context = kwargs.pop("context", None)
        
        # Get actor from context or use default
        actor = self._get_actor(context)
        
        # Check write permission from persona
        if operation in self.WRITE_OPERATIONS:
            if not self._persona_allows_write_operations(context):
                return {
                    "error": "Calendar write operations disabled",
                    "message": "Persona must enable 'personal_assistant.calendar_write_enabled'",
                }
        
        # Dispatch to operation handler
        handler = self._get_handler(operation)
        if handler is None:
            return {
                "error": f"Unknown operation: {operation}",
                "supported_operations": [
                    "list", "detail", "search", "create", "update", "delete",
                    "list_calendars", "upcoming", "availability", "find_free_time",
                    "summary", "suggest_times",
                ],
            }
        
        return await handler(actor, **kwargs)
    
    def _get_handler(self, operation: str):
        """Get the handler method for an operation."""
        handlers = {
            "list": self._handle_list,
            "detail": self._handle_detail,
            "search": self._handle_search,
            "create": self._handle_create,
            "update": self._handle_update,
            "delete": self._handle_delete,
            "list_calendars": self._handle_list_calendars,
            # Agent convenience operations
            "upcoming": self._handle_upcoming,
            "availability": self._handle_availability,
            "find_free_time": self._handle_find_free_time,
            "summary": self._handle_summary,
            "suggest_times": self._handle_suggest_times,
        }
        return handlers.get(operation)
    
    def _get_actor(self, context: Optional[Dict[str, Any]]) -> Actor:
        """Extract or create actor from context."""
        if context and isinstance(context, dict):
            user_id = context.get("user_id") or context.get("actor_id")
            tenant_id = context.get("tenant_id")
            permissions = context.get("permissions", ["calendar:read"])
            
            if user_id:
                return Actor(
                    type="user",
                    id=user_id,
                    tenant_id=tenant_id or "default",
                    permissions=permissions,
                )
        
        if self._default_actor:
            return self._default_actor
        
        # Fallback actor for anonymous/system use
        return Actor(
            type="system",
            id="calendar-tool",
            tenant_id="default",
            permissions=["calendar:read", "calendar:write"],
        )
    
    def _persona_allows_write_operations(
        self, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if persona allows calendar write operations."""
        if not context or not isinstance(context, dict):
            return True  # Allow if no context (standalone usage)
        
        persona_flags = context.get("persona_flags", {})
        if isinstance(persona_flags, dict):
            return bool(
                persona_flags.get("calendar_write_enabled", True)
                or persona_flags.get("personal_assistant", {}).get(
                    "calendar_write_enabled", True
                )
            )
        return True
    
    # ========================================================================
    # Standard Operations (compatible with debian12_calendar)
    # ========================================================================
    
    async def _handle_list(
        self,
        actor: Actor,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        calendar: Optional[str] = None,
        limit: Optional[int] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """List calendar events."""
        start_date = self._parse_datetime(start)
        end_date = self._parse_datetime(end)
        
        result = await self.service.list_events(
            actor,
            start_date=start_date,
            end_date=end_date,
            limit=limit or 100,
        )
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        events = result.value or []
        return {
            "events": [self._serialize_event(e) for e in events],
            "count": len(events),
        }
    
    async def _handle_detail(
        self,
        actor: Actor,
        *,
        event_id: str = "",
        calendar: Optional[str] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Get event details by ID."""
        if not event_id:
            return {"error": "event_id is required"}
        
        result = await self.service.get_event(actor, event_id)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return self._serialize_event(result.value)
    
    async def _handle_search(
        self,
        actor: Actor,
        *,
        query: str = "",
        start: Optional[str] = None,
        end: Optional[str] = None,
        calendar: Optional[str] = None,
        limit: Optional[int] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Search for events by text."""
        if not query:
            return {"error": "query is required", "events": []}
        
        start_date = self._parse_datetime(start)
        end_date = self._parse_datetime(end)
        
        result = await self.service.search_events(
            actor,
            query=query,
            start_date=start_date,
            end_date=end_date,
            limit=limit or 50,
        )
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code, "events": []}
        
        events = result.value or []
        return {
            "events": [self._serialize_event(e) for e in events],
            "count": len(events),
            "query": query,
        }
    
    async def _handle_create(
        self,
        actor: Actor,
        *,
        title: str = "",
        start: Optional[str] = None,
        end: Optional[str] = None,
        all_day: bool = False,
        calendar: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[Dict[str, Any]]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Create a new calendar event."""
        if not title:
            return {"error": "title is required"}
        
        start_time = self._parse_datetime(start)
        end_time = self._parse_datetime(end)
        
        if not start_time:
            start_time = datetime.now(timezone.utc)
        if not end_time:
            end_time = start_time + timedelta(hours=1)
        
        event_data = CalendarEventCreate(
            title=title,
            description=description or "",
            start_time=start_time,
            end_time=end_time,
            timezone_name="UTC",
            location=location,
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            all_day=all_day,
            category_id=self._parse_uuid(calendar),
            metadata={
                "attendees": attendees or [],
                "source": "calendar_service_provider",
            },
        )
        
        result = await self.service.create_event(actor, event_data)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return {
            "status": "created",
            "event": self._serialize_event(result.value),
        }
    
    async def _handle_update(
        self,
        actor: Actor,
        *,
        event_id: str = "",
        title: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        all_day: Optional[bool] = None,
        calendar: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[Dict[str, Any]]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Update an existing calendar event."""
        if not event_id:
            return {"error": "event_id is required"}
        
        update_data = CalendarEventUpdate(
            title=title,
            description=description,
            start_time=self._parse_datetime(start),
            end_time=self._parse_datetime(end),
            location=location,
            all_day=all_day,
            category_id=self._parse_uuid(calendar),
            metadata={"attendees": attendees} if attendees else None,
        )
        
        result = await self.service.update_event(actor, event_id, update_data)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return {
            "status": "updated",
            "event": self._serialize_event(result.value),
        }
    
    async def _handle_delete(
        self,
        actor: Actor,
        *,
        event_id: str = "",
        calendar: Optional[str] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Delete a calendar event."""
        if not event_id:
            return {"error": "event_id is required"}
        
        result = await self.service.delete_event(actor, event_id)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return {"status": "deleted", "event_id": event_id}
    
    async def _handle_list_calendars(
        self,
        actor: Actor,
        **_kwargs,
    ) -> Dict[str, Any]:
        """List available calendars (categories)."""
        # This would need a category listing method on the service
        # For now, return a placeholder
        return {
            "calendars": [
                {"id": "default", "name": "Default Calendar", "color": "#4285F4"},
            ],
            "message": "Category listing not yet implemented in service layer",
        }
    
    # ========================================================================
    # Agent Convenience Operations
    # ========================================================================
    
    async def _handle_upcoming(
        self,
        actor: Actor,
        *,
        hours: int = 24,
        limit: int = 10,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Get upcoming events (agent convenience method)."""
        result = await self.service.get_upcoming_events(
            actor,
            hours_ahead=hours,
            limit=limit,
        )
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        events = result.value or []
        return {
            "events": [self._serialize_event(e) for e in events],
            "count": len(events),
            "hours_ahead": hours,
        }
    
    async def _handle_availability(
        self,
        actor: Actor,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Check if a time range is available."""
        start_time = self._parse_datetime(start)
        end_time = self._parse_datetime(end)
        
        if not start_time or not end_time:
            return {"error": "start and end are required"}
        
        result = await self.service.check_availability(actor, start_time, end_time)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return result.value or {}
    
    async def _handle_find_free_time(
        self,
        actor: Actor,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        duration_minutes: int = 60,
        working_hours_start: int = 9,
        working_hours_end: int = 17,
        exclude_weekends: bool = True,
        max_slots: int = 5,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Find available time slots."""
        start_range = self._parse_datetime(start) or datetime.now(timezone.utc)
        end_range = self._parse_datetime(end) or (start_range + timedelta(days=7))
        
        result = await self.service.find_free_time(
            actor,
            start_range=start_range,
            end_range=end_range,
            duration_minutes=duration_minutes,
            working_hours=(working_hours_start, working_hours_end),
            exclude_weekends=exclude_weekends,
            max_slots=max_slots,
        )
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        # Serialize datetime objects in slots
        slots = []
        for slot in (result.value or []):
            slots.append({
                "start": slot["start"].isoformat() if isinstance(slot["start"], datetime) else slot["start"],
                "end": slot["end"].isoformat() if isinstance(slot["end"], datetime) else slot["end"],
                "duration_minutes": slot["duration_minutes"],
            })
        
        return {
            "slots": slots,
            "count": len(slots),
            "search_range": {
                "start": start_range.isoformat(),
                "end": end_range.isoformat(),
            },
        }
    
    async def _handle_summary(
        self,
        actor: Actor,
        *,
        period: str = "today",
        **_kwargs,
    ) -> Dict[str, Any]:
        """Get calendar summary for a period."""
        result = await self.service.get_calendar_summary(actor, period=period)
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        return result.value or {"period": period, "events": [], "total_events": 0}
    
    async def _handle_suggest_times(
        self,
        actor: Actor,
        *,
        duration_minutes: int = 60,
        start: Optional[str] = None,
        end: Optional[str] = None,
        working_hours_start: int = 9,
        working_hours_end: int = 17,
        exclude_weekends: bool = True,
        max_suggestions: int = 3,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Suggest optimal meeting times."""
        preferred_start = self._parse_datetime(start)
        preferred_end = self._parse_datetime(end)
        
        result = await self.service.suggest_meeting_times(
            actor,
            duration_minutes=duration_minutes,
            preferred_start=preferred_start,
            preferred_end=preferred_end,
            working_hours=(working_hours_start, working_hours_end),
            exclude_weekends=exclude_weekends,
            max_suggestions=max_suggestions,
        )
        
        if result.is_failure:
            return {"error": result.error, "code": result.error_code}
        
        # Serialize datetime objects
        suggestions = []
        for slot in (result.value or []):
            suggestions.append({
                "start": slot["start"].isoformat() if isinstance(slot["start"], datetime) else slot["start"],
                "end": slot["end"].isoformat() if isinstance(slot["end"], datetime) else slot["end"],
                "duration_minutes": slot["duration_minutes"],
            })
        
        return {
            "suggestions": suggestions,
            "count": len(suggestions),
            "duration_minutes": duration_minutes,
        }
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    def _parse_datetime(
        self, value: Optional[str]
    ) -> Optional[datetime]:
        """Parse datetime from string."""
        if not value:
            return None
        
        if isinstance(value, datetime):
            return value
        
        try:
            # Try ISO format first
            if "T" in value:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            else:
                # Try date only
                dt = datetime.strptime(value, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Ensure timezone aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt
        except (ValueError, TypeError):
            return None
    
    def _parse_uuid(self, value: Optional[str]) -> Optional[UUID]:
        """Parse UUID from string."""
        if not value:
            return None
        
        if isinstance(value, UUID):
            return value
        
        try:
            return UUID(value)
        except (ValueError, TypeError):
            return None
    
    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """Serialize a CalendarEvent to a dict for JSON response."""
        return {
            "id": str(event.event_id) if hasattr(event, "event_id") else str(getattr(event, "id", "")),
            "title": event.title,
            "start": event.start_time.isoformat() if event.start_time else None,
            "end": event.end_time.isoformat() if event.end_time else None,
            "all_day": getattr(event, "all_day", False),
            "location": getattr(event, "location", None),
            "description": getattr(event, "description", None),
            "calendar": str(event.category_id) if getattr(event, "category_id", None) else None,
            "status": getattr(event, "status", None),
            "visibility": getattr(event, "visibility", None),
        }


# Register the provider
tool_provider_registry.register(
    "calendar_service",
    lambda spec, tool_name, fallback: CalendarServiceProvider(spec, tool_name=tool_name, fallback_callable=fallback)
)

__all__ = ["CalendarServiceProvider"]
