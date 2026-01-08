"""Calendar context utilities for agent/LLM context injection.

This module provides utilities for generating calendar context that can be
injected into LLM prompts to give agents awareness of the user's schedule.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.services.common import Actor


async def get_calendar_context_for_prompt(
    service,
    actor: Actor,
    *,
    include_today: bool = True,
    include_upcoming: bool = True,
    include_next_event: bool = True,
    max_events: int = 5,
) -> str:
    """Generate calendar context for LLM prompt injection.
    
    Creates a formatted string summarizing the user's calendar state
    that can be appended to the system prompt.
    
    Parameters
    ----------
    service
        CalendarEventService instance
    actor
        Actor to fetch calendar for
    include_today
        Include today's schedule summary
    include_upcoming
        Include upcoming events list
    include_next_event
        Include details of next upcoming event
    max_events
        Maximum events to include in lists
        
    Returns
    -------
    str
        Formatted calendar context for LLM prompt
    """
    parts = []
    
    if include_today:
        today_summary = await _get_today_summary(service, actor)
        if today_summary:
            parts.append(today_summary)
    
    if include_upcoming:
        upcoming = await _get_upcoming_list(service, actor, max_events)
        if upcoming:
            parts.append(upcoming)
    
    if include_next_event:
        next_event = await _get_next_event_detail(service, actor)
        if next_event:
            parts.append(next_event)
    
    if not parts:
        return ""
    
    return "## Current Calendar State\n" + "\n\n".join(parts)


async def _get_today_summary(
    service,
    actor: Actor,
) -> Optional[str]:
    """Get today's calendar summary."""
    result = await service.get_calendar_summary(actor, period="today")
    
    if result.is_failure:
        return None
    
    summary = result.value
    total = summary.get("total_events", 0)
    busy_hours = summary.get("busy_hours", 0)
    
    if total == 0:
        return "**Today**: No scheduled events"
    
    return f"**Today**: {total} event(s), approximately {busy_hours:.1f} hours of commitments"


async def _get_upcoming_list(
    service,
    actor: Actor,
    max_events: int,
) -> Optional[str]:
    """Get list of upcoming events."""
    result = await service.get_upcoming_events(
        actor,
        hours_ahead=24,
        limit=max_events,
    )
    
    if result.is_failure or not result.value:
        return None
    
    events = result.value
    lines = ["**Upcoming in next 24 hours:**"]
    
    for event in events:
        start_time = event.start_time
        if start_time:
            time_str = start_time.strftime("%H:%M")
            lines.append(f"- {time_str}: {event.title}")
        else:
            lines.append(f"- {event.title}")
    
    return "\n".join(lines)


async def _get_next_event_detail(
    service,
    actor: Actor,
) -> Optional[str]:
    """Get detailed info about the next upcoming event."""
    result = await service.get_upcoming_events(
        actor,
        hours_ahead=48,
        limit=1,
    )
    
    if result.is_failure or not result.value:
        return None
    
    event = result.value[0]
    now = datetime.now(timezone.utc)
    
    lines = [f"**Next Event**: {event.title}"]
    
    if event.start_time:
        time_until = event.start_time - now
        minutes = int(time_until.total_seconds() / 60)
        
        if minutes < 60:
            lines.append(f"  - Starts in: {minutes} minutes")
        elif minutes < 1440:  # Less than 24 hours
            hours = minutes // 60
            lines.append(f"  - Starts in: {hours} hour(s)")
        else:
            days = minutes // 1440
            lines.append(f"  - Starts in: {days} day(s)")
        
        lines.append(f"  - Time: {event.start_time.strftime('%Y-%m-%d %H:%M')}")
    
    if event.location:
        lines.append(f"  - Location: {event.location}")
    
    if event.description:
        # Truncate long descriptions
        desc = event.description[:100]
        if len(event.description) > 100:
            desc += "..."
        lines.append(f"  - Description: {desc}")
    
    return "\n".join(lines)


def format_availability_response(
    available: bool,
    conflicts: list,
    start_time: datetime,
    end_time: datetime,
) -> str:
    """Format an availability check response for natural language.
    
    Parameters
    ----------
    available
        Whether the time slot is available
    conflicts
        List of conflicting events
    start_time
        Start of the time range
    end_time
        End of the time range
        
    Returns
    -------
    str
        Natural language description of availability
    """
    time_range = f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')}"
    
    if available:
        return f"The time slot {time_range} is available."
    
    conflict_count = len(conflicts)
    if conflict_count == 1:
        conflict = conflicts[0]
        title = getattr(conflict, "title", "an event")
        return f"The time slot {time_range} conflicts with {title}."
    
    return f"The time slot {time_range} conflicts with {conflict_count} existing events."


def format_free_time_slots(slots: list) -> str:
    """Format free time slots for natural language response.
    
    Parameters
    ----------
    slots
        List of slot dicts with start, end, duration_minutes
        
    Returns
    -------
    str
        Natural language description of available slots
    """
    if not slots:
        return "No available time slots found in the requested range."
    
    lines = [f"Found {len(slots)} available time slot(s):"]
    
    for i, slot in enumerate(slots, 1):
        start = slot.get("start")
        end = slot.get("end")
        duration = slot.get("duration_minutes", 0)
        
        if isinstance(start, datetime):
            start_str = start.strftime("%Y-%m-%d %H:%M")
        else:
            start_str = str(start)
        
        if isinstance(end, datetime):
            end_str = end.strftime("%H:%M")
        else:
            end_str = str(end)
        
        lines.append(f"  {i}. {start_str} - {end_str} ({duration} min)")
    
    return "\n".join(lines)


class CalendarContextInjector:
    """Helper class for injecting calendar context into LLM prompts.
    
    Can be registered with LLMContextManager or used directly.
    
    Example::
    
        injector = CalendarContextInjector(calendar_service)
        context = await injector.get_context(actor)
        
        # Use with LLMContextManager
        llm_context = await context_manager.build_context(
            conversation_id="conv-123",
            messages=history,
            additional_system_context=context,
        )
    """
    
    def __init__(
        self,
        service,
        *,
        include_today: bool = True,
        include_upcoming: bool = True,
        include_next_event: bool = True,
        max_events: int = 5,
    ) -> None:
        """Initialize the context injector.
        
        Parameters
        ----------
        service
            CalendarEventService instance
        include_today
            Include today's schedule summary
        include_upcoming
            Include upcoming events list
        include_next_event
            Include details of next upcoming event
        max_events
            Maximum events to include
        """
        self._service = service
        self._include_today = include_today
        self._include_upcoming = include_upcoming
        self._include_next_event = include_next_event
        self._max_events = max_events
    
    async def get_context(self, actor: Actor) -> str:
        """Get formatted calendar context for injection."""
        return await get_calendar_context_for_prompt(
            self._service,
            actor,
            include_today=self._include_today,
            include_upcoming=self._include_upcoming,
            include_next_event=self._include_next_event,
            max_events=self._max_events,
        )
    
    async def get_context_dict(self, actor: Actor) -> Dict[str, Any]:
        """Get calendar context as a structured dict.
        
        Useful for programmatic access or JSON serialization.
        """
        summary_result = await self._service.get_calendar_summary(actor, period="today")
        upcoming_result = await self._service.get_upcoming_events(
            actor, hours_ahead=24, limit=self._max_events
        )
        
        return {
            "today_summary": summary_result.value if summary_result.is_success else None,
            "upcoming_events": [
                {
                    "id": e.event_id,
                    "title": e.title,
                    "start": e.start_time.isoformat() if e.start_time else None,
                    "end": e.end_time.isoformat() if e.end_time else None,
                    "location": e.location,
                }
                for e in (upcoming_result.value if upcoming_result.is_success else [])
            ],
        }


__all__ = [
    "CalendarContextInjector",
    "format_availability_response",
    "format_free_time_slots",
    "get_calendar_context_for_prompt",
]
