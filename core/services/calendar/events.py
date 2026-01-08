"""
Calendar domain events.

Events published when calendar-related changes occur, enabling
loose coupling between services and reactive UI updates.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union
from uuid import UUID

from core.services.common import DomainEvent

# Type alias for actor types
ActorType = Literal["user", "system", "agent", "sync", "job", "task"]


def _safe_entity_id(event_id: Union[str, UUID]) -> UUID:
    """Convert event_id to UUID."""
    if isinstance(event_id, UUID):
        return event_id
    return UUID(event_id)


@dataclass(frozen=True)
class CalendarEventCreated(DomainEvent):
    """Published when a calendar event is created."""
    
    # Use default values to avoid dataclass issues with inheritance
    event_title: str = ""
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    category_id: Optional[UUID] = None
    
    @classmethod
    def create_for_event(
        cls,
        event_id: str | UUID,
        tenant_id: str,
        actor_type: ActorType,
        event_title: str,
        event_start: datetime,
        event_end: datetime,
        category_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "CalendarEventCreated":
        """Create a calendar event created event with proper data."""
        return cls(
            event_type="calendar.event_created",
            entity_id=_safe_entity_id(event_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            event_title=event_title,
            event_start=event_start,
            event_end=event_end,
            category_id=category_id,
        )


@dataclass(frozen=True) 
class CalendarEventUpdated(DomainEvent):
    """Published when a calendar event is updated."""
    
    event_title: str = ""
    original_start: Optional[datetime] = None
    new_start: Optional[datetime] = None
    original_end: Optional[datetime] = None  
    new_end: Optional[datetime] = None
    changed_fields: Optional[list[str]] = None
    
    def __post_init__(self):
        if self.changed_fields is None:
            object.__setattr__(self, 'changed_fields', [])
    
    @classmethod
    def create_for_event(
        cls,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        event_title: str,
        changed_fields: list[str],
        original_start: Optional[datetime] = None,
        new_start: Optional[datetime] = None,
        original_end: Optional[datetime] = None,
        new_end: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "CalendarEventUpdated":
        """Create a calendar event updated event."""
        return cls(
            event_type="calendar.event_updated",
            entity_id=_safe_entity_id(event_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            event_title=event_title,
            original_start=original_start,
            new_start=new_start,
            original_end=original_end,
            new_end=new_end,
            changed_fields=changed_fields or [],
        )


@dataclass(frozen=True)
class CalendarEventDeleted(DomainEvent):
    """Published when a calendar event is deleted."""
    
    event_title: str = ""
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    soft_delete: bool = False
    
    @classmethod
    def create_for_event(
        cls,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        event_title: str,
        event_start: datetime,
        event_end: datetime,
        soft_delete: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "CalendarEventDeleted":
        """Create a calendar event deleted event."""
        return cls(
            event_type="calendar.event_deleted",
            entity_id=_safe_entity_id(event_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            event_title=event_title,
            event_start=event_start,
            event_end=event_end,
            soft_delete=soft_delete,
        )


@dataclass(frozen=True)
class CalendarEventRescheduled(DomainEvent):
    """Published when a calendar event is rescheduled (time changed)."""
    
    event_title: str = ""
    original_start: Optional[datetime] = None
    new_start: Optional[datetime] = None
    original_end: Optional[datetime] = None
    new_end: Optional[datetime] = None
    reschedule_reason: str = ""
    
    @classmethod
    def create_for_event(
        cls,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        event_title: str,
        original_start: datetime,
        new_start: datetime,
        original_end: datetime,
        new_end: datetime,
        reschedule_reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "CalendarEventRescheduled":
        """Create a calendar event rescheduled event."""
        return cls(
            event_type="calendar.event_rescheduled",
            entity_id=_safe_entity_id(event_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            event_title=event_title,
            original_start=original_start,
            new_start=new_start,
            original_end=original_end,
            new_end=new_end,
            reschedule_reason=reschedule_reason,
        )


@dataclass(frozen=True)
class CalendarEventCancelled(DomainEvent):
    """Published when a calendar event is cancelled."""
    
    event_title: str = ""
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    cancellation_reason: str = ""
    notify_attendees: bool = True
    
    @classmethod
    def create_for_event(
        cls,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        event_title: str,
        event_start: datetime,
        event_end: datetime,
        cancellation_reason: str = "",
        notify_attendees: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "CalendarEventCancelled":
        """Create a calendar event cancelled event."""
        return cls(
            event_type="calendar.event_cancelled",
            entity_id=_safe_entity_id(event_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            event_title=event_title,
            event_start=event_start,
            event_end=event_end,
            cancellation_reason=cancellation_reason,
            notify_attendees=notify_attendees,
        )


@dataclass(frozen=True)
class RecurringEventSeriesModified(DomainEvent):
    """Published when a recurring event series is modified."""
    
    series_title: str = ""
    recurrence_pattern: str = ""
    modification_type: str = ""  # "all", "future", "single"
    affected_instances: int = 0
    
    @classmethod
    def create_for_series(
        cls,
        series_id: str,
        tenant_id: str,
        actor_type: ActorType,
        series_title: str,
        recurrence_pattern: str,
        modification_type: str,
        affected_instances: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "RecurringEventSeriesModified":
        """Create a recurring event series modified event."""
        return cls(
            event_type="calendar.recurring_series_modified",
            entity_id=_safe_entity_id(series_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            series_title=series_title,
            recurrence_pattern=recurrence_pattern,
            modification_type=modification_type,
            affected_instances=affected_instances,
        )


@dataclass(frozen=True)
class ReminderScheduled(DomainEvent):
    """Published when a calendar reminder is scheduled."""
    
    calendar_event_id: str = ""
    scheduled_time: Optional[datetime] = None
    method: str = "notification"
    
    @classmethod
    def create_for_reminder(
        cls,
        reminder_id: str | UUID,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        scheduled_time: datetime,
        method: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "ReminderScheduled":
        """Create a reminder scheduled event."""
        return cls(
            event_type="calendar.reminder.scheduled",
            entity_id=_safe_entity_id(reminder_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            calendar_event_id=event_id,
            scheduled_time=scheduled_time,
            method=method,
        )


@dataclass(frozen=True)
class ReminderTriggered(DomainEvent):
    """Published when a calendar reminder is triggered."""
    
    calendar_event_id: str = ""
    method: str = "notification"
    message: str = ""
    
    @classmethod
    def create_for_reminder(
        cls,
        reminder_id: str | UUID,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        method: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "ReminderTriggered":
        """Create a reminder triggered event."""
        return cls(
            event_type="calendar.reminder.triggered",
            entity_id=_safe_entity_id(reminder_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            calendar_event_id=event_id,
            method=method,
            message=message,
        )


@dataclass(frozen=True)
class ReminderDelivered(DomainEvent):
    """Published when a calendar reminder delivery is attempted."""
    
    calendar_event_id: str = ""
    method: str = "notification"
    delivery_success: bool = False
    
    @classmethod
    def create_for_reminder(
        cls,
        reminder_id: str | UUID,
        event_id: str,
        tenant_id: str,
        actor_type: ActorType,
        method: str,
        delivery_success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "ReminderDelivered":
        """Create a reminder delivered event."""
        return cls(
            event_type="calendar.reminder.delivered",
            entity_id=_safe_entity_id(reminder_id),
            tenant_id=tenant_id,
            actor=actor_type,
            metadata=metadata or {},
            timestamp=timestamp or datetime.utcnow(),
            calendar_event_id=event_id,
            method=method,
            delivery_success=delivery_success,
        )