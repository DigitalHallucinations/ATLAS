"""
Calendar service types and data structures.

Defines the domain types for calendar events, reminders, and related
data structures used by the calendar services.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

# Re-export the existing enums from the store layer
from modules.calendar_store.dataclasses import (
    BusyStatus,
    EventStatus,
    EventVisibility,
    SyncDirection,
    SyncStatus,
)


class EventPriority(str, Enum):
    """Calendar event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EventCategory(str, Enum):
    """Calendar event categories."""
    WORK = "work"
    PERSONAL = "personal"
    HEALTH = "health"
    EDUCATION = "education"
    SOCIAL = "social"
    TRAVEL = "travel"
    OTHER = "other"


class ReminderMethod(str, Enum):
    """Reminder delivery methods."""
    NOTIFICATION = "notification"  # Desktop notification
    EMAIL = "email"              # Email notification  
    POPUP = "popup"              # UI popup dialog
    SOUND = "sound"              # Audio alert
    SPEECH = "speech"            # Text-to-speech


class ReminderStatus(str, Enum):
    """Reminder processing status."""
    SCHEDULED = "scheduled"      # Waiting to be triggered
    TRIGGERED = "triggered"      # Currently being processed
    DELIVERED = "delivered"      # Successfully delivered
    FAILED = "failed"            # Delivery failed
    CANCELLED = "cancelled"      # Cancelled before delivery
    SNOOZED = "snoozed"          # User snoozed for later


@dataclass
class CalendarReminder:
    """Calendar event reminder domain model."""
    
    reminder_id: str
    event_id: str
    method: ReminderMethod = ReminderMethod.NOTIFICATION
    minutes_before: int = 15  # Minutes before event start
    message: str | None = None  # Custom message, defaults to event title
    status: ReminderStatus = ReminderStatus.SCHEDULED
    scheduled_time: datetime | None = None  # When reminder should fire
    triggered_at: datetime | None = None   # When reminder was triggered
    delivered_at: datetime | None = None   # When reminder was delivered
    snooze_until: datetime | None = None   # If snoozed, when to retry
    retry_count: int = 0  # Number of delivery attempts
    max_retries: int = 3  # Maximum retry attempts
    tenant_id: str | None = None
    created_by: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Dict[str, Any] | None = None
    
    def is_due(self, reference_time: datetime | None = None) -> bool:
        """Check if reminder is due to be triggered."""
        if self.status != ReminderStatus.SCHEDULED or not self.scheduled_time:
            return False
        ref = reference_time or datetime.now(timezone.utc)
        return self.scheduled_time <= ref
    
    def can_retry(self) -> bool:
        """Check if reminder can be retried after failure."""
        return (
            self.status == ReminderStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    def is_snoozed(self, reference_time: datetime | None = None) -> bool:
        """Check if reminder is currently snoozed."""
        if self.status != ReminderStatus.SNOOZED or not self.snooze_until:
            return False
        ref = reference_time or datetime.now(timezone.utc)
        return self.snooze_until > ref


@dataclass
class RecurrenceRule:
    """Recurrence rule for repeating events."""
    frequency: str  # DAILY, WEEKLY, MONTHLY, YEARLY
    interval: int = 1  # Every N frequencies
    count: Optional[int] = None  # Number of occurrences
    until: Optional[datetime] = None  # End date
    by_day: Optional[List[str]] = None  # Days of week (MO, TU, etc)
    by_month_day: Optional[List[int]] = None  # Days of month
    by_month: Optional[List[int]] = None  # Months of year


@dataclass
class CalendarEvent:
    """Calendar event domain model used by the service layer."""

    event_id: str
    title: str
    description: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    timezone_name: str = "UTC"
    location: str | None = None
    status: EventStatus = EventStatus.CONFIRMED
    visibility: EventVisibility = EventVisibility.PRIVATE
    busy_status: BusyStatus = BusyStatus.BUSY
    all_day: bool = False
    category_id: UUID | None = None
    created_by: str = ""
    tenant_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_recurring: bool = False
    recurrence_pattern: str | None = None
    recurrence_exceptions: List[datetime] | None = None
    metadata: Dict[str, Any] | None = None

    def is_active(self) -> bool:
        """Return True when the event is not cancelled."""
        return self.status != EventStatus.CANCELLED

    def is_in_past(self, reference_time: datetime | None = None) -> bool:
        """Return True when the event ends before the reference time."""
        if not self.end_time:
            return False
        ref = reference_time or datetime.now(timezone.utc)
        return self.end_time < ref

    def duration_minutes(self) -> int:
        """Return the duration in minutes, falling back to zero when unknown."""
        if not self.start_time or not self.end_time:
            return 0
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)

    def overlaps_with(self, other: "CalendarEvent") -> bool:
        """Return True when the event overlaps with another event."""
        if not self.start_time or not self.end_time:
            return False
        if not other.start_time or not other.end_time:
            return False
        return self.start_time < other.end_time and self.end_time > other.start_time


@dataclass
class CalendarEventCreate:
    """Data structure for creating calendar events."""
    title: str
    description: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    timezone_name: str = "UTC"
    location: str | None = None
    status: EventStatus = EventStatus.CONFIRMED
    visibility: EventVisibility = EventVisibility.PRIVATE
    busy_status: BusyStatus = BusyStatus.BUSY
    all_day: bool = False
    category_id: UUID | None = None
    
    # Recurrence
    is_recurring: bool = False
    recurrence_pattern: str | None = None
    
    # Metadata
    metadata: Dict[str, Any] | None = None


@dataclass
class CalendarEventUpdate:
    """Data structure for updating calendar events."""
    title: str | None = None
    description: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    timezone_name: str | None = None
    location: str | None = None
    status: EventStatus | None = None
    visibility: EventVisibility | None = None
    busy_status: BusyStatus | None = None
    all_day: bool | None = None
    category_id: UUID | None = None
    
    # Recurrence
    is_recurring: bool | None = None
    recurrence_pattern: str | None = None
    recurrence_exceptions: List[datetime] | None = None
    
    # Metadata
    metadata: Dict[str, Any] | None = None


@dataclass 
class CalendarEventFilter:
    """Filter criteria for querying calendar events."""
    start_date: datetime | None = None
    end_date: datetime | None = None
    category_ids: List[UUID] | None = None
    status: List[EventStatus] | None = None
    visibility: List[EventVisibility] | None = None
    search_query: str | None = None
    include_recurring: bool = True
    include_all_day: bool = True
    user_id: str | None = None


@dataclass
class EventReminder:
    """Reminder configuration for calendar events."""
    id: UUID
    event_id: UUID
    reminder_time: datetime
    method: str  # email, notification, popup, etc.
    message: str | None = None
    delivered: bool = False
    delivered_at: datetime | None = None
    
    def is_due(self, current_time: datetime | None = None) -> bool:
        """Check if the reminder is due."""
        current = current_time or datetime.now(timezone.utc)
        return not self.delivered and self.reminder_time <= current


@dataclass
class CalendarCategory:
    """Calendar category/calendar grouping."""
    id: UUID
    name: str
    description: str | None
    color: str | None
    user_id: str
    tenant_id: str
    is_default: bool = False
    sync_enabled: bool = False
    sync_direction: SyncDirection | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class EventConflict:
    """Represents a scheduling conflict between events."""
    event1: CalendarEvent
    event2: CalendarEvent
    overlap_start: datetime
    overlap_end: datetime
    severity: str  # "minor", "major", "critical"
    
    def overlap_duration_minutes(self) -> int:
        """Get overlap duration in minutes."""
        delta = self.overlap_end - self.overlap_start
        return int(delta.total_seconds() / 60)


@dataclass
class EventRecurrenceInfo:
    """Detailed recurrence information for an event."""
    pattern: str
    frequency: str  # daily, weekly, monthly, yearly
    interval: int = 1
    end_date: datetime | None = None
    count: int | None = None
    by_weekday: List[int] | None = None  # 0=Monday, 6=Sunday
    by_monthday: List[int] | None = None
    by_month: List[int] | None = None
    
    def is_infinite(self) -> bool:
        """Check if recurrence has no end."""
        return self.end_date is None and self.count is None


@dataclass
class CalendarEventStats:
    """Statistics about calendar events."""
    total_events: int
    upcoming_events: int
    overdue_events: int
    events_today: int
    events_this_week: int
    events_this_month: int
    busiest_day: datetime | None = None
    average_duration_minutes: float = 0.0


# Repository protocol for calendar events
from typing import Protocol
from core.services.common import Repository


class CalendarEventRepositoryProtocol(Repository[CalendarEvent, UUID], Protocol):
    """Protocol for calendar event repository operations."""
    
    async def find_events_by_filter(
        self, 
        filter_criteria: CalendarEventFilter, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[CalendarEvent]:
        """Find events matching filter criteria."""
        ...
    
    async def find_conflicts(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: str,
        exclude_event_id: UUID | None = None,
    ) -> List[CalendarEvent]:
        """Find events that conflict with the given time range."""
        ...
    
    async def get_events_for_reminder_processing(
        self,
        check_time: datetime,
    ) -> List[CalendarEvent]:
        """Get events that need reminder processing."""
        ...


class ReminderRepositoryProtocol(Repository[EventReminder, UUID], Protocol):
    """Protocol for reminder repository operations."""
    
    async def find_due_reminders(
        self, 
        current_time: datetime
    ) -> List[EventReminder]:
        """Find reminders that are due for delivery."""
        ...
    
    async def mark_delivered(
        self, 
        reminder_id: UUID, 
        delivered_at: datetime
    ) -> bool:
        """Mark a reminder as delivered."""
        ...


class CalendarCategoryRepositoryProtocol(Repository[CalendarCategory, UUID], Protocol):
    """Protocol for calendar category repository operations."""
    
    async def find_by_user(self, user_id: str) -> List[CalendarCategory]:
        """Find all categories for a user."""
        ...
    
    async def get_default_category(self, user_id: str) -> CalendarCategory | None:
        """Get the default category for a user."""
        ...