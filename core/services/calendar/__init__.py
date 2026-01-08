"""
ATLAS Calendar Services

Provides calendar event management, synchronization, and reminder services
following the ATLAS service pattern.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

from .event_service import CalendarEventService
from .reminder_service import ReminderService
from .permissions import CalendarPermissionChecker
from .validation import CalendarEventValidator
from .types import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventUpdate,
    CalendarReminder,
    EventStatus,
    EventPriority,
    EventVisibility,
    EventCategory,
    RecurrenceRule,
    ReminderMethod,
    ReminderStatus,
)
from .events import (
    CalendarEventCreated,
    CalendarEventUpdated,
    CalendarEventDeleted,
    CalendarEventRescheduled,
    CalendarEventCancelled,
    RecurringEventSeriesModified,
    ReminderScheduled,
    ReminderTriggered,
    ReminderDelivered,
)

__all__ = [
    # Services
    "CalendarEventService", 
    "ReminderService",
    "CalendarPermissionChecker",
    "CalendarEventValidator",
    
    # Types
    "CalendarEvent",
    "CalendarEventCreate", 
    "CalendarEventUpdate",
    "CalendarReminder",
    "EventStatus",
    "EventPriority", 
    "EventVisibility",
    "EventCategory",
    "RecurrenceRule",
    "ReminderMethod",
    "ReminderStatus",
    
    # Events
    "CalendarEventCreated",
    "CalendarEventUpdated",
    "CalendarEventDeleted", 
    "CalendarEventRescheduled",
    "CalendarEventCancelled",
    "RecurringEventSeriesModified",
    "ReminderScheduled",
    "ReminderTriggered",
    "ReminderDelivered",
]