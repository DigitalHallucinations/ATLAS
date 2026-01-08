"""
ATLAS Calendar Services

Provides calendar event management, synchronization, and reminder services
following the ATLAS service pattern.

Author: ATLAS Team  
Date: Jan 7, 2026
"""

from .event_service import CalendarEventService
from .reminder_service import ReminderService
from .sync_service import (
    CalendarSyncService,
    create_sync_service,
    SyncConfiguration,
    SyncResult,
    SyncProgress,
    SyncConflict,
    SyncDirection,
    SyncStatus,
    ConflictResolution,
    ProviderInfo,
    CalendarSyncStarted,
    CalendarSyncProgress,
    CalendarSyncCompleted,
    CalendarSyncConflict,
    CalendarSyncFailed,
)
from .permissions import CalendarPermissionChecker
from .validation import CalendarEventValidator
from .job_task_integration import JobTaskEventHandler, create_job_task_handler
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
from .context import (
    CalendarContextInjector,
    format_availability_response,
    format_free_time_slots,
    get_calendar_context_for_prompt,
)

__all__ = [
    # Services
    "CalendarEventService", 
    "ReminderService",
    "CalendarSyncService",
    "create_sync_service",
    "CalendarPermissionChecker",
    "CalendarEventValidator",
    
    # Job/Task Integration
    "JobTaskEventHandler",
    "create_job_task_handler",
    
    # Context Utilities
    "CalendarContextInjector",
    "format_availability_response",
    "format_free_time_slots",
    "get_calendar_context_for_prompt",
    
    # Sync Types
    "SyncConfiguration",
    "SyncResult",
    "SyncProgress",
    "SyncConflict",
    "SyncDirection",
    "SyncStatus",
    "ConflictResolution",
    "ProviderInfo",
    
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
    
    # Sync Events
    "CalendarSyncStarted",
    "CalendarSyncProgress",
    "CalendarSyncCompleted",
    "CalendarSyncConflict",
    "CalendarSyncFailed",
]