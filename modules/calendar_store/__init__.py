"""ATLAS Master Calendar Store.

This module provides the centralized calendar storage system for ATLAS,
serving as the authoritative source for all calendar data. External calendars
(Google, Outlook, CalDAV, ICS) sync into this master store.

Core Components:
    - CalendarCategory: Category/calendar organization (dataclass)
    - CalendarEvent: Event data model (dataclass)
    - CalendarCategoryModel: SQLAlchemy model for categories
    - CalendarEventModel: SQLAlchemy model for events
    - CalendarStoreRepository: Main repository for CRUD operations

Example:
    from modules.calendar_store import (
        CalendarStoreRepository,
        CalendarCategory,
        CalendarEvent,
        create_calendar_engine,
    )
    from sqlalchemy.orm import sessionmaker

    # Create engine and repository
    engine = create_calendar_engine("postgresql://user:pass@host/db")
    Session = sessionmaker(bind=engine)
    repo = CalendarStoreRepository(Session)

    # Create schema (with built-in categories)
    repo.create_schema()

    # Work with categories
    categories = repo.list_categories()
    work_category = repo.get_category_by_slug("work")

    # Work with events
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    events = repo.list_events(start=now, end=now + timedelta(days=7))
    event = repo.create_event(
        title="Team Meeting",
        start_time=now,
        end_time=now + timedelta(hours=1),
        category_id=work_category["id"],
    )

    # Search events
    results = repo.search_events("meeting")
"""

from .models import (
    Base,
    CalendarCategoryModel,
    CalendarEventModel,
    CalendarImportMappingModel,
    CalendarSyncStateModel,
    ensure_calendar_schema,
)
from .dataclasses import (
    # Dataclasses
    CalendarCategory,
    CalendarEvent,
    CalendarReminder,
    Attendee,
    ImportMapping,
    SyncState,
    # Enums
    EventStatus,
    EventVisibility,
    BusyStatus,
    SyncDirection,
    SyncStatus,
    ReminderMethod,
    AttendeeStatus,
    AttendeeRole,
    # Constants
    BUILTIN_CATEGORIES,
    COLOR_PALETTE,
)
from .schema import (
    create_calendar_engine,
    create_schema,
    seed_builtin_categories,
    reset_schema,
)
from .repository import (
    CalendarStoreRepository,
    CalendarStoreError,
    CategoryNotFoundError,
    EventNotFoundError,
    CategorySlugExistsError,
    ReadOnlyCategoryError,
)
from .recurrence import (
    # Recurrence classes
    RecurrenceRule,
    RecurrenceExpander,
    # Enums
    Frequency,
    Weekday,
    # Helper functions
    daily,
    weekly,
    monthly_by_day,
    monthly_by_weekday,
    yearly,
    workdays,
    describe_recurrence,
)
from .reminders import (
    # Reminder classes
    Reminder,
    ScheduledReminder,
    ReminderScheduler,
    # Enums
    ReminderMethod as ReminderMethodEnum,
    ReminderStatus,
    # Helper functions
    create_default_reminders,
    parse_reminder_from_natural_language,
    get_reminder_preset_labels,
    minutes_from_preset_label,
    # Constants
    REMINDER_PRESETS,
    DEFAULT_SNOOZE_OPTIONS,
)
from .sync_engine import (
    # Sync engine classes
    SyncEngine,
    CalendarSyncProvider,
    ExternalEvent,
    SyncConflict,
    SyncResult,
    # Enums
    SyncDirection as SyncEngineDirection,
    SyncStatus as SyncEngineStatus,
    ConflictResolution,
)
from .providers import (
    ICSProvider,
    CalDAVProvider,
)

__all__ = [
    # SQLAlchemy models
    "Base",
    "CalendarCategoryModel",
    "CalendarEventModel",
    "CalendarImportMappingModel",
    "CalendarSyncStateModel",
    "ensure_calendar_schema",
    # Dataclasses
    "CalendarCategory",
    "CalendarEvent",
    "CalendarReminder",
    "Attendee",
    "ImportMapping",
    "SyncState",
    # Enums
    "EventStatus",
    "EventVisibility",
    "BusyStatus",
    "SyncDirection",
    "SyncStatus",
    "ReminderMethod",
    "AttendeeStatus",
    "AttendeeRole",
    # Constants
    "BUILTIN_CATEGORIES",
    "COLOR_PALETTE",
    # Schema utilities
    "create_calendar_engine",
    "create_schema",
    "seed_builtin_categories",
    "reset_schema",
    # Repository
    "CalendarStoreRepository",
    # Exceptions
    "CalendarStoreError",
    "CategoryNotFoundError",
    "EventNotFoundError",
    "CategorySlugExistsError",
    "ReadOnlyCategoryError",
    # Recurrence
    "RecurrenceRule",
    "RecurrenceExpander",
    "Frequency",
    "Weekday",
    "daily",
    "weekly",
    "monthly_by_day",
    "monthly_by_weekday",
    "yearly",
    "workdays",
    "describe_recurrence",
    # Reminders
    "Reminder",
    "ScheduledReminder",
    "ReminderScheduler",
    "ReminderMethodEnum",
    "ReminderStatus",
    "create_default_reminders",
    "parse_reminder_from_natural_language",
    "get_reminder_preset_labels",
    "minutes_from_preset_label",
    "REMINDER_PRESETS",
    "DEFAULT_SNOOZE_OPTIONS",
    # Sync engine
    "SyncEngine",
    "CalendarSyncProvider",
    "ExternalEvent",
    "SyncConflict",
    "SyncResult",
    "SyncEngineDirection",
    "SyncEngineStatus",
    "ConflictResolution",
    # Providers
    "ICSProvider",
    "CalDAVProvider",
]
