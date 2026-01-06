"""Multi-calendar support package.

This package provides infrastructure for managing multiple calendar backends
with a unified interface.

Core Components:
    - CalendarConfig: Per-calendar configuration
    - CalendarProviderRegistry: Manages backend instances
    - CompositeCalendarBackend: Unified interface over multiple backends
    - CalendarBackend: Abstract base class for backends
    - ICSCalendarBackend: Local ICS file backend
    - DBusCalendarBackend: Debian 12 DBus integration backend
    - GoogleCalendarBackend: Google Calendar API integration
    - OutlookCalendarBackend: Microsoft 365/Outlook integration

Example:
    from modules.Tools.Base_Tools.calendar import (
        CalendarConfig,
        CalendarBackendType,
        CalendarProviderRegistry,
        CompositeCalendarBackend,
        SyncMode,
        ICSCalendarBackend,
        initialize_calendar_system,
    )

    # Quick initialization from config
    calendar = await initialize_calendar_system(config_dict)
    events = await calendar.list_events(start, end)

    # Or manual setup with registry
    registry = CalendarProviderRegistry()
    registry.register_factory(CalendarBackendType.ICS, create_ics_backend)
    await registry.load_from_config(global_config)
    composite = CompositeCalendarBackend(registry)

    # Query all calendars
    events = await composite.list_events(start, end)

    # Create event in specific calendar
    await composite.create_event(payload, calendar="work")
"""

from .config import (
    CalendarBackendType,
    CalendarConfig,
    CalendarInfo,
    CalendarsGlobalConfig,
    SyncMode,
)
from .registry import (
    BackendFactory,
    CalendarAlreadyExistsError,
    CalendarNotFoundError,
    CalendarProviderRegistry,
    CalendarReadOnlyError,
    CalendarRegistryError,
    RegisteredCalendar,
)
from .composite import (
    CompositeCalendarBackend,
    CompositeCalendarError,
    NoWritableCalendarError,
)
from .backends import (
    CalendarBackend,
    CalendarBackendError,
    CalendarEvent,
    EventNotFoundError,
    AuthenticationError,
    RateLimitError,
    ConnectionError,
    NullCalendarBackend,
    ICSCalendarBackend,
    DBusCalendarBackend,
    GoogleCalendarBackend,
    OutlookCalendarBackend,
    CalDAVCalendarBackend,
)
from .factory import (
    create_ics_backend,
    create_dbus_backend,
    create_google_backend,
    create_outlook_backend,
    create_caldav_backend,
    create_null_backend,
    create_registry_with_defaults,
    create_calendar_registry_from_config,
    create_composite_backend,
    initialize_calendar_system,
    DEFAULT_BACKEND_FACTORIES,
)


__all__ = [
    # Config
    "CalendarBackendType",
    "CalendarConfig",
    "CalendarInfo",
    "CalendarsGlobalConfig",
    "SyncMode",
    # Registry
    "BackendFactory",
    "CalendarAlreadyExistsError",
    "CalendarNotFoundError",
    "CalendarProviderRegistry",
    "CalendarReadOnlyError",
    "CalendarRegistryError",
    "RegisteredCalendar",
    # Composite
    "CompositeCalendarBackend",
    "CompositeCalendarError",
    "NoWritableCalendarError",
    # Backends
    "CalendarBackend",
    "CalendarBackendError",
    "CalendarEvent",
    "EventNotFoundError",
    "AuthenticationError",
    "RateLimitError",
    "ConnectionError",
    "NullCalendarBackend",
    "ICSCalendarBackend",
    "DBusCalendarBackend",
    "GoogleCalendarBackend",
    "OutlookCalendarBackend",
    "CalDAVCalendarBackend",
    # Factory
    "create_ics_backend",
    "create_dbus_backend",
    "create_google_backend",
    "create_outlook_backend",
    "create_caldav_backend",
    "create_null_backend",
    "create_registry_with_defaults",
    "create_calendar_registry_from_config",
    "create_composite_backend",
    "initialize_calendar_system",
    "DEFAULT_BACKEND_FACTORIES",
]
