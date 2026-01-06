"""Calendar backend factory and initialization.

This module provides factory functions for creating calendar backends from
configuration, and convenience functions for setting up the multi-calendar
system.

Example:
    # Initialize from global configuration
    registry = await create_calendar_registry_from_config(config_manager)
    composite = CompositeCalendarBackend(registry)

    # Use composite backend
    events = await composite.list_events(start, end)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TYPE_CHECKING,
)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import (
    CalendarBackendType,
    CalendarConfig,
    CalendarsGlobalConfig,
)
from .registry import CalendarProviderRegistry, BackendFactory
from .composite import CompositeCalendarBackend
from .backends import (
    CalendarBackend,
    ICSCalendarBackend,
    DBusCalendarBackend,
    GoogleCalendarBackend,
    OutlookCalendarBackend,
    CalDAVCalendarBackend,
    NullCalendarBackend,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_ics_backend(config: CalendarConfig) -> CalendarBackend:
    """Factory function for ICS backends.

    Args:
        config: Calendar configuration.

    Returns:
        Configured ICSCalendarBackend instance.
    """
    return ICSCalendarBackend.from_config(config)


def create_dbus_backend(
    config: CalendarConfig,
    dbus_client: Optional[Any] = None,
) -> CalendarBackend:
    """Factory function for DBus backends.

    Args:
        config: Calendar configuration.
        dbus_client: Optional DBus calendar client. If not provided,
            will attempt to create one.

    Returns:
        Configured DBusCalendarBackend instance.

    Raises:
        ImportError: If DBus client cannot be created and none provided.
    """
    if dbus_client is None:
        # Attempt to import and create a DBus client
        try:
            from modules.Tools.Base_Tools.debian12_calendar import (
                get_dbus_calendar_client,
            )
            dbus_client = get_dbus_calendar_client()
        except ImportError:
            logger.warning(
                "DBus calendar client not available; calendar '%s' will be disabled",
                config.name,
            )
            return NullCalendarBackend()

    return DBusCalendarBackend.from_config(config, dbus_client)


def create_google_backend(config: CalendarConfig) -> CalendarBackend:
    """Factory function for Google Calendar backends.

    Args:
        config: Calendar configuration.

    Returns:
        Configured GoogleCalendarBackend instance.
    """
    try:
        return GoogleCalendarBackend.from_config(config)
    except ImportError:
        logger.warning(
            "Google Calendar dependencies not available; calendar '%s' will be disabled",
            config.name,
        )
        return NullCalendarBackend()


def create_outlook_backend(config: CalendarConfig) -> CalendarBackend:
    """Factory function for Outlook/Office 365 Calendar backends.

    Args:
        config: Calendar configuration.

    Returns:
        Configured OutlookCalendarBackend instance.
    """
    try:
        return OutlookCalendarBackend.from_config(config)
    except (ImportError, ValueError) as exc:
        logger.warning(
            "Outlook Calendar backend not available; calendar '%s' will be disabled: %s",
            config.name,
            exc,
        )
        return NullCalendarBackend()


def create_null_backend(config: CalendarConfig) -> CalendarBackend:
    """Factory function for null/disabled backends.

    Args:
        config: Calendar configuration (ignored).

    Returns:
        NullCalendarBackend instance.
    """
    return NullCalendarBackend()


def create_caldav_backend(config: CalendarConfig) -> CalendarBackend:
    """Factory function for CalDAV backends.

    Supports CalDAV servers such as Nextcloud, Fastmail, iCloud, Radicale, etc.

    Args:
        config: Calendar configuration.

    Returns:
        Configured CalDAVCalendarBackend instance.
    """
    try:
        return CalDAVCalendarBackend.from_config(config)
    except ImportError:
        logger.warning(
            "CalDAV dependencies not available; calendar '%s' will be disabled. "
            "Install with: pip install caldav",
            config.name,
        )
        return NullCalendarBackend()
    except Exception as exc:
        logger.warning(
            "CalDAV backend initialization failed; calendar '%s' will be disabled: %s",
            config.name,
            exc,
        )
        return NullCalendarBackend()


# Mapping of backend types to their factory functions
DEFAULT_BACKEND_FACTORIES: Dict[CalendarBackendType, BackendFactory] = {
    CalendarBackendType.ICS: create_ics_backend,
    CalendarBackendType.DBUS: create_dbus_backend,
    CalendarBackendType.GOOGLE: create_google_backend,
    CalendarBackendType.OUTLOOK: create_outlook_backend,
    CalendarBackendType.CALDAV: create_caldav_backend,
}


def create_registry_with_defaults(
    additional_factories: Optional[Dict[CalendarBackendType, BackendFactory]] = None,
) -> CalendarProviderRegistry:
    """Create a registry with default backend factories registered.

    Args:
        additional_factories: Optional additional factories to register.

    Returns:
        Configured CalendarProviderRegistry.
    """
    registry = CalendarProviderRegistry()

    # Register default factories
    for backend_type, factory in DEFAULT_BACKEND_FACTORIES.items():
        registry.register_factory(backend_type, factory)

    # Register additional factories
    if additional_factories:
        for backend_type, factory in additional_factories.items():
            registry.register_factory(backend_type, factory)

    return registry


async def create_calendar_registry_from_config(
    config_dict: Dict[str, Any],
    additional_factories: Optional[Dict[CalendarBackendType, BackendFactory]] = None,
) -> CalendarProviderRegistry:
    """Create and populate a calendar registry from configuration dictionary.

    This is the primary entry point for initializing the multi-calendar system.

    Args:
        config_dict: Configuration dictionary containing 'calendars' section.
        additional_factories: Optional additional backend factories.

    Returns:
        Populated CalendarProviderRegistry.

    Example:
        config = config_manager.get_config()
        registry = await create_calendar_registry_from_config(config)
        composite = CompositeCalendarBackend(registry)
    """
    registry = create_registry_with_defaults(additional_factories)

    calendars_section = config_dict.get("calendars", {})
    if not calendars_section:
        logger.info("No calendars configured; using empty registry")
        return registry

    global_config = CalendarsGlobalConfig.from_dict(calendars_section)
    await registry.load_from_config(global_config)

    return registry


def create_composite_backend(
    registry: CalendarProviderRegistry,
    query_timeout: float = 30.0,
) -> CompositeCalendarBackend:
    """Create a composite backend from a registry.

    Args:
        registry: Populated calendar registry.
        query_timeout: Timeout for parallel queries (seconds).

    Returns:
        Configured CompositeCalendarBackend.
    """
    return CompositeCalendarBackend(
        registry=registry,
        query_timeout=query_timeout,
    )


async def initialize_calendar_system(
    config_dict: Dict[str, Any],
    additional_factories: Optional[Dict[CalendarBackendType, BackendFactory]] = None,
    query_timeout: float = 30.0,
) -> CompositeCalendarBackend:
    """Initialize the complete calendar system from configuration.

    This is a convenience function that combines registry creation and
    composite backend creation.

    Args:
        config_dict: Configuration dictionary containing 'calendars' section.
        additional_factories: Optional additional backend factories.
        query_timeout: Timeout for parallel queries (seconds).

    Returns:
        Fully initialized CompositeCalendarBackend ready for use.

    Example:
        config = config_manager.get_config()
        calendar = await initialize_calendar_system(config)

        # Now use the unified calendar interface
        events = await calendar.list_events(start, end)
        await calendar.create_event(payload, calendar="personal")
    """
    registry = await create_calendar_registry_from_config(
        config_dict,
        additional_factories,
    )
    return create_composite_backend(registry, query_timeout)


__all__ = [
    # Factory functions
    "create_ics_backend",
    "create_dbus_backend",
    "create_google_backend",
    "create_outlook_backend",
    "create_caldav_backend",
    "create_null_backend",
    # Registry setup
    "create_registry_with_defaults",
    "create_calendar_registry_from_config",
    "create_composite_backend",
    # High-level initialization
    "initialize_calendar_system",
    # Default factories
    "DEFAULT_BACKEND_FACTORIES",
]
