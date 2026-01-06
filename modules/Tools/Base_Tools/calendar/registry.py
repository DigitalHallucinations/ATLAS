"""Calendar provider registry.

This module provides the CalendarProviderRegistry which manages multiple
calendar backend instances, handling registration, lookup, and lifecycle.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from .config import (
    CalendarBackendType,
    CalendarConfig,
    CalendarInfo,
    CalendarsGlobalConfig,
    SyncMode,
)

if TYPE_CHECKING:
    from ..debian12_calendar import CalendarBackend

logger = logging.getLogger(__name__)


class CalendarRegistryError(RuntimeError):
    """Base exception for registry operations."""


class CalendarNotFoundError(CalendarRegistryError):
    """Raised when a requested calendar is not registered."""


class CalendarAlreadyExistsError(CalendarRegistryError):
    """Raised when attempting to register a calendar that already exists."""


class CalendarReadOnlyError(CalendarRegistryError):
    """Raised when attempting to write to a read-only calendar."""


BackendFactory = Callable[[CalendarConfig], "CalendarBackend"]


@dataclass
class RegisteredCalendar:
    """A registered calendar with its backend and metadata."""

    config: CalendarConfig
    backend: "CalendarBackend"
    last_sync: Optional[_dt.datetime] = None
    is_connected: bool = True
    error_message: Optional[str] = None
    _event_count: Optional[int] = None

    def to_info(self) -> CalendarInfo:
        """Create a CalendarInfo summary."""
        return CalendarInfo(
            name=self.config.name,
            display_name=self.config.effective_display_name,
            backend_type=self.config.backend_type,
            color=self.config.color,
            write_enabled=self.config.write_enabled,
            sync_mode=self.config.sync_mode,
            is_connected=self.is_connected,
            last_sync=self.last_sync,
            event_count=self._event_count,
            error_message=self.error_message,
        )


class CalendarProviderRegistry:
    """Registry for managing multiple calendar backends.

    The registry handles:
    - Registration and deregistration of calendar backends
    - Backend factory management for different backend types
    - Lookup by calendar name
    - Lifecycle management (initialization, shutdown)

    Example:
        registry = CalendarProviderRegistry()
        registry.register_factory(CalendarBackendType.ICS, create_ics_backend)
        registry.register_factory(CalendarBackendType.GOOGLE, create_google_backend)

        await registry.load_from_config(global_config)

        backend = registry.get_backend("work")
        events = await backend.list_events(start, end)
    """

    def __init__(self) -> None:
        self._calendars: Dict[str, RegisteredCalendar] = {}
        self._factories: Dict[CalendarBackendType, BackendFactory] = {}
        self._global_config: Optional[CalendarsGlobalConfig] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return True if the registry has been initialized."""
        return self._initialized

    @property
    def calendar_names(self) -> List[str]:
        """Return list of registered calendar names."""
        return list(self._calendars.keys())

    @property
    def default_calendar_name(self) -> Optional[str]:
        """Return the default calendar name from config."""
        if self._global_config:
            default_config = self._global_config.get_default_calendar()
            if default_config:
                return default_config.name
        return None

    # -------------------------------------------------------------------------
    # Factory Management
    # -------------------------------------------------------------------------

    def register_factory(
        self,
        backend_type: CalendarBackendType,
        factory: BackendFactory,
    ) -> None:
        """Register a factory function for creating backends of a given type.

        Args:
            backend_type: The backend type this factory creates.
            factory: Callable that takes CalendarConfig and returns CalendarBackend.
        """
        self._factories[backend_type] = factory
        logger.debug("Registered factory for backend type: %s", backend_type.value)

    def get_factory(self, backend_type: CalendarBackendType) -> Optional[BackendFactory]:
        """Get the factory for a backend type."""
        return self._factories.get(backend_type)

    # -------------------------------------------------------------------------
    # Calendar Registration
    # -------------------------------------------------------------------------

    def register(
        self,
        config: CalendarConfig,
        backend: Optional["CalendarBackend"] = None,
    ) -> RegisteredCalendar:
        """Register a calendar with the registry.

        Args:
            config: Configuration for the calendar.
            backend: Optional pre-created backend. If not provided,
                    the appropriate factory will be used.

        Returns:
            The RegisteredCalendar entry.

        Raises:
            CalendarAlreadyExistsError: If a calendar with this name exists.
            CalendarRegistryError: If no factory is available for the backend type.
        """
        if config.name in self._calendars:
            raise CalendarAlreadyExistsError(
                f"Calendar '{config.name}' is already registered"
            )

        if backend is None:
            factory = self._factories.get(config.backend_type)
            if factory is None:
                raise CalendarRegistryError(
                    f"No factory registered for backend type: {config.backend_type.value}"
                )
            try:
                backend = factory(config)
            except Exception as exc:
                logger.error(
                    "Failed to create backend for calendar '%s': %s",
                    config.name,
                    exc,
                )
                raise CalendarRegistryError(
                    f"Failed to create backend for '{config.name}'"
                ) from exc

        registered = RegisteredCalendar(config=config, backend=backend)
        self._calendars[config.name] = registered
        logger.info(
            "Registered calendar '%s' (type: %s)",
            config.name,
            config.backend_type.value,
        )
        return registered

    def unregister(self, name: str) -> bool:
        """Unregister a calendar by name.

        Args:
            name: The calendar name to unregister.

        Returns:
            True if the calendar was removed, False if it wasn't registered.
        """
        if name in self._calendars:
            del self._calendars[name]
            logger.info("Unregistered calendar: %s", name)
            return True
        return False

    # -------------------------------------------------------------------------
    # Calendar Lookup
    # -------------------------------------------------------------------------

    def get(self, name: str) -> Optional[RegisteredCalendar]:
        """Get a registered calendar by name."""
        return self._calendars.get(name)

    def get_backend(self, name: str) -> "CalendarBackend":
        """Get a calendar backend by name.

        Args:
            name: The calendar name.

        Returns:
            The CalendarBackend instance.

        Raises:
            CalendarNotFoundError: If the calendar is not registered.
        """
        registered = self._calendars.get(name)
        if registered is None:
            raise CalendarNotFoundError(f"Calendar '{name}' is not registered")
        return registered.backend

    def get_config(self, name: str) -> Optional[CalendarConfig]:
        """Get the configuration for a calendar."""
        registered = self._calendars.get(name)
        return registered.config if registered else None

    def get_writable_backend(self, name: str) -> "CalendarBackend":
        """Get a backend, ensuring it supports writes.

        Args:
            name: The calendar name.

        Returns:
            The CalendarBackend instance.

        Raises:
            CalendarNotFoundError: If the calendar is not registered.
            CalendarReadOnlyError: If the calendar is read-only.
        """
        registered = self._calendars.get(name)
        if registered is None:
            raise CalendarNotFoundError(f"Calendar '{name}' is not registered")
        if registered.config.is_read_only:
            raise CalendarReadOnlyError(
                f"Calendar '{name}' is read-only (sync_mode: {registered.config.sync_mode.value})"
            )
        return registered.backend

    # -------------------------------------------------------------------------
    # Listing
    # -------------------------------------------------------------------------

    def list_calendars(self) -> List[CalendarInfo]:
        """List all registered calendars as CalendarInfo objects."""
        calendars = [
            registered.to_info()
            for registered in self._calendars.values()
        ]
        # Sort by priority, then name
        return sorted(
            calendars,
            key=lambda c: (
                self._calendars[c.name].config.priority,
                c.name,
            ),
        )

    def list_backends(self) -> List["CalendarBackend"]:
        """List all registered backends."""
        return [registered.backend for registered in self._calendars.values()]

    def list_writable_calendars(self) -> List[CalendarInfo]:
        """List calendars that support write operations."""
        return [
            info for info in self.list_calendars()
            if info.write_enabled
        ]

    # -------------------------------------------------------------------------
    # Configuration Loading
    # -------------------------------------------------------------------------

    async def load_from_config(
        self,
        config: CalendarsGlobalConfig,
        *,
        clear_existing: bool = True,
    ) -> List[str]:
        """Load calendars from global configuration.

        Args:
            config: The global calendar configuration.
            clear_existing: If True, unregister all existing calendars first.

        Returns:
            List of calendar names that were successfully registered.
        """
        if clear_existing:
            self._calendars.clear()

        self._global_config = config
        registered_names: List[str] = []
        errors: List[str] = []

        for cal_config in config.calendars:
            try:
                self.register(cal_config)
                registered_names.append(cal_config.name)
            except Exception as exc:
                error_msg = f"Failed to register '{cal_config.name}': {exc}"
                logger.warning(error_msg)
                errors.append(error_msg)

        self._initialized = True
        logger.info(
            "Loaded %d calendars from config (%d errors)",
            len(registered_names),
            len(errors),
        )
        return registered_names

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize_all(self) -> Dict[str, Optional[str]]:
        """Initialize all registered backends.

        Returns:
            Dict mapping calendar name to error message (None if success).
        """
        results: Dict[str, Optional[str]] = {}

        for name, registered in self._calendars.items():
            try:
                # Check if backend has an initialize method
                if hasattr(registered.backend, "initialize"):
                    await registered.backend.initialize()
                registered.is_connected = True
                registered.error_message = None
                results[name] = None
            except Exception as exc:
                registered.is_connected = False
                registered.error_message = str(exc)
                results[name] = str(exc)
                logger.warning("Failed to initialize calendar '%s': %s", name, exc)

        return results

    async def shutdown_all(self) -> None:
        """Shutdown all registered backends."""
        for name, registered in self._calendars.items():
            try:
                if hasattr(registered.backend, "shutdown"):
                    await registered.backend.shutdown()
            except Exception as exc:
                logger.warning("Error shutting down calendar '%s': %s", name, exc)

    def update_sync_status(
        self,
        name: str,
        *,
        last_sync: Optional[_dt.datetime] = None,
        is_connected: Optional[bool] = None,
        error_message: Optional[str] = None,
        event_count: Optional[int] = None,
    ) -> None:
        """Update sync status for a calendar."""
        registered = self._calendars.get(name)
        if registered is None:
            return

        if last_sync is not None:
            registered.last_sync = last_sync
        if is_connected is not None:
            registered.is_connected = is_connected
        if error_message is not None:
            registered.error_message = error_message
        if event_count is not None:
            registered._event_count = event_count


__all__ = [
    "BackendFactory",
    "CalendarAlreadyExistsError",
    "CalendarNotFoundError",
    "CalendarProviderRegistry",
    "CalendarReadOnlyError",
    "CalendarRegistryError",
    "RegisteredCalendar",
]
