"""Calendar configuration section for ATLAS ConfigManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, MutableMapping, Optional
from collections.abc import Mapping
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalendarConfigSection:
    """Manage calendar-related configuration defaults."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any

    def apply(self) -> None:
        """Populate the shared configuration dictionary with calendar defaults."""
        self._ensure_calendars_block()

    def _ensure_calendars_block(self) -> None:
        """Ensure the calendars configuration block exists with defaults."""
        calendars_block = self.config.get("calendars")
        if not isinstance(calendars_block, Mapping):
            calendars_block = {}
        else:
            calendars_block = dict(calendars_block)

        # Global calendar settings
        calendars_block.setdefault("default_calendar", None)
        calendars_block.setdefault("merge_duplicates", False)
        calendars_block.setdefault("parallel_queries", True)
        calendars_block.setdefault("query_timeout", 30.0)

        # Ensure sources dict exists
        sources = calendars_block.get("sources")
        if not isinstance(sources, Mapping):
            sources = {}
        else:
            sources = dict(sources)
        calendars_block["sources"] = sources

        # Apply environment overrides
        self._apply_env_overrides(calendars_block)

        self.config["calendars"] = calendars_block

    def _apply_env_overrides(self, calendars_block: MutableMapping[str, Any]) -> None:
        """Apply environment variable overrides to calendar config."""
        # Default calendar from environment
        env_default = self.env_config.get("ATLAS_DEFAULT_CALENDAR")
        if env_default:
            calendars_block["default_calendar"] = env_default

        # Google Calendar credentials path
        env_google_creds = self.env_config.get("GOOGLE_CALENDAR_CREDENTIALS")
        if env_google_creds:
            sources = calendars_block.get("sources", {})
            for name, config in sources.items():
                if isinstance(config, Mapping) and config.get("type") == "google":
                    if not config.get("credentials_path"):
                        sources[name] = dict(config)
                        sources[name]["credentials_path"] = env_google_creds

        # Outlook/Azure credentials
        env_azure_client = self.env_config.get("AZURE_CLIENT_ID")
        env_azure_tenant = self.env_config.get("AZURE_TENANT_ID")
        if env_azure_client or env_azure_tenant:
            sources = calendars_block.get("sources", {})
            for name, config in sources.items():
                if isinstance(config, Mapping) and config.get("type") == "outlook":
                    updated = dict(config)
                    if env_azure_client and not config.get("client_id"):
                        updated["client_id"] = env_azure_client
                    if env_azure_tenant and not config.get("tenant_id"):
                        updated["tenant_id"] = env_azure_tenant
                    sources[name] = updated

    def get_calendar_sources(self) -> List[dict]:
        """Return list of configured calendar sources."""
        calendars_block = self.config.get("calendars", {})
        sources = calendars_block.get("sources", {})
        result = []
        for name, config in sources.items():
            if isinstance(config, Mapping):
                cal = dict(config)
                cal["name"] = name
                result.append(cal)
        return result

    def get_default_calendar(self) -> Optional[str]:
        """Return the name of the default calendar."""
        calendars_block = self.config.get("calendars", {})
        return calendars_block.get("default_calendar")

    def add_calendar_source(
        self,
        name: str,
        backend_type: str,
        **kwargs: Any,
    ) -> None:
        """Add a new calendar source to configuration.

        Args:
            name: Unique identifier for the calendar.
            backend_type: Type of backend (ics, google, outlook, caldav, dbus).
            **kwargs: Additional configuration options.
        """
        calendars_block = self.config.get("calendars", {})
        sources = calendars_block.get("sources", {})

        source_config = {"type": backend_type}
        source_config.update(kwargs)
        sources[name] = source_config

        calendars_block["sources"] = sources
        self.config["calendars"] = calendars_block

    def remove_calendar_source(self, name: str) -> bool:
        """Remove a calendar source from configuration.

        Args:
            name: Name of the calendar to remove.

        Returns:
            True if removed, False if not found.
        """
        calendars_block = self.config.get("calendars", {})
        sources = calendars_block.get("sources", {})

        if name in sources:
            del sources[name]
            calendars_block["sources"] = sources
            self.config["calendars"] = calendars_block
            return True
        return False

    def set_default_calendar(self, name: Optional[str]) -> None:
        """Set the default calendar for write operations.

        Args:
            name: Name of the calendar, or None to clear.
        """
        calendars_block = self.config.get("calendars", {})
        calendars_block["default_calendar"] = name
        self.config["calendars"] = calendars_block


__all__ = ["CalendarConfigSection"]
