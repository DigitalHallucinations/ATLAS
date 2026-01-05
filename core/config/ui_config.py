"""Utilities for persisting UI specific configuration options."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional


class UIConfig:
    """Encapsulates persistence helpers for UI specific preferences.

    The class is intentionally lightweight and expects the caller to provide the
    dictionaries (``config`` and ``yaml_config``) that mirror the state managed
    by :class:`ATLAS.config.config_manager.ConfigManager`.  Read and write
    operations are delegated to callbacks so the class can be exercised in
    isolation during tests.
    """

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        yaml_config: Dict[str, Any],
        read_config: Callable[[str, Any], Any],
        write_config: Callable[[], None],
    ) -> None:
        self._config = config
        self._yaml_config = yaml_config
        self._read_callback = read_config
        self._write_callback = write_config

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _write_yaml(self) -> None:
        """Invoke the configured persistence callback."""

        if self._write_callback is not None:
            self._write_callback()

    def _get(self, key: str, default: Any = None) -> Any:
        """Fetch a configuration value via the injected reader callback."""

        return self._read_callback(key, default)

    # ------------------------------------------------------------------
    # Terminal wrap helpers
    # ------------------------------------------------------------------

    def get_terminal_wrap_enabled(self) -> bool:
        """Return the persisted terminal wrap preference (defaults to ``True``)."""

        value = self._get("UI_TERMINAL_WRAP_ENABLED", None)

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False

        return True

    def set_terminal_wrap_enabled(self, enabled: bool) -> bool:
        """Persist the terminal wrap preference."""

        normalized = bool(enabled)
        self._yaml_config["UI_TERMINAL_WRAP_ENABLED"] = normalized
        self._config["UI_TERMINAL_WRAP_ENABLED"] = normalized
        self._write_yaml()
        return normalized

    # ------------------------------------------------------------------
    # Debug log helpers
    # ------------------------------------------------------------------

    def get_debug_log_level(self) -> Optional[Any]:
        """Return the configured debug log level (raw representation)."""

        value = self._get("UI_DEBUG_LOG_LEVEL", None)
        if isinstance(value, str):
            token = value.strip()
            return token or None
        if isinstance(value, int):
            return value
        return None

    def set_debug_log_level(self, level: Optional[Any]) -> Optional[Any]:
        """Persist the debug log level value."""

        if level is None:
            self._yaml_config.pop("UI_DEBUG_LOG_LEVEL", None)
            self._config.pop("UI_DEBUG_LOG_LEVEL", None)
            self._write_yaml()
            return None

        persisted: Any
        if isinstance(level, str):
            candidate = level.strip()
            if not candidate:
                return self.set_debug_log_level(None)
            mapped = getattr(logging, candidate.upper(), None)
            if isinstance(mapped, int):
                level_name = logging.getLevelName(mapped)
                persisted = level_name if isinstance(level_name, str) else mapped
            else:
                try:
                    numeric = int(candidate)
                except ValueError:
                    persisted = candidate
                else:
                    level_name = logging.getLevelName(numeric)
                    persisted = level_name if isinstance(level_name, str) else numeric
        else:
            try:
                numeric_level = int(level)
            except (TypeError, ValueError):
                return self.set_debug_log_level(None)
            level_name = logging.getLevelName(numeric_level)
            persisted = level_name if isinstance(level_name, str) else numeric_level

        self._yaml_config["UI_DEBUG_LOG_LEVEL"] = persisted
        self._config["UI_DEBUG_LOG_LEVEL"] = persisted
        self._write_yaml()
        return persisted

    def get_debug_log_max_lines(self, default: Optional[int] = None) -> Optional[int]:
        """Return the configured maximum number of debug log lines."""

        value = self._get("UI_DEBUG_LOG_MAX_LINES", default)
        if value is None:
            return default
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default
        return max(100, normalized)

    def set_debug_log_max_lines(self, max_lines: Optional[int]) -> Optional[int]:
        """Persist the debug log retention limit."""

        normalized: Optional[int]
        if max_lines is None:
            normalized = None
        else:
            try:
                normalized = int(max_lines)
            except (TypeError, ValueError):
                normalized = None

        if normalized is not None:
            normalized = max(100, normalized)
            self._yaml_config["UI_DEBUG_LOG_MAX_LINES"] = normalized
            self._config["UI_DEBUG_LOG_MAX_LINES"] = normalized
        else:
            self._yaml_config.pop("UI_DEBUG_LOG_MAX_LINES", None)
            self._config.pop("UI_DEBUG_LOG_MAX_LINES", None)

        self._write_yaml()
        return normalized

    def get_debug_log_initial_lines(self, default: Optional[int] = None) -> Optional[int]:
        """Return the configured initial slice size for debug log display."""

        value = self._get("UI_DEBUG_LOG_INITIAL_LINES", default)
        if value is None:
            return default
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, normalized)

    def get_debug_logger_names(self) -> List[str]:
        """Return the configured list of logger names mirrored in the UI."""

        value = self._get("UI_DEBUG_LOGGERS", None)
        if value is None:
            return []

        names: List[str] = []
        if isinstance(value, str):
            tokens = value.split(",")
        elif isinstance(value, Sequence):
            tokens = value
        else:
            tokens = []

        for entry in tokens:
            sanitized = str(entry).strip()
            if sanitized:
                names.append(sanitized)
        return names

    def set_debug_logger_names(self, logger_names: Optional[Sequence[str]]) -> List[str]:
        """Persist the list of logger names mirrored in the UI console."""

        normalized: List[str] = []
        if logger_names is not None:
            for entry in logger_names:
                sanitized = str(entry).strip()
                if sanitized:
                    normalized.append(sanitized)

        if normalized:
            self._yaml_config["UI_DEBUG_LOGGERS"] = list(normalized)
            self._config["UI_DEBUG_LOGGERS"] = list(normalized)
        else:
            self._yaml_config.pop("UI_DEBUG_LOGGERS", None)
            self._config.pop("UI_DEBUG_LOGGERS", None)

        self._write_yaml()
        return list(normalized)

    def get_debug_log_format(self) -> Optional[str]:
        """Return the configured debug log format string."""

        value = self._get("UI_DEBUG_LOG_FORMAT", None)
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

    def get_debug_log_file_name(self) -> Optional[str]:
        """Return the configured debug log file name, if set."""

        value = self._get("UI_DEBUG_LOG_FILE", None)
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

    def get_app_root(self) -> Optional[str]:
        """Expose the application root path used when resolving UI resources."""

        value = self._get("APP_ROOT", None)
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

