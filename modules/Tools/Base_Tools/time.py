"""Utilities for exposing the current time in multiple formats.

The tool now resolves its timezone dynamically using (in order of priority):

1. The ``timezone`` argument provided by the caller.
2. The ``TIME_TOOL_DEFAULT_TZ``/``tools.time.default_timezone`` configuration
   knobs managed by :class:`ATLAS.config.ConfigManager`.
3. The system local timezone when it can be resolved safely.
4. ``UTC`` as a final fallback.
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import Mapping
from typing import Optional

import pytz


_DEFAULT_TZ_NAME = "UTC"
_CONFIG_TZ_KEYS = ("TIME_TOOL_DEFAULT_TZ", "TIME_TOOL_DEFAULT_TIMEZONE")
_CONFIG_MANAGER_NOT_LOADED = object()
_CONFIG_MANAGER_CACHE: object = _CONFIG_MANAGER_NOT_LOADED


def _load_config_manager() -> Optional[type[object]]:
    """Load and cache :class:`ConfigManager` lazily."""

    global _CONFIG_MANAGER_CACHE

    if _CONFIG_MANAGER_CACHE is _CONFIG_MANAGER_NOT_LOADED:
        try:  # ConfigManager is optional in certain test contexts
            from core.config import ConfigManager as _ConfigManager
        except Exception:  # pragma: no cover - exercised when configuration is unavailable
            _CONFIG_MANAGER_CACHE = None
        else:
            _CONFIG_MANAGER_CACHE = _ConfigManager

    return _CONFIG_MANAGER_CACHE if isinstance(_CONFIG_MANAGER_CACHE, type) else None


def _get_configured_timezone_name() -> Optional[str]:
    """Return a configured default timezone identifier if available."""

    config_manager_cls = _load_config_manager()

    if config_manager_cls is None:
        return None

    try:
        manager = config_manager_cls()  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - defensive guard around config bootstrap
        return None

    for key in _CONFIG_TZ_KEYS:
        value = manager.get_config(key)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate

    tools_block = manager.get_config("tools")
    if isinstance(tools_block, Mapping):
        time_block = tools_block.get("time")
        if isinstance(time_block, Mapping):
            value = time_block.get("default_timezone") or time_block.get("timezone")
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate

    return None


def _get_system_timezone_name() -> Optional[str]:
    """Attempt to derive the system's local timezone name."""

    try:
        tzinfo = datetime.datetime.now().astimezone().tzinfo
    except Exception:  # pragma: no cover - platform specific edge cases
        tzinfo = None

    if tzinfo is None:
        return None

    name = getattr(tzinfo, "key", None) or getattr(tzinfo, "zone", None)
    if isinstance(name, str) and name in pytz.all_timezones_set:
        return name

    tzname = tzinfo.tzname(None) if callable(getattr(tzinfo, "tzname", None)) else getattr(tzinfo, "tzname", None)
    if isinstance(tzname, str) and tzname in pytz.all_timezones_set:
        return tzname

    return None


def _coerce_timezone(name: Optional[str]):
    """Return a pytz timezone for *name* when possible."""

    if not isinstance(name, str):
        return None

    candidate = name.strip()
    if not candidate:
        return None

    try:
        return pytz.timezone(candidate)
    except (pytz.UnknownTimeZoneError, AttributeError):
        return None


def _resolve_timezone(timezone: Optional[str]):
    """Resolve the appropriate timezone object for the tool."""

    for candidate in (
        timezone,
        _get_configured_timezone_name(),
        _get_system_timezone_name(),
        _DEFAULT_TZ_NAME,
    ):
        tz = _coerce_timezone(candidate)
        if tz is not None:
            return tz

    return pytz.utc


async def get_current_info(format_type: str = 'timestamp', timezone: Optional[str] = None) -> str:
    """Return formatted current date/time information.

    Parameters
    ----------
    format_type:
        A string that specifies the type of information to return. Options are
        ``time``, ``date``, ``day``, ``month_year``, ``timestamp``. Defaults to
        ``timestamp``.
    timezone:
        Optional IANA/Olson timezone identifier (e.g., ``"America/New_York"``).
        When omitted the tool uses the configured default timezone, falls back
        to the system timezone when available, and ultimately to ``UTC``.

    Returns
    -------
    str
        The formatted time information based on ``format_type``.
    """

    await asyncio.sleep(0)

    tzinfo = _resolve_timezone(timezone)
    now = datetime.datetime.now(tzinfo)

    format_options = {
        'time': "%H:%M:%S",
        'date': "%Y-%m-%d",
        'day': "%A",
        'month_year': "%B %Y",
        'timestamp': "%Y-%m-%d %H:%M:%S"
    }

    format_string = format_options.get(format_type, "%Y-%m-%d %H:%M:%S")
    return now.strftime(format_string)
