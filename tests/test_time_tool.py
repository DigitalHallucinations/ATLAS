"""Unit tests for the time base tool."""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib
import sys
import types

import pytest


class _StubTimezone(_dt.tzinfo):
    def __init__(self, name: str):
        self.zone = name

    def utcoffset(self, dt):  # type: ignore[override]
        return _dt.timedelta(0)

    def dst(self, dt):  # type: ignore[override]
        return _dt.timedelta(0)

    def tzname(self, dt):  # type: ignore[override]
        return self.zone


class _UnknownTimeZoneError(Exception):
    """Stub replacement for pytz.UnknownTimeZoneError."""


_VALID_TIMEZONES = {
    "UTC",
    "Asia/Tokyo",
    "Europe/Berlin",
    "America/Los_Angeles",
}


pytz_stub = types.SimpleNamespace()


def _timezone(name: str):
    if name not in _VALID_TIMEZONES:
        raise _UnknownTimeZoneError(name)
    if name == "UTC":
        return pytz_stub.utc
    return _StubTimezone(name)


pytz_stub.timezone = _timezone
pytz_stub.UnknownTimeZoneError = _UnknownTimeZoneError
pytz_stub.all_timezones_set = set(_VALID_TIMEZONES)
pytz_stub.utc = _StubTimezone("UTC")

sys.modules["pytz"] = pytz_stub

sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}, dump=lambda *_args, **_kwargs: None),
)
sys.modules.setdefault(
    "dotenv",
    types.SimpleNamespace(
        load_dotenv=lambda *_args, **_kwargs: None,
        set_key=lambda *_args, **_kwargs: None,
        find_dotenv=lambda *_args, **_kwargs: "",
    ),
)

# Reload the target module to ensure it picks up the stubbed pytz implementation.
time_tool = importlib.import_module("modules.Tools.Base_Tools.time")
importlib.reload(time_tool)

pytz = pytz_stub


def test_resolve_timezone_prefers_user_override(monkeypatch):
    monkeypatch.setattr(time_tool, "_get_configured_timezone_name", lambda: "Europe/Berlin")
    monkeypatch.setattr(time_tool, "_get_system_timezone_name", lambda: "America/Los_Angeles")

    tz = time_tool._resolve_timezone("Asia/Tokyo")

    assert getattr(tz, "zone", None) == "Asia/Tokyo"


def test_resolve_timezone_uses_config_when_no_override(monkeypatch):
    monkeypatch.setattr(time_tool, "_get_configured_timezone_name", lambda: "Europe/Berlin")
    monkeypatch.setattr(time_tool, "_get_system_timezone_name", lambda: "America/Los_Angeles")

    tz = time_tool._resolve_timezone(None)

    assert getattr(tz, "zone", None) == "Europe/Berlin"


def test_resolve_timezone_defaults_to_utc(monkeypatch):
    monkeypatch.setattr(time_tool, "_get_configured_timezone_name", lambda: "Invalid/Timezone")
    monkeypatch.setattr(time_tool, "_get_system_timezone_name", lambda: "Also/Invalid")

    tz = time_tool._resolve_timezone(None)

    assert tz is pytz.utc


def test_get_current_info_passes_through_timezone(monkeypatch):
    captured = {}

    def _capture(timezone):
        captured["value"] = timezone
        return pytz.utc

    monkeypatch.setattr(time_tool, "_resolve_timezone", _capture)

    asyncio.run(time_tool.get_current_info(timezone="Asia/Tokyo"))

    assert captured["value"] == "Asia/Tokyo"
