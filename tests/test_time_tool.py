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

dynamic_pkg = sys.modules.setdefault("tests.dynamic", types.ModuleType("tests.dynamic"))
if not getattr(dynamic_pkg, "__path__", None):
    dynamic_pkg.__path__ = []

# Reload the target module to ensure it picks up the stubbed pytz implementation.
sys.modules.pop("modules.Tools.Base_Tools.time", None)
sys.modules.pop("tests.dynamic.time", None)
time_tool = importlib.import_module("modules.Tools.Base_Tools.time")

pytz = pytz_stub


@pytest.fixture(autouse=True)
def _reset_config_manager_cache(monkeypatch):
    monkeypatch.setattr(
        time_tool,
        "_CONFIG_MANAGER_CACHE",
        time_tool._CONFIG_MANAGER_NOT_LOADED,
        raising=False,
    )
    yield
    monkeypatch.setattr(
        time_tool,
        "_CONFIG_MANAGER_CACHE",
        time_tool._CONFIG_MANAGER_NOT_LOADED,
        raising=False,
    )


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


def test_get_configured_timezone_name_uses_config_manager(monkeypatch):
    class _Config:
        def __init__(self):
            self._data = {
                "TIME_TOOL_DEFAULT_TZ": "Europe/Berlin",
            }

        def get_config(self, key):
            return self._data.get(key)

    monkeypatch.setattr(time_tool, "_CONFIG_MANAGER_CACHE", _Config)

    assert time_tool._get_configured_timezone_name() == "Europe/Berlin"


def test_resolve_timezone_reads_nested_config(monkeypatch):
    class _Config:
        def get_config(self, key):
            if key in {"TIME_TOOL_DEFAULT_TZ", "TIME_TOOL_DEFAULT_TIMEZONE"}:
                return None
            if key == "tools":
                return {"time": {"default_timezone": "America/Los_Angeles"}}
            return None

    monkeypatch.setattr(time_tool, "_CONFIG_MANAGER_CACHE", _Config)
    monkeypatch.setattr(time_tool, "_get_system_timezone_name", lambda: None)

    tz = time_tool._resolve_timezone(None)

    assert getattr(tz, "zone", None) == "America/Los_Angeles"
