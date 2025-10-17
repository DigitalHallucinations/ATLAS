"""Unit tests for the Devian 12 calendar tool."""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import pytest


_MODULE_PATH = Path("modules/Tools/Base_Tools/deviant12_calendar.py")
# Ensure parent packages exist so dataclasses can resolve module lookups.
for package_name in ["modules", "modules.Tools", "modules.Tools.Base_Tools"]:
    if package_name not in sys.modules:
        module = types.ModuleType(package_name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package_name] = module

_SPEC = importlib.util.spec_from_file_location(
    "modules.Tools.Base_Tools.deviant12_calendar", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_calendar_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _calendar_module
_SPEC.loader.exec_module(_calendar_module)

CalendarBackend = _calendar_module.CalendarBackend
CalendarBackendError = _calendar_module.CalendarBackendError
CalendarEvent = _calendar_module.CalendarEvent
Devian12CalendarTool = _calendar_module.Devian12CalendarTool
EventNotFoundError = _calendar_module.EventNotFoundError


class _StubConfigManager:
    """Minimal ConfigManager replacement for the tests."""

    UNSET = object()

    def __init__(self, **overrides: Any) -> None:
        self._overrides = overrides

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._overrides.get(key, default)


class _DummyBackend(CalendarBackend):
    def __init__(self, events: Sequence[CalendarEvent]):
        self._events = list(events)
        self.called = {"list": 0, "detail": 0, "search": 0}

    async def list_events(self, start: _dt.datetime, end: _dt.datetime, calendar=None):
        self.called["list"] += 1
        return list(self._events)

    async def get_event(self, event_id: str, calendar=None):
        self.called["detail"] += 1
        for event in self._events:
            if event.id == event_id:
                return event
        raise EventNotFoundError(event_id)

    async def search_events(
        self, query: str, start: _dt.datetime, end: _dt.datetime, calendar=None
    ):
        self.called["search"] += 1
        if not query:
            return list(self._events)
        lower = query.lower()
        return [
            event
            for event in self._events
            if lower in event.title.lower()
            or (event.description or "").lower().find(lower) >= 0
            or (event.location or "").lower().find(lower) >= 0
        ]


def _make_event(event_id: str, title: str = "Demo") -> CalendarEvent:
    start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
    end = start + _dt.timedelta(hours=1)
    return CalendarEvent(
        id=event_id,
        title=title,
        start=start,
        end=end,
        all_day=False,
        description="Description for " + title,
        location="HQ",
        calendar="primary",
        raw={"UID": event_id},
    )


def test_list_events_returns_normalized_payload() -> None:
    backend = _DummyBackend([_make_event("1", "Standup")])
    tool = Devian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    result = asyncio.run(tool.list_events())

    assert backend.called["list"] == 1
    assert result[0]["id"] == "1"
    assert result[0]["title"] == "Standup"
    assert result[0]["calendar"] == "primary"


def test_list_events_handles_empty_backend_result() -> None:
    backend = _DummyBackend([])
    tool = Devian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    result = asyncio.run(tool.list_events())

    assert result == []


def test_get_event_detail_propagates_not_found() -> None:
    backend = _DummyBackend([_make_event("1")])
    tool = Devian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    with pytest.raises(EventNotFoundError):
        asyncio.run(tool.get_event_detail("missing"))


def test_search_events_wraps_backend_error() -> None:
    class _ErrorBackend(_DummyBackend):
        async def search_events(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise RuntimeError("boom")

    backend = _ErrorBackend([_make_event("1")])
    tool = Devian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    with pytest.raises(CalendarBackendError):
        asyncio.run(tool.search_events("test"))

