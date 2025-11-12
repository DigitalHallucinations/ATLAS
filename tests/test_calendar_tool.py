"""Unit tests for the Debian 12 calendar tool."""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pytest


_MODULE_PATH = Path("modules/Tools/Base_Tools/debian12_calendar.py")
# Ensure parent packages exist so dataclasses can resolve module lookups.
for package_name in ["modules", "modules.Tools", "modules.Tools.Base_Tools"]:
    if package_name not in sys.modules:
        module = types.ModuleType(package_name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package_name] = module

_SPEC = importlib.util.spec_from_file_location(
    "modules.Tools.Base_Tools.debian12_calendar", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_calendar_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _calendar_module
_SPEC.loader.exec_module(_calendar_module)

CalendarBackend = _calendar_module.CalendarBackend
CalendarBackendError = _calendar_module.CalendarBackendError
CalendarEvent = _calendar_module.CalendarEvent
Debian12CalendarTool = _calendar_module.Debian12CalendarTool
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
        self.called = {
            "list": 0,
            "detail": 0,
            "search": 0,
            "create": 0,
            "update": 0,
            "delete": 0,
        }

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

    async def create_event(self, payload: Any, calendar=None):
        self.called["create"] += 1
        event = CalendarEvent(
            id=str(payload.get("id", "new")),
            title=payload.get("title", "Untitled"),
            start=payload.get("start", _dt.datetime.now(tz=_dt.timezone.utc)),
            end=payload.get("end", _dt.datetime.now(tz=_dt.timezone.utc)),
            all_day=payload.get("all_day", False),
            description=payload.get("description"),
            location=payload.get("location"),
            calendar=calendar or "primary",
            attendees=payload.get("attendees", []),
            raw={},
        )
        self._events.append(event)
        return event

    async def update_event(self, event_id: str, payload: Any, calendar=None):
        self.called["update"] += 1
        for event in self._events:
            if event.id == event_id:
                if "title" in payload:
                    event.title = payload["title"]  # type: ignore[misc]
                if "description" in payload:
                    event.description = payload["description"]
                return event
        raise EventNotFoundError(event_id)

    async def delete_event(self, event_id: str, calendar=None):
        self.called["delete"] += 1
        for index, event in enumerate(self._events):
            if event.id == event_id:
                self._events.pop(index)
                return None
        raise EventNotFoundError(event_id)


class _StubDBusClient:
    def __init__(self) -> None:
        self.calls = {
            "create": [],
            "update": [],
            "delete": [],
        }
        self._store: dict[str, dict[str, Any]] = {}

    async def list_events(self, start: Any, end: Any, calendar: Any = None):
        return [dict(value) for value in self._store.values()]

    async def get_event(self, event_id: str, calendar: Any = None):
        try:
            return dict(self._store[event_id])
        except KeyError as exc:
            raise EventNotFoundError(event_id) from exc

    async def search_events(self, query: str, start: Any, end: Any, calendar: Any = None):
        if not query:
            return await self.list_events(start, end, calendar)
        needle = query.lower()
        results = []
        for record in self._store.values():
            title = str(record.get("title") or "").lower()
            description = str(record.get("description") or "").lower()
            if needle in title or needle in description:
                results.append(dict(record))
        return results

    async def create_event(self, payload: Any, calendar: Any = None):
        event_payload = dict(payload)
        event_id = str(event_payload.get("id") or uuid4())
        event_payload["id"] = event_id
        event_payload["calendar"] = calendar or event_payload.get("calendar") or "primary"
        self._store[event_id] = event_payload
        self.calls["create"].append((dict(payload), calendar))
        return dict(event_payload)

    async def update_event(self, event_id: str, payload: Any, calendar: Any = None):
        if event_id not in self._store:
            raise EventNotFoundError(event_id)
        update_payload = dict(payload)
        record = dict(self._store[event_id])
        record.update(update_payload)
        if calendar:
            record["calendar"] = calendar
        self._store[event_id] = record
        self.calls["update"].append((event_id, dict(payload), calendar))
        return dict(record)

    async def delete_event(self, event_id: str, calendar: Any = None):
        if event_id not in self._store:
            raise EventNotFoundError(event_id)
        self._store.pop(event_id)
        self.calls["delete"].append((event_id, calendar))
        return None


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
    tool = Debian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    result = asyncio.run(tool.list_events())

    assert backend.called["list"] == 1
    assert result[0]["id"] == "1"
    assert result[0]["title"] == "Standup"
    assert result[0]["calendar"] == "primary"


def test_list_events_handles_empty_backend_result() -> None:
    backend = _DummyBackend([])
    tool = Debian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    result = asyncio.run(tool.list_events())

    assert result == []


def test_get_event_detail_propagates_not_found() -> None:
    backend = _DummyBackend([_make_event("1")])
    tool = Debian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    with pytest.raises(EventNotFoundError):
        asyncio.run(tool.get_event_detail("missing"))


def test_search_events_wraps_backend_error() -> None:
    class _ErrorBackend(_DummyBackend):
        async def search_events(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise RuntimeError("boom")

    backend = _ErrorBackend([_make_event("1")])
    tool = Debian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    with pytest.raises(CalendarBackendError):
        asyncio.run(tool.search_events("test"))


def test_create_update_delete_event_round_trip(tmp_path: Path) -> None:
    calendar_path = tmp_path / "primary.ics"
    calendar_path.write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//ATLAS//EN\nEND:VCALENDAR\n",
        encoding="utf-8",
    )

    manager = _StubConfigManager(DEBIAN12_CALENDAR_PATHS=[str(calendar_path)])
    tool = Debian12CalendarTool(config_manager=manager)

    created = asyncio.run(
        tool.create_event(
            title="Planning",
            start="2024-03-01T10:00:00+00:00",
            end="2024-03-01T11:00:00+00:00",
            attendees=[{"email": "alice@example.com", "name": "Alice"}],
            location="HQ",
            description="Quarterly planning",
        )
    )

    assert created["title"] == "Planning"
    assert created["attendees"][0]["email"] == "alice@example.com"

    text = calendar_path.read_text(encoding="utf-8")
    assert "SUMMARY:Planning" in text
    assert "ATTENDEE;CN=Alice:mailto:alice@example.com" in text

    updated = asyncio.run(
        tool.update_event(
            event_id=created["id"],
            title="Planning (Updated)",
            description="Updated description",
        )
    )

    assert updated["title"] == "Planning (Updated)"
    updated_text = calendar_path.read_text(encoding="utf-8")
    assert "SUMMARY:Planning (Updated)" in updated_text
    assert "Updated description" in updated_text

    deleted = asyncio.run(tool.delete_event(created["id"]))

    assert deleted == {"status": "deleted", "event_id": created["id"]}
    final_text = calendar_path.read_text(encoding="utf-8")
    assert "Planning (Updated)" not in final_text
    assert final_text.strip().startswith("BEGIN:VCALENDAR")


def test_missing_calendar_path_is_materialized_on_write(tmp_path: Path) -> None:
    calendar_path = tmp_path / "primary.ics"
    manager = _StubConfigManager(DEBIAN12_CALENDAR_PATHS=[str(calendar_path)])
    tool = Debian12CalendarTool(config_manager=manager)

    assert not calendar_path.exists()

    listed = asyncio.run(tool.list_events())
    assert listed == []

    searched = asyncio.run(tool.search_events("anything"))
    assert searched == []

    created = asyncio.run(
        tool.create_event(
            title="Bootstrap",
            start="2024-04-01T09:00:00+00:00",
            end="2024-04-01T10:00:00+00:00",
        )
    )

    assert calendar_path.exists()
    text = calendar_path.read_text(encoding="utf-8")
    assert "SUMMARY:Bootstrap" in text

    updated = asyncio.run(
        tool.update_event(
            event_id=created["id"],
            title="Bootstrap (Updated)",
        )
    )

    assert updated["title"] == "Bootstrap (Updated)"
    updated_text = calendar_path.read_text(encoding="utf-8")
    assert "SUMMARY:Bootstrap (Updated)" in updated_text

    deleted = asyncio.run(tool.delete_event(created["id"]))

    assert deleted == {"status": "deleted", "event_id": created["id"]}
    final_text = calendar_path.read_text(encoding="utf-8")
    assert "Bootstrap (Updated)" not in final_text
    assert final_text.strip().startswith("BEGIN:VCALENDAR")


def test_update_event_missing_raises(tmp_path: Path) -> None:
    calendar_path = tmp_path / "primary.ics"
    calendar_path.write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//ATLAS//EN\nEND:VCALENDAR\n",
        encoding="utf-8",
    )

    manager = _StubConfigManager(DEBIAN12_CALENDAR_PATHS=[str(calendar_path)])
    tool = Debian12CalendarTool(config_manager=manager)

    with pytest.raises(EventNotFoundError):
        asyncio.run(tool.update_event("missing", title="Nope"))


def test_run_dispatch_supports_write_operations() -> None:
    backend = _DummyBackend([_make_event("1", "Existing")])

    async def _fake_create(payload: Any, calendar: Any = None):  # type: ignore[override]
        return CalendarEvent(
            id="new",
            title=payload["title"],
            start=payload["start"],
            end=payload["end"],
            all_day=False,
            calendar="primary",
            attendees=[],
            raw={},
        )

    async def _fake_update(event_id: str, payload: Any, calendar: Any = None):  # type: ignore[override]
        return CalendarEvent(
            id=event_id,
            title=payload.get("title", ""),
            start=payload.get("start", _dt.datetime.now(tz=_dt.timezone.utc)),
            end=payload.get("end", _dt.datetime.now(tz=_dt.timezone.utc)),
            all_day=payload.get("all_day", False),
            calendar="primary",
            attendees=[],
            raw={},
        )

    async def _fake_delete(event_id: str, calendar: Any = None):  # type: ignore[override]
        return None

    backend.create_event = _fake_create  # type: ignore[assignment]
    backend.update_event = _fake_update  # type: ignore[assignment]
    backend.delete_event = _fake_delete  # type: ignore[assignment]

    tool = Debian12CalendarTool(config_manager=_StubConfigManager(), backend=backend)

    created = asyncio.run(
        tool.run(
            "create",
            title="Async",
            start="2024-02-01T09:00:00+00:00",
            end="2024-02-01T10:00:00+00:00",
        )
    )
    assert created["title"] == "Async"

    updated = asyncio.run(
        tool.run(
            "update",
            event_id="new",
            title="Updated",
            start="2024-02-01T11:00:00+00:00",
            end="2024-02-01T12:00:00+00:00",
        )
    )
    assert updated["title"] == "Updated"

    deleted = asyncio.run(tool.run("delete", event_id="new"))
    assert deleted["status"] == "deleted"


def test_dbus_backend_round_trip_operations() -> None:
    client = _StubDBusClient()
    manager = _StubConfigManager(DEBIAN12_CALENDAR_BACKEND="dbus")
    tool = Debian12CalendarTool(config_manager=manager, dbus_client=client)

    created = asyncio.run(
        tool.create_event(
            title="Sync",
            start="2024-05-01T09:00:00+00:00",
            end="2024-05-01T09:30:00+00:00",
            description="Morning sync",
        )
    )

    assert client.calls["create"]
    create_payload, create_calendar = client.calls["create"][0]
    assert create_calendar is None
    assert create_payload["start"].endswith("+00:00")

    updated = asyncio.run(
        tool.update_event(
            created["id"],
            title="Sync (Updated)",
            description="Agenda updated",
        )
    )

    assert updated["title"] == "Sync (Updated)"
    update_event_id, update_payload, update_calendar = client.calls["update"][0]
    assert update_event_id == created["id"]
    assert "title" in update_payload

    listed = asyncio.run(tool.list_events())
    assert listed and listed[0]["title"] == "Sync (Updated)"

    deleted = asyncio.run(tool.delete_event(created["id"]))
    assert deleted == {"status": "deleted", "event_id": created["id"]}
    assert client.calls["delete"]
