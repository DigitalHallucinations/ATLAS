"""Unit tests for the multi-calendar system.

Tests the CompositeCalendarBackend, CalendarProviderRegistry,
and individual backend implementations.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import calendar package components
try:
    from modules.Tools.Base_Tools.calendar import (
        CalendarBackend,
        CalendarBackendType,
        CalendarConfig,
        CalendarEvent,
        CalendarProviderRegistry,
        CalendarsGlobalConfig,
        CompositeCalendarBackend,
        CompositeCalendarError,
        EventNotFoundError,
        ICSCalendarBackend,
        NoWritableCalendarError,
        NullCalendarBackend,
        SyncMode,
        create_registry_with_defaults,
    )
    CALENDAR_PACKAGE_AVAILABLE = True
except ImportError:
    CALENDAR_PACKAGE_AVAILABLE = False
    # Define stub types to allow module to be parsed
    CalendarBackend = type("CalendarBackend", (), {})
    CalendarBackendType = type("CalendarBackendType", (), {})
    CalendarConfig = type("CalendarConfig", (), {})
    CalendarEvent = type("CalendarEvent", (), {})
    CalendarProviderRegistry = type("CalendarProviderRegistry", (), {})
    CalendarsGlobalConfig = type("CalendarsGlobalConfig", (), {})
    CompositeCalendarBackend = type("CompositeCalendarBackend", (), {})
    CompositeCalendarError = Exception
    EventNotFoundError = Exception
    ICSCalendarBackend = type("ICSCalendarBackend", (), {})
    NoWritableCalendarError = Exception
    NullCalendarBackend = type("NullCalendarBackend", (), {})
    SyncMode = type("SyncMode", (), {})
    create_registry_with_defaults = lambda *a, **k: None


pytestmark = pytest.mark.skipif(
    not CALENDAR_PACKAGE_AVAILABLE,
    reason="Calendar package not available",
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> CalendarEvent:
    """Create a sample calendar event."""
    return CalendarEvent(
        id="event-1",
        title="Team Meeting",
        start=_dt.datetime(2024, 1, 15, 10, 0, tzinfo=_dt.timezone.utc),
        end=_dt.datetime(2024, 1, 15, 11, 0, tzinfo=_dt.timezone.utc),
        all_day=False,
        location="Conference Room A",
        description="Weekly team sync",
        calendar="work",
    )


@pytest.fixture
def sample_events() -> List[CalendarEvent]:
    """Create a list of sample events."""
    return [
        CalendarEvent(
            id="evt-1",
            title="Morning Standup",
            start=_dt.datetime(2024, 1, 15, 9, 0, tzinfo=_dt.timezone.utc),
            end=_dt.datetime(2024, 1, 15, 9, 30, tzinfo=_dt.timezone.utc),
            all_day=False,
            calendar="work",
        ),
        CalendarEvent(
            id="evt-2",
            title="Dentist Appointment",
            start=_dt.datetime(2024, 1, 15, 14, 0, tzinfo=_dt.timezone.utc),
            end=_dt.datetime(2024, 1, 15, 15, 0, tzinfo=_dt.timezone.utc),
            all_day=False,
            calendar="personal",
        ),
        CalendarEvent(
            id="evt-3",
            title="Team Lunch",
            start=_dt.datetime(2024, 1, 15, 12, 0, tzinfo=_dt.timezone.utc),
            end=_dt.datetime(2024, 1, 15, 13, 0, tzinfo=_dt.timezone.utc),
            all_day=False,
            calendar="work",
        ),
    ]


class MockCalendarBackend(CalendarBackend):
    """Mock backend for testing."""

    def __init__(
        self,
        name: str = "mock",
        events: Optional[List[CalendarEvent]] = None,
        raise_on_write: bool = False,
    ):
        self.name = name
        self._events = list(events) if events else []
        self._raise_on_write = raise_on_write
        self.call_counts: Dict[str, int] = {
            "list_events": 0,
            "get_event": 0,
            "search_events": 0,
            "create_event": 0,
            "update_event": 0,
            "delete_event": 0,
        }

    async def list_events(
        self,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        self.call_counts["list_events"] += 1
        return [
            evt for evt in self._events
            if start <= evt.start <= end
        ]

    async def get_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        self.call_counts["get_event"] += 1
        for evt in self._events:
            if evt.id == event_id:
                return evt
        raise EventNotFoundError(f"Event {event_id} not found")

    async def search_events(
        self,
        query: str,
        start: _dt.datetime,
        end: _dt.datetime,
        calendar: Optional[str] = None,
    ) -> Sequence[CalendarEvent]:
        self.call_counts["search_events"] += 1
        query_lower = query.lower()
        return [
            evt for evt in self._events
            if query_lower in evt.title.lower()
            or (evt.description and query_lower in evt.description.lower())
        ]

    async def create_event(
        self,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        self.call_counts["create_event"] += 1
        if self._raise_on_write:
            raise RuntimeError("Write not allowed")
        event = CalendarEvent(
            id=str(payload.get("id", str(uuid4()))),
            title=str(payload.get("title", "")),
            start=payload.get("start", _dt.datetime.now(_dt.timezone.utc)),
            end=payload.get("end", _dt.datetime.now(_dt.timezone.utc)),
            all_day=bool(payload.get("all_day", False)),
            calendar=calendar or self.name,
        )
        self._events.append(event)
        return event

    async def update_event(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        calendar: Optional[str] = None,
    ) -> CalendarEvent:
        self.call_counts["update_event"] += 1
        if self._raise_on_write:
            raise RuntimeError("Write not allowed")
        for i, evt in enumerate(self._events):
            if evt.id == event_id:
                updated = CalendarEvent(
                    id=evt.id,
                    title=str(payload.get("title", evt.title)),
                    start=payload.get("start", evt.start),
                    end=payload.get("end", evt.end),
                    all_day=payload.get("all_day", evt.all_day),
                    location=payload.get("location", evt.location),
                    description=payload.get("description", evt.description),
                    calendar=evt.calendar,
                )
                self._events[i] = updated
                return updated
        raise EventNotFoundError(f"Event {event_id} not found")

    async def delete_event(
        self,
        event_id: str,
        calendar: Optional[str] = None,
    ) -> None:
        self.call_counts["delete_event"] += 1
        if self._raise_on_write:
            raise RuntimeError("Write not allowed")
        for i, evt in enumerate(self._events):
            if evt.id == event_id:
                del self._events[i]
                return
        raise EventNotFoundError(f"Event {event_id} not found")


# ---------------------------------------------------------------------------
# CalendarConfig Tests
# ---------------------------------------------------------------------------


class TestCalendarConfig:
    """Tests for CalendarConfig dataclass."""

    def test_from_dict_minimal(self):
        """Test creating config with minimal fields."""
        data = {
            "name": "test",
            "type": "ics",
            "path": "/path/to/calendar.ics",
        }
        config = CalendarConfig.from_dict("test", data)
        assert config.name == "test"
        assert config.backend_type == CalendarBackendType.ICS
        assert config.path == Path("/path/to/calendar.ics")
        assert config.write_enabled is True
        assert config.sync_mode == SyncMode.ON_DEMAND

    def test_from_dict_full(self):
        """Test creating config with all fields."""
        data = {
            "name": "work",
            "type": "google",
            "credentials_path": "~/.config/google.json",
            "calendar_id": "work@example.com",
            "write_enabled": False,
            "sync_mode": "realtime",
            "color": "#ff0000",
            "priority": 50,
            "timezone": "America/New_York",
        }
        config = CalendarConfig.from_dict("work", data)
        assert config.name == "work"
        assert config.backend_type == CalendarBackendType.GOOGLE
        assert config.write_enabled is False
        assert config.sync_mode == SyncMode.REALTIME
        assert config.color == "#ff0000"
        assert config.priority == 50

    def test_from_dict_unknown_type(self):
        """Test handling unknown backend type."""
        data = {"type": "unknown_backend"}
        with pytest.raises(ValueError, match="Unknown backend type"):
            CalendarConfig.from_dict("test", data)


# ---------------------------------------------------------------------------
# CalendarProviderRegistry Tests
# ---------------------------------------------------------------------------


class TestCalendarProviderRegistry:
    """Tests for CalendarProviderRegistry."""

    def test_register_provider(self):
        """Test registering a calendar provider."""
        registry = CalendarProviderRegistry()
        backend = MockCalendarBackend("test")
        config = CalendarConfig(
            name="test",
            backend_type=CalendarBackendType.ICS,
            write_enabled=True,
        )

        registry.register_provider("test", backend, config)

        assert "test" in registry.list_providers()
        assert registry.get_provider("test") is backend

    def test_register_duplicate_raises(self):
        """Test that registering duplicate name raises."""
        registry = CalendarProviderRegistry()
        backend = MockCalendarBackend("test")
        config = CalendarConfig(name="test", backend_type=CalendarBackendType.ICS)

        registry.register_provider("test", backend, config)

        from modules.Tools.Base_Tools.calendar import CalendarAlreadyExistsError
        with pytest.raises(CalendarAlreadyExistsError):
            registry.register_provider("test", backend, config)

    def test_unregister_provider(self):
        """Test unregistering a calendar provider."""
        registry = CalendarProviderRegistry()
        backend = MockCalendarBackend("test")
        config = CalendarConfig(name="test", backend_type=CalendarBackendType.ICS)

        registry.register_provider("test", backend, config)
        registry.unregister_provider("test")

        assert "test" not in registry.list_providers()

    def test_default_calendar(self):
        """Test default calendar selection."""
        registry = CalendarProviderRegistry()
        backend1 = MockCalendarBackend("cal1")
        backend2 = MockCalendarBackend("cal2")
        config1 = CalendarConfig(name="cal1", backend_type=CalendarBackendType.ICS)
        config2 = CalendarConfig(name="cal2", backend_type=CalendarBackendType.ICS)

        registry.register_provider("cal1", backend1, config1)
        registry.register_provider("cal2", backend2, config2)
        registry.set_default("cal2")

        assert registry.default_calendar_name == "cal2"

    def test_list_providers(self):
        """Test listing all providers."""
        registry = CalendarProviderRegistry()

        for i in range(3):
            backend = MockCalendarBackend(f"cal{i}")
            config = CalendarConfig(name=f"cal{i}", backend_type=CalendarBackendType.ICS)
            registry.register_provider(f"cal{i}", backend, config)

        providers = registry.list_providers()
        assert len(providers) == 3
        assert all(f"cal{i}" in providers for i in range(3))


# ---------------------------------------------------------------------------
# CompositeCalendarBackend Tests
# ---------------------------------------------------------------------------


class TestCompositeCalendarBackend:
    """Tests for CompositeCalendarBackend."""

    @pytest.fixture
    def setup_composite(self, sample_events):
        """Set up a composite backend with mock backends."""
        registry = CalendarProviderRegistry()

        # Create work backend with work events
        work_events = [e for e in sample_events if e.calendar == "work"]
        work_backend = MockCalendarBackend("work", work_events)
        work_config = CalendarConfig(
            name="work",
            backend_type=CalendarBackendType.GOOGLE,
            write_enabled=True,
        )
        registry.register_provider("work", work_backend, work_config)

        # Create personal backend with personal events
        personal_events = [e for e in sample_events if e.calendar == "personal"]
        personal_backend = MockCalendarBackend("personal", personal_events)
        personal_config = CalendarConfig(
            name="personal",
            backend_type=CalendarBackendType.ICS,
            write_enabled=True,
        )
        registry.register_provider("personal", personal_backend, personal_config)

        registry.set_default("personal")

        composite = CompositeCalendarBackend(registry)
        return composite, registry, work_backend, personal_backend

    @pytest.mark.asyncio
    async def test_list_events_all_calendars(self, setup_composite, sample_events):
        """Test listing events from all calendars."""
        composite, registry, work_backend, personal_backend = setup_composite

        start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        end = _dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc)

        events = await composite.list_events(start, end)

        # Should get events from both backends
        assert len(events) == 3
        assert work_backend.call_counts["list_events"] == 1
        assert personal_backend.call_counts["list_events"] == 1

    @pytest.mark.asyncio
    async def test_list_events_specific_calendar(self, setup_composite):
        """Test listing events from specific calendar."""
        composite, registry, work_backend, personal_backend = setup_composite

        start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        end = _dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc)

        events = await composite.list_events(start, end, calendar="work")

        # Should only query work backend
        assert work_backend.call_counts["list_events"] == 1
        assert personal_backend.call_counts["list_events"] == 0

    @pytest.mark.asyncio
    async def test_get_event_searches_all(self, setup_composite):
        """Test get_event searches all backends."""
        composite, registry, work_backend, personal_backend = setup_composite

        event = await composite.get_event("evt-2")  # Personal event

        assert event.id == "evt-2"
        assert event.calendar == "personal"

    @pytest.mark.asyncio
    async def test_get_event_not_found(self, setup_composite):
        """Test get_event raises for non-existent event."""
        composite, registry, work_backend, personal_backend = setup_composite

        with pytest.raises(EventNotFoundError):
            await composite.get_event("non-existent")

    @pytest.mark.asyncio
    async def test_create_event_routes_to_specified_calendar(self, setup_composite):
        """Test create routes to specified calendar."""
        composite, registry, work_backend, personal_backend = setup_composite

        payload = {
            "title": "New Meeting",
            "start": _dt.datetime(2024, 2, 1, 10, 0, tzinfo=_dt.timezone.utc),
            "end": _dt.datetime(2024, 2, 1, 11, 0, tzinfo=_dt.timezone.utc),
        }

        event = await composite.create_event(payload, calendar="work")

        assert event.calendar == "work"
        assert work_backend.call_counts["create_event"] == 1
        assert personal_backend.call_counts["create_event"] == 0

    @pytest.mark.asyncio
    async def test_create_event_uses_default_calendar(self, setup_composite):
        """Test create uses default calendar when none specified."""
        composite, registry, work_backend, personal_backend = setup_composite

        payload = {
            "title": "New Event",
            "start": _dt.datetime(2024, 2, 1, 10, 0, tzinfo=_dt.timezone.utc),
        }

        event = await composite.create_event(payload)

        # Should use personal (the default)
        assert personal_backend.call_counts["create_event"] == 1
        assert work_backend.call_counts["create_event"] == 0

    @pytest.mark.asyncio
    async def test_search_events_all_calendars(self, setup_composite):
        """Test search queries all calendars."""
        composite, registry, work_backend, personal_backend = setup_composite

        start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        end = _dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc)

        events = await composite.search_events("standup", start, end)

        assert work_backend.call_counts["search_events"] == 1
        assert personal_backend.call_counts["search_events"] == 1

    @pytest.mark.asyncio
    async def test_delete_event_routes_correctly(self, setup_composite):
        """Test delete routes to correct calendar."""
        composite, registry, work_backend, personal_backend = setup_composite

        await composite.delete_event("evt-1", calendar="work")

        assert work_backend.call_counts["delete_event"] == 1
        assert personal_backend.call_counts["delete_event"] == 0


# ---------------------------------------------------------------------------
# NullCalendarBackend Tests
# ---------------------------------------------------------------------------


class TestNullCalendarBackend:
    """Tests for NullCalendarBackend."""

    @pytest.mark.asyncio
    async def test_list_events_returns_empty(self):
        """Test list_events returns empty list."""
        backend = NullCalendarBackend()
        start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        end = _dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc)

        events = await backend.list_events(start, end)

        assert events == []

    @pytest.mark.asyncio
    async def test_get_event_raises(self):
        """Test get_event raises EventNotFoundError."""
        backend = NullCalendarBackend()

        with pytest.raises(EventNotFoundError):
            await backend.get_event("any-id")

    @pytest.mark.asyncio
    async def test_search_events_returns_empty(self):
        """Test search_events returns empty list."""
        backend = NullCalendarBackend()
        start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
        end = _dt.datetime(2024, 12, 31, tzinfo=_dt.timezone.utc)

        events = await backend.search_events("query", start, end)

        assert events == []


# ---------------------------------------------------------------------------
# CalendarEvent Tests
# ---------------------------------------------------------------------------


class TestCalendarEvent:
    """Tests for CalendarEvent dataclass."""

    def test_to_dict(self, sample_event):
        """Test serialization to dict."""
        result = sample_event.to_dict()

        assert result["id"] == "event-1"
        assert result["title"] == "Team Meeting"
        assert result["location"] == "Conference Room A"
        assert result["calendar"] == "work"
        assert "start" in result
        assert "end" in result

    def test_to_dict_iso_format(self, sample_event):
        """Test that datetime fields are ISO formatted."""
        result = sample_event.to_dict()

        # Should be ISO format strings
        assert isinstance(result["start"], str)
        assert isinstance(result["end"], str)
        assert "2024-01-15" in result["start"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestCalendarSystemIntegration:
    """Integration tests for the calendar system."""

    @pytest.mark.asyncio
    async def test_create_registry_with_defaults(self):
        """Test creating registry with default factories."""
        registry = create_registry_with_defaults()

        # Should have factories registered for known types
        assert registry is not None

    @pytest.mark.asyncio
    async def test_calendars_global_config_from_dict(self):
        """Test loading global config from dict."""
        config_dict = {
            "default_calendar": "personal",
            "sources": {
                "personal": {
                    "type": "ics",
                    "path": "~/.local/share/calendars/personal.ics",
                    "write_enabled": True,
                },
                "work": {
                    "type": "ics",
                    "path": "~/.local/share/calendars/work.ics",
                    "write_enabled": False,
                },
            },
        }

        global_config = CalendarsGlobalConfig.from_dict(config_dict)

        assert global_config.default_calendar == "personal"
        assert len(global_config.sources) == 2
        assert "personal" in global_config.sources
        assert "work" in global_config.sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
