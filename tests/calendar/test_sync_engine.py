"""Tests for calendar sync engine and providers.

Tests the sync infrastructure including:
- SyncEngine base functionality
- Provider registration and management
- ICS provider parsing
- CalDAV provider (mocked)
- Conflict resolution
- Event mapping
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from modules.calendar_store.sync_engine import (
    SyncEngine,
    CalendarSyncProvider,
    ExternalEvent,
    SyncConflict,
    SyncResult,
    SyncDirection,
    SyncStatus,
    ConflictResolution,
)
from modules.calendar_store.providers.ics_provider import ICSProvider


def _icalendar_available() -> bool:
    """Check if icalendar library is available."""
    try:
        import icalendar
        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_external_event() -> ExternalEvent:
    """Create a sample external event."""
    return ExternalEvent(
        external_id="ext-123",
        title="External Meeting",
        start_time=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 6, 15, 11, 0, tzinfo=timezone.utc),
        description="A meeting from external calendar",
        location="Room 101",
        is_all_day=False,
        recurrence_rule=None,
        attendees=[{"email": "user@example.com", "status": "ACCEPTED"}],
        reminders=[{"minutes_before": 15}],
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        updated_at=datetime(2024, 6, 10, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_ics_content() -> str:
    """Create sample ICS content."""
    return """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Calendar//EN
BEGIN:VEVENT
UID:event-001@example.com
DTSTART:20240615T100000Z
DTEND:20240615T110000Z
SUMMARY:Test Meeting
DESCRIPTION:A test meeting description
LOCATION:Conference Room A
CREATED:20240601T000000Z
LAST-MODIFIED:20240610T120000Z
BEGIN:VALARM
TRIGGER:-PT15M
ACTION:DISPLAY
DESCRIPTION:Reminder
END:VALARM
END:VEVENT
BEGIN:VEVENT
UID:event-002@example.com
DTSTART;VALUE=DATE:20240620
DTEND;VALUE=DATE:20240621
SUMMARY:All Day Event
END:VEVENT
BEGIN:VEVENT
UID:event-003@example.com
DTSTART:20240625T140000Z
DTEND:20240625T150000Z
SUMMARY:Recurring Meeting
RRULE:FREQ=WEEKLY;COUNT=4;BYDAY=TU
END:VEVENT
END:VCALENDAR"""


@pytest.fixture
def mock_repository() -> Mock:
    """Create a mock repository."""
    repo = Mock()
    repo.get_event_by_external_id = Mock(return_value=None)
    repo.create_event = Mock(return_value={"id": str(uuid4())})
    repo.update_event = Mock(return_value={"id": str(uuid4())})
    repo.get_sync_state = Mock(return_value=None)
    repo.update_sync_state = Mock()
    repo.list_categories = Mock(return_value=[
        {"id": str(uuid4()), "name": "Work", "slug": "work"},
        {"id": str(uuid4()), "name": "Personal", "slug": "personal"},
    ])
    return repo


class MockProvider(CalendarSyncProvider):
    """Mock provider for testing."""
    
    def __init__(self, events: list[ExternalEvent] | None = None):
        self._events = events or []
        self._connected = False
        self._config = {}
    
    @property
    def provider_type(self) -> str:
        return "mock"
    
    @property
    def display_name(self) -> str:
        return "Mock Provider"
    
    def connect(self, config: dict) -> bool:
        self._config = config
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        self._connected = False
    
    def list_calendars(self) -> list[dict]:
        return [{"id": "default", "name": "Mock Calendar"}]
    
    def fetch_events(
        self,
        calendar_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        sync_token: str | None = None,
    ) -> tuple[list[ExternalEvent], str | None]:
        return self._events, "token-123"


# -----------------------------------------------------------------------------
# SyncEngine Tests
# -----------------------------------------------------------------------------

class TestSyncEngine:
    """Tests for SyncEngine class."""
    
    def test_register_provider(self, mock_repository: Mock):
        """Test provider registration."""
        engine = SyncEngine(mock_repository)
        provider = MockProvider()
        
        engine.register_provider(provider)
        
        assert "mock" in engine._providers
        assert engine._providers["mock"] is provider
    
    def test_register_multiple_providers(self, mock_repository: Mock):
        """Test registering multiple providers."""
        engine = SyncEngine(mock_repository)
        
        provider1 = MockProvider()
        provider2 = ICSProvider()
        
        engine.register_provider(provider1)
        engine.register_provider(provider2)
        
        assert len(engine._providers) == 2
        assert "mock" in engine._providers
        assert "ics" in engine._providers
    
    def test_get_provider(self, mock_repository: Mock):
        """Test getting a registered provider."""
        engine = SyncEngine(mock_repository)
        provider = MockProvider()
        engine.register_provider(provider)
        
        result = engine.get_provider("mock")
        
        assert result is provider
    
    def test_get_nonexistent_provider(self, mock_repository: Mock):
        """Test getting a provider that doesn't exist."""
        engine = SyncEngine(mock_repository)
        
        result = engine.get_provider("nonexistent")
        
        assert result is None
    
    def test_list_providers(self, mock_repository: Mock):
        """Test listing registered providers."""
        engine = SyncEngine(mock_repository)
        provider1 = MockProvider()
        provider2 = ICSProvider()
        engine.register_provider(provider1)
        engine.register_provider(provider2)
        
        providers = engine.list_providers()
        
        assert len(providers) == 2
        types = [p["type"] for p in providers]
        assert "mock" in types
        assert "ics" in types
    
    def test_sync_with_provider_not_found(self, mock_repository: Mock):
        """Test sync fails gracefully for unknown provider."""
        engine = SyncEngine(mock_repository)
        
        result = engine.sync_calendar(
            provider_type="nonexistent",
            config={"account": "test"},
            calendar_id="default",
        )
        
        assert result.status == SyncStatus.FAILED
        assert len(result.errors) > 0
        assert "Unknown provider" in result.errors[0]
    
    def test_sync_with_connection_failure(self, mock_repository: Mock):
        """Test sync handles connection failure."""
        engine = SyncEngine(mock_repository)
        
        # Provider that fails to connect
        provider = MockProvider()
        provider.connect = Mock(return_value=False)
        engine.register_provider(provider)
        
        result = engine.sync_calendar(
            provider_type="mock",
            config={"account": "test"},
            calendar_id="default",
        )
        
        assert result.status == SyncStatus.FAILED
        assert any("connect" in e.lower() for e in result.errors)


class TestConflictResolution:
    """Tests for conflict resolution strategies."""
    
    def test_conflict_resolution_enum_values(self):
        """Test ConflictResolution enum has expected values."""
        assert ConflictResolution.ASK.value == "ask"
        assert ConflictResolution.LOCAL_WINS.value == "local_wins"
        assert ConflictResolution.REMOTE_WINS.value == "remote_wins"
        assert ConflictResolution.NEWEST_WINS.value == "newest_wins"
        assert ConflictResolution.MERGE.value == "merge"
    
    def test_sync_engine_default_conflict_resolution(self, mock_repository: Mock):
        """Test SyncEngine has default conflict resolution."""
        engine = SyncEngine(mock_repository)
        
        assert engine._conflict_resolution == ConflictResolution.REMOTE_WINS
    
    def test_sync_engine_custom_conflict_resolution(self, mock_repository: Mock):
        """Test SyncEngine accepts custom conflict resolution."""
        engine = SyncEngine(
            mock_repository,
            conflict_resolution=ConflictResolution.LOCAL_WINS,
        )
        
        assert engine._conflict_resolution == ConflictResolution.LOCAL_WINS


# -----------------------------------------------------------------------------
# ICS Provider Tests
# -----------------------------------------------------------------------------

class TestICSProvider:
    """Tests for ICS provider."""
    
    def test_provider_type(self):
        """Test provider type property."""
        provider = ICSProvider()
        assert provider.provider_type == "ics"
    
    def test_display_name(self):
        """Test display name property."""
        provider = ICSProvider()
        assert provider.display_name == "ICS File / URL"
    
    def test_connect_with_single_source(self):
        """Test connecting with a single source."""
        provider = ICSProvider()
        
        result = provider.connect({
            "path": "/path/to/calendar.ics",
            "name": "My Calendar",
        })
        
        assert result is True
        calendars = provider.list_calendars()
        assert len(calendars) == 1
        assert calendars[0]["name"] == "My Calendar"
    
    def test_connect_with_multiple_sources(self):
        """Test connecting with multiple sources."""
        provider = ICSProvider()
        
        result = provider.connect({
            "sources": [
                {"id": "work", "name": "Work", "path": "/work.ics"},
                {"id": "personal", "name": "Personal", "path": "/personal.ics"},
            ]
        })
        
        assert result is True
        calendars = provider.list_calendars()
        assert len(calendars) == 2
    
    def test_disconnect(self):
        """Test disconnecting clears state."""
        provider = ICSProvider()
        provider.connect({"path": "/test.ics"})
        
        provider.disconnect()
        
        assert provider.list_calendars() == []
    
    @pytest.mark.skipif(
        not _icalendar_available(),
        reason="icalendar library not installed"
    )
    def test_parse_ics_with_library(self, sample_ics_content: str):
        """Test parsing ICS with icalendar library."""
        provider = ICSProvider()
        provider.connect({"path": "dummy"})
        
        events = provider._parse_ics(sample_ics_content)
        
        assert len(events) == 3
        
        # Check first event
        event1 = events[0]
        assert event1.external_id == "event-001@example.com"
        assert event1.title == "Test Meeting"
        assert event1.location == "Conference Room A"
        assert event1.is_all_day is False
        
        # Check all-day event
        event2 = events[1]
        assert event2.external_id == "event-002@example.com"
        assert event2.is_all_day is True
        
        # Check recurring event
        event3 = events[2]
        assert event3.external_id == "event-003@example.com"
        assert event3.recurrence_rule is not None
    
    def test_parse_ics_simple_fallback(self, sample_ics_content: str):
        """Test simple ICS parsing fallback."""
        provider = ICSProvider()
        provider.connect({"path": "dummy"})
        
        # Force simple parser
        events = provider._simple_parse_ics(sample_ics_content)
        
        assert len(events) >= 1
        assert events[0].title == "Test Meeting"
    
    def test_fetch_events_with_date_filter(self, sample_ics_content: str):
        """Test fetching events with date filter."""
        provider = ICSProvider()
        provider.connect({"path": "dummy"})
        
        # Mock the data loading
        provider._load_ics_data = Mock(return_value=sample_ics_content)
        
        start = datetime(2024, 6, 14, tzinfo=timezone.utc)
        end = datetime(2024, 6, 16, tzinfo=timezone.utc)
        
        events, token = provider.fetch_events("default", start=start, end=end)
        
        # Should only get the first event (June 15)
        assert len(events) >= 1
        assert all(e.start_time >= start and e.start_time <= end for e in events)
    
    def test_read_local_file(self, tmp_path, sample_ics_content: str):
        """Test reading ICS from local file."""
        # Create temp file
        ics_file = tmp_path / "test.ics"
        ics_file.write_text(sample_ics_content)
        
        provider = ICSProvider()
        
        content = provider._read_local_ics(str(ics_file))
        
        assert "BEGIN:VCALENDAR" in content
        assert "Test Meeting" in content
    
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        provider = ICSProvider()
        
        with pytest.raises(FileNotFoundError):
            provider._read_local_ics("/nonexistent/path.ics")


# -----------------------------------------------------------------------------
# ExternalEvent Tests
# -----------------------------------------------------------------------------

class TestExternalEvent:
    """Tests for ExternalEvent dataclass."""
    
    def test_create_minimal_event(self):
        """Test creating event with minimal fields."""
        event = ExternalEvent(
            external_id="ext-1",
            title="Test",
            start_time=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        )
        
        assert event.external_id == "ext-1"
        assert event.title == "Test"
        assert event.description is None
        assert event.is_all_day is False
        assert event.timezone == "UTC"
    
    def test_create_all_day_event(self):
        """Test creating all-day event."""
        event = ExternalEvent(
            external_id="ext-2",
            title="Holiday",
            start_time=datetime(2024, 12, 25, tzinfo=timezone.utc),
            end_time=datetime(2024, 12, 26, tzinfo=timezone.utc),
            is_all_day=True,
        )
        
        assert event.is_all_day is True
    
    def test_create_event_with_all_fields(self, sample_external_event: ExternalEvent):
        """Test event with all fields populated."""
        event = sample_external_event
        
        assert event.external_id == "ext-123"
        assert event.description is not None
        assert event.location is not None
        assert len(event.attendees) > 0
        assert len(event.reminders) > 0
        assert event.created_at is not None
        assert event.updated_at is not None
    
    def test_event_raw_data_field(self):
        """Test raw_data field for provider-specific data."""
        event = ExternalEvent(
            external_id="ext-3",
            title="Test",
            start_time=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            raw_data={"google_event_id": "abc123", "hangout_link": "https://meet.google.com/xyz"},
        )
        
        assert event.raw_data["google_event_id"] == "abc123"
        assert "hangout_link" in event.raw_data


# -----------------------------------------------------------------------------
# SyncResult Tests
# -----------------------------------------------------------------------------

class TestSyncResult:
    """Tests for SyncResult dataclass."""
    
    def test_successful_result(self):
        """Test creating successful sync result."""
        result = SyncResult(
            status=SyncStatus.SUCCESS,
            source_type="ics",
            source_account="local",
            source_calendar="work",
            events_added=5,
            events_updated=3,
            events_deleted=1,
        )
        
        assert result.status == SyncStatus.SUCCESS
        assert result.events_added == 5
        assert result.events_updated == 3
        assert result.events_deleted == 1
        assert result.total_processed == 9  # 5 + 3 + 1
    
    def test_failed_result_with_errors(self):
        """Test creating failed sync result."""
        result = SyncResult(
            status=SyncStatus.FAILED,
            source_type="caldav",
            source_account="user@example.com",
            errors=["Connection failed", "Invalid credentials"],
        )
        
        assert result.status == SyncStatus.FAILED
        assert len(result.errors) == 2
        assert result.has_errors is True
    
    def test_result_with_conflicts(self):
        """Test sync result with conflicts."""
        conflict = SyncConflict(
            event_id="local-1",
            external_id="ext-1",
            local_event={"title": "Local Title"},
            remote_event={"title": "Remote Title"},
            conflict_type="modified",
        )
        
        result = SyncResult(
            status=SyncStatus.PARTIAL,
            source_type="caldav",
            source_account="user@example.com",
            events_added=2,
            events_updated=1,
            conflicts=[conflict],
        )
        
        assert result.status == SyncStatus.PARTIAL
        assert len(result.conflicts) == 1
        assert result.has_conflicts is True
    
    def test_result_timing(self):
        """Test sync result timing fields."""
        start_time = datetime.now(timezone.utc)
        
        result = SyncResult(
            status=SyncStatus.SUCCESS,
            source_type="ics",
            source_account="local",
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
        )
        
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at


# -----------------------------------------------------------------------------
# SyncConflict Tests
# -----------------------------------------------------------------------------

class TestSyncConflict:
    """Tests for SyncConflict dataclass."""
    
    def test_create_conflict(self):
        """Test creating a sync conflict."""
        conflict = SyncConflict(
            event_id="local-123",
            external_id="ext-456",
            local_event={"title": "Local Meeting", "updated_at": "2024-06-10"},
            remote_event={"title": "Remote Meeting", "updated_at": "2024-06-11"},
            conflict_type="modified",
        )
        
        assert conflict.event_id == "local-123"
        assert conflict.external_id == "ext-456"
        assert conflict.conflict_type == "modified"
        assert conflict.resolution is None
    
    def test_conflict_with_resolution(self):
        """Test conflict with resolution set."""
        conflict = SyncConflict(
            event_id="local-1",
            external_id="ext-1",
            local_event={},
            remote_event={},
            conflict_type="deleted_remote",
            resolution=ConflictResolution.LOCAL_WINS,
            resolved_event={"title": "Kept Local"},
        )
        
        assert conflict.resolution == ConflictResolution.LOCAL_WINS
        assert conflict.resolved_event is not None


# -----------------------------------------------------------------------------
# SyncStatus Tests
# -----------------------------------------------------------------------------

class TestSyncStatus:
    """Tests for SyncStatus enum."""
    
    def test_status_values(self):
        """Test all status values exist."""
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.IN_PROGRESS.value == "in_progress"
        assert SyncStatus.SUCCESS.value == "success"
        assert SyncStatus.PARTIAL.value == "partial"
        assert SyncStatus.FAILED.value == "failed"
        assert SyncStatus.CANCELLED.value == "cancelled"


# -----------------------------------------------------------------------------
# SyncDirection Tests
# -----------------------------------------------------------------------------

class TestSyncDirection:
    """Tests for SyncDirection enum."""
    
    def test_direction_values(self):
        """Test all direction values exist."""
        assert SyncDirection.IMPORT.value == "import"
        assert SyncDirection.EXPORT.value == "export"
        assert SyncDirection.BIDIRECTIONAL.value == "bidirectional"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestSyncIntegration:
    """Integration tests for full sync workflow."""
    
    def test_ics_provider_list_calendars_after_connect(self, sample_ics_content: str):
        """Test ICS provider returns calendars after connect."""
        provider = ICSProvider()
        
        provider.connect({
            "sources": [
                {"id": "work", "name": "Work Calendar", "path": "/work.ics"},
                {"id": "home", "name": "Home Calendar", "path": "/home.ics"},
            ]
        })
        
        calendars = provider.list_calendars()
        
        assert len(calendars) == 2
        names = [c["name"] for c in calendars]
        assert "Work Calendar" in names
        assert "Home Calendar" in names
    
    def test_ics_provider_fetch_returns_token(self, sample_ics_content: str):
        """Test ICS provider returns sync token."""
        provider = ICSProvider()
        provider.connect({"path": "/test.ics"})
        provider._load_ics_data = Mock(return_value=sample_ics_content)
        
        events, token = provider.fetch_events("default")
        
        assert token is not None
        assert len(token) > 0  # MD5 hash
    
    def test_engine_prevents_concurrent_sync(self, mock_repository: Mock):
        """Test engine prevents concurrent sync of same calendar."""
        engine = SyncEngine(mock_repository)
        provider = MockProvider()
        engine.register_provider(provider)
        
        # Start first sync (mark it as running)
        engine._running_syncs["mock:default"] = True
        
        # Try to start another sync
        result = engine.sync_calendar(
            provider_type="mock",
            config={"account": "test"},
            calendar_id="default",
        )
        
        assert result.status == SyncStatus.FAILED
        assert any("already in progress" in e.lower() for e in result.errors)
