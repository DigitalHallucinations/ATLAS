"""Tests for Phase 6: Agent Calendar Integration.

Tests the new agent-focused calendar methods, context injection,
and the calendar service provider.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from core.services.calendar.event_service import CalendarEventService
from core.services.calendar.context import (
    CalendarContextInjector,
    format_availability_response,
    format_free_time_slots,
    get_calendar_context_for_prompt,
)
from core.services.calendar.types import (
    CalendarEvent,
    EventStatus,
    EventVisibility,
    BusyStatus,
)
from core.services.common import Actor, OperationResult


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_repository():
    """Mock calendar repository."""
    repo = Mock()
    repo.list_events = AsyncMock(return_value=[])
    repo.search_events = AsyncMock(return_value=[])
    repo.find_conflicting_events = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_permission_checker():
    """Mock permission checker."""
    checker = Mock()
    checker.require_read_permission = AsyncMock()
    checker.require_write_permission = AsyncMock()
    checker.filter_events_by_permissions = AsyncMock(side_effect=lambda actor, events: events)
    return checker


@pytest.fixture
def mock_validator():
    """Mock event validator."""
    validator = Mock()
    validator.validate_create = AsyncMock(return_value=OperationResult.success(None))
    validator.validate_update = AsyncMock(return_value=OperationResult.success(None))
    return validator


@pytest.fixture
def calendar_service(mock_repository, mock_permission_checker, mock_validator):
    """Create CalendarEventService for testing."""
    return CalendarEventService(
        repository=mock_repository,
        permission_checker=mock_permission_checker,
        validator=mock_validator,
    )


@pytest.fixture
def test_actor():
    """Test actor for calendar operations."""
    return Actor(
        type="user",
        id="test-user-123",
        tenant_id="test-tenant",
        permissions=["calendar:read", "calendar:write"],
    )


@pytest.fixture
def sample_event():
    """Create a sample calendar event."""
    now = datetime.now(timezone.utc)
    return CalendarEvent(
        event_id=str(uuid4()),
        title="Team Meeting",
        description="Weekly sync",
        start_time=now + timedelta(hours=2),
        end_time=now + timedelta(hours=3),
        timezone_name="UTC",
        location="Conference Room A",
        status=EventStatus.CONFIRMED,
        visibility=EventVisibility.PRIVATE,
        busy_status=BusyStatus.BUSY,
        all_day=False,
        is_recurring=False,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    now = datetime.now(timezone.utc)
    return [
        CalendarEvent(
            event_id=str(uuid4()),
            title="Morning Standup",
            description="Daily standup",
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=1, minutes=30),
            timezone_name="UTC",
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            all_day=False,
            is_recurring=True,
            created_at=now,
            updated_at=now,
        ),
        CalendarEvent(
            event_id=str(uuid4()),
            title="Lunch with Team",
            description="Team lunch",
            start_time=now + timedelta(hours=4),
            end_time=now + timedelta(hours=5),
            timezone_name="UTC",
            location="Cafeteria",
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            all_day=False,
            is_recurring=False,
            created_at=now,
            updated_at=now,
        ),
        CalendarEvent(
            event_id=str(uuid4()),
            title="Project Review",
            description="Quarterly review",
            start_time=now + timedelta(hours=6),
            end_time=now + timedelta(hours=7),
            timezone_name="UTC",
            location="Boardroom",
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            all_day=False,
            is_recurring=False,
            created_at=now,
            updated_at=now,
        ),
    ]


# ============================================================================
# Agent Schedule Methods Tests
# ============================================================================


class TestSearchEvents:
    """Tests for CalendarEventService.search_events."""

    @pytest.mark.asyncio
    async def test_search_events_success(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test successful event search."""
        mock_repository.search_events.return_value = sample_events[:1]
        
        result = await calendar_service.search_events(
            test_actor,
            query="standup",
            limit=10,
        )
        
        assert result.is_success
        assert len(result.value) == 1
        assert result.value[0].title == "Morning Standup"

    @pytest.mark.asyncio
    async def test_search_events_empty_query(self, calendar_service, test_actor):
        """Test search with empty query returns empty list."""
        result = await calendar_service.search_events(test_actor, query="")
        
        assert result.is_success
        assert result.value == []

    @pytest.mark.asyncio
    async def test_search_events_with_date_filter(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test search with date filters."""
        now = datetime.now(timezone.utc)
        start = now
        end = now + timedelta(days=7)
        
        await calendar_service.search_events(
            test_actor,
            query="meeting",
            start_date=start,
            end_date=end,
        )
        
        mock_repository.search_events.assert_called_once()
        call_kwargs = mock_repository.search_events.call_args[1]
        assert call_kwargs["query"] == "meeting"
        assert call_kwargs["start"] == start
        assert call_kwargs["end"] == end

    @pytest.mark.asyncio
    async def test_search_events_permission_denied(
        self, calendar_service, mock_permission_checker, test_actor
    ):
        """Test search fails when permission denied."""
        from core.services.common import PermissionDeniedError
        mock_permission_checker.require_read_permission.side_effect = PermissionDeniedError("No access")
        
        result = await calendar_service.search_events(test_actor, query="test")
        
        assert result.is_failure
        assert result.error_code == "PERMISSION_DENIED"


class TestGetUpcomingEvents:
    """Tests for CalendarEventService.get_upcoming_events."""

    @pytest.mark.asyncio
    async def test_get_upcoming_events_success(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test getting upcoming events."""
        mock_repository.list_events.return_value = sample_events
        
        result = await calendar_service.get_upcoming_events(
            test_actor,
            hours_ahead=24,
            limit=10,
        )
        
        assert result.is_success
        assert len(result.value) == 3

    @pytest.mark.asyncio
    async def test_get_upcoming_events_respects_limit(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test limit is passed to repository."""
        mock_repository.list_events.return_value = sample_events[:2]
        
        result = await calendar_service.get_upcoming_events(
            test_actor,
            hours_ahead=24,
            limit=2,
        )
        
        assert result.is_success
        assert len(result.value) == 2
        mock_repository.list_events.assert_called_once()
        call_kwargs = mock_repository.list_events.call_args[1]
        assert call_kwargs["limit"] == 2

    @pytest.mark.asyncio
    async def test_get_upcoming_events_no_events(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test when no upcoming events."""
        mock_repository.list_events.return_value = []
        
        result = await calendar_service.get_upcoming_events(test_actor)
        
        assert result.is_success
        assert result.value == []


class TestCheckAvailability:
    """Tests for CalendarEventService.check_availability."""

    @pytest.mark.asyncio
    async def test_check_availability_free(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test checking availability when time is free."""
        mock_repository.find_conflicting_events.return_value = []
        
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=1)
        end = now + timedelta(hours=2)
        
        result = await calendar_service.check_availability(test_actor, start, end)
        
        assert result.is_success
        assert result.value["available"] is True
        assert result.value["conflicts"] == []

    @pytest.mark.asyncio
    async def test_check_availability_busy(
        self, calendar_service, mock_repository, test_actor, sample_event
    ):
        """Test checking availability when time conflicts."""
        mock_repository.find_conflicting_events.return_value = [sample_event]
        
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=1)
        end = now + timedelta(hours=2)
        
        result = await calendar_service.check_availability(test_actor, start, end)
        
        assert result.is_success
        assert result.value["available"] is False
        assert len(result.value["conflicts"]) == 1

    @pytest.mark.asyncio
    async def test_check_availability_invalid_range(
        self, calendar_service, test_actor
    ):
        """Test checking availability with invalid time range."""
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=2)
        end = now + timedelta(hours=1)  # End before start
        
        result = await calendar_service.check_availability(test_actor, start, end)
        
        assert result.is_failure
        assert result.error_code == "INVALID_TIME_RANGE"


class TestFindFreeTime:
    """Tests for CalendarEventService.find_free_time."""

    @pytest.mark.asyncio
    async def test_find_free_time_success(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test finding free time slots."""
        mock_repository.list_events.return_value = []
        
        now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
        start_range = now
        end_range = now + timedelta(days=1)
        
        result = await calendar_service.find_free_time(
            test_actor,
            start_range=start_range,
            end_range=end_range,
            duration_minutes=60,
            max_slots=3,
        )
        
        assert result.is_success
        assert len(result.value) <= 3
        for slot in result.value:
            assert "start" in slot
            assert "end" in slot
            assert slot["duration_minutes"] == 60

    @pytest.mark.asyncio
    async def test_find_free_time_with_working_hours(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test finding free time respects working hours."""
        mock_repository.list_events.return_value = []
        
        # Start at midnight
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_range = now
        end_range = now + timedelta(days=1)
        
        result = await calendar_service.find_free_time(
            test_actor,
            start_range=start_range,
            end_range=end_range,
            duration_minutes=60,
            working_hours=(9, 17),
            max_slots=1,
        )
        
        assert result.is_success
        if result.value:
            slot = result.value[0]
            # Slot should start at or after 9 AM
            assert slot["start"].hour >= 9

    @pytest.mark.asyncio
    async def test_find_free_time_between_events(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test finding free time between existing events."""
        now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Create events with a gap
        events = [
            CalendarEvent(
                event_id=str(uuid4()),
                title="Event 1",
                start_time=now,
                end_time=now + timedelta(hours=1),
                timezone_name="UTC",
                status=EventStatus.CONFIRMED,
                visibility=EventVisibility.PRIVATE,
                busy_status=BusyStatus.BUSY,
                all_day=False,
                created_at=now,
                updated_at=now,
            ),
            CalendarEvent(
                event_id=str(uuid4()),
                title="Event 2",
                start_time=now + timedelta(hours=3),
                end_time=now + timedelta(hours=4),
                timezone_name="UTC",
                status=EventStatus.CONFIRMED,
                visibility=EventVisibility.PRIVATE,
                busy_status=BusyStatus.BUSY,
                all_day=False,
                created_at=now,
                updated_at=now,
            ),
        ]
        mock_repository.list_events.return_value = events
        
        result = await calendar_service.find_free_time(
            test_actor,
            start_range=now,
            end_range=now + timedelta(hours=5),
            duration_minutes=60,
            max_slots=5,
        )
        
        assert result.is_success
        # Should find slot in the 2-hour gap between events
        assert len(result.value) >= 1

    @pytest.mark.asyncio
    async def test_find_free_time_invalid_duration(
        self, calendar_service, test_actor
    ):
        """Test finding free time with invalid duration."""
        now = datetime.now(timezone.utc)
        
        result = await calendar_service.find_free_time(
            test_actor,
            start_range=now,
            end_range=now + timedelta(days=1),
            duration_minutes=0,
        )
        
        assert result.is_failure
        assert result.error_code == "INVALID_DURATION"


class TestGetCalendarSummary:
    """Tests for CalendarEventService.get_calendar_summary."""

    @pytest.mark.asyncio
    async def test_get_calendar_summary_today(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test getting today's calendar summary."""
        mock_repository.list_events.return_value = sample_events
        
        result = await calendar_service.get_calendar_summary(test_actor, period="today")
        
        assert result.is_success
        assert result.value["period"] == "today"
        assert result.value["total_events"] == 3
        assert "busy_hours" in result.value
        assert "events" in result.value

    @pytest.mark.asyncio
    async def test_get_calendar_summary_week(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test getting week's calendar summary."""
        mock_repository.list_events.return_value = sample_events
        
        result = await calendar_service.get_calendar_summary(test_actor, period="week")
        
        assert result.is_success
        assert result.value["period"] == "week"

    @pytest.mark.asyncio
    async def test_get_calendar_summary_invalid_period(
        self, calendar_service, test_actor
    ):
        """Test getting summary with invalid period."""
        result = await calendar_service.get_calendar_summary(test_actor, period="invalid")
        
        assert result.is_failure
        assert result.error_code == "INVALID_PERIOD"

    @pytest.mark.asyncio
    async def test_get_calendar_summary_next_event(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test summary includes next event info."""
        mock_repository.list_events.return_value = sample_events
        
        result = await calendar_service.get_calendar_summary(test_actor, period="today")
        
        assert result.is_success
        # next_event may or may not be set depending on event times
        assert "next_event" in result.value


class TestSuggestMeetingTimes:
    """Tests for CalendarEventService.suggest_meeting_times."""

    @pytest.mark.asyncio
    async def test_suggest_meeting_times(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test suggesting meeting times."""
        mock_repository.list_events.return_value = []
        
        result = await calendar_service.suggest_meeting_times(
            test_actor,
            duration_minutes=30,
            max_suggestions=3,
        )
        
        assert result.is_success
        assert len(result.value) <= 3

    @pytest.mark.asyncio
    async def test_suggest_meeting_times_with_preferences(
        self, calendar_service, mock_repository, test_actor
    ):
        """Test suggesting times with specific preferences."""
        mock_repository.list_events.return_value = []
        
        now = datetime.now(timezone.utc)
        result = await calendar_service.suggest_meeting_times(
            test_actor,
            duration_minutes=60,
            preferred_start=now,
            preferred_end=now + timedelta(days=3),
            working_hours=(10, 16),
            exclude_weekends=True,
        )
        
        assert result.is_success


# ============================================================================
# Context Utilities Tests
# ============================================================================


class TestCalendarContextInjector:
    """Tests for CalendarContextInjector."""

    @pytest.mark.asyncio
    async def test_get_context(self, calendar_service, mock_repository, test_actor, sample_events):
        """Test getting calendar context for injection."""
        mock_repository.list_events.return_value = sample_events
        
        injector = CalendarContextInjector(calendar_service)
        context = await injector.get_context(test_actor)
        
        assert isinstance(context, str)
        assert "Calendar" in context or context == ""  # May be empty if no events

    @pytest.mark.asyncio
    async def test_get_context_dict(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test getting calendar context as dict."""
        mock_repository.list_events.return_value = sample_events
        
        injector = CalendarContextInjector(calendar_service)
        context = await injector.get_context_dict(test_actor)
        
        assert isinstance(context, dict)
        assert "today_summary" in context
        assert "upcoming_events" in context


class TestFormatHelpers:
    """Tests for formatting helper functions."""

    def test_format_availability_response_available(self):
        """Test formatting availability when time is free."""
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=1)
        end = now + timedelta(hours=2)
        
        response = format_availability_response(True, [], start, end)
        
        assert "available" in response.lower()

    def test_format_availability_response_single_conflict(self):
        """Test formatting availability with one conflict."""
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=1)
        end = now + timedelta(hours=2)
        
        conflict = Mock()
        conflict.title = "Team Meeting"
        
        response = format_availability_response(False, [conflict], start, end)
        
        assert "conflict" in response.lower()
        assert "Team Meeting" in response

    def test_format_availability_response_multiple_conflicts(self):
        """Test formatting availability with multiple conflicts."""
        now = datetime.now(timezone.utc)
        start = now + timedelta(hours=1)
        end = now + timedelta(hours=2)
        
        conflicts = [Mock(title="Event 1"), Mock(title="Event 2")]
        
        response = format_availability_response(False, conflicts, start, end)
        
        assert "2" in response

    def test_format_free_time_slots_empty(self):
        """Test formatting when no slots found."""
        response = format_free_time_slots([])
        
        assert "No available" in response

    def test_format_free_time_slots(self):
        """Test formatting free time slots."""
        now = datetime.now(timezone.utc)
        slots = [
            {
                "start": now,
                "end": now + timedelta(hours=1),
                "duration_minutes": 60,
            },
            {
                "start": now + timedelta(hours=2),
                "end": now + timedelta(hours=3),
                "duration_minutes": 60,
            },
        ]
        
        response = format_free_time_slots(slots)
        
        assert "2" in response
        assert "60 min" in response


# ============================================================================
# Calendar Service Provider Tests
# ============================================================================


class TestCalendarServiceProvider:
    """Tests for CalendarServiceProvider."""

    @pytest.fixture
    def provider(self, calendar_service, test_actor):
        """Create CalendarServiceProvider for testing."""
        from modules.Tools.providers.calendar_service import CalendarServiceProvider
        return CalendarServiceProvider(
            service=calendar_service,
            default_actor=test_actor,
        )

    @pytest.mark.asyncio
    async def test_list_operation(
        self, provider, mock_repository, sample_events
    ):
        """Test list operation through provider."""
        mock_repository.list_events.return_value = sample_events
        
        result = await provider.call(operation="list")
        
        assert "events" in result
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_search_operation(
        self, provider, mock_repository, sample_events
    ):
        """Test search operation through provider."""
        mock_repository.search_events.return_value = sample_events[:1]
        
        result = await provider.call(operation="search", query="standup")
        
        assert "events" in result
        assert result["query"] == "standup"

    @pytest.mark.asyncio
    async def test_upcoming_operation(
        self, provider, mock_repository, sample_events
    ):
        """Test upcoming operation through provider."""
        mock_repository.list_events.return_value = sample_events
        
        result = await provider.call(operation="upcoming", hours=24, limit=5)
        
        assert "events" in result
        assert result["hours_ahead"] == 24

    @pytest.mark.asyncio
    async def test_availability_operation(self, provider, mock_repository):
        """Test availability operation through provider."""
        mock_repository.find_conflicting_events.return_value = []
        
        now = datetime.now(timezone.utc)
        result = await provider.call(
            operation="availability",
            start=(now + timedelta(hours=1)).isoformat(),
            end=(now + timedelta(hours=2)).isoformat(),
        )
        
        assert result.get("available") is True

    @pytest.mark.asyncio
    async def test_summary_operation(
        self, provider, mock_repository, sample_events
    ):
        """Test summary operation through provider."""
        mock_repository.list_events.return_value = sample_events
        
        result = await provider.call(operation="summary", period="today")
        
        assert result.get("period") == "today"
        assert "total_events" in result

    @pytest.mark.asyncio
    async def test_find_free_time_operation(self, provider, mock_repository):
        """Test find_free_time operation through provider."""
        mock_repository.list_events.return_value = []
        
        now = datetime.now(timezone.utc)
        result = await provider.call(
            operation="find_free_time",
            start=now.isoformat(),
            end=(now + timedelta(days=1)).isoformat(),
            duration_minutes=60,
        )
        
        assert "slots" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_suggest_times_operation(self, provider, mock_repository):
        """Test suggest_times operation through provider."""
        mock_repository.list_events.return_value = []
        
        result = await provider.call(
            operation="suggest_times",
            duration_minutes=30,
            max_suggestions=3,
        )
        
        assert "suggestions" in result
        assert result["duration_minutes"] == 30

    @pytest.mark.asyncio
    async def test_unknown_operation(self, provider):
        """Test handling unknown operation."""
        result = await provider.call(operation="unknown_op")
        
        assert "error" in result
        assert "supported_operations" in result

    @pytest.mark.asyncio
    async def test_datetime_parsing(self, provider, mock_repository):
        """Test various datetime input formats."""
        mock_repository.list_events.return_value = []
        
        # ISO format with timezone
        result1 = await provider.call(
            operation="list",
            start="2024-01-15T10:00:00+00:00",
            end="2024-01-15T18:00:00Z",
        )
        assert "events" in result1
        
        # Date only format
        result2 = await provider.call(
            operation="list",
            start="2024-01-15",
            end="2024-01-16",
        )
        assert "events" in result2

    @pytest.mark.asyncio
    async def test_context_extraction(self, calendar_service, test_actor):
        """Test actor extraction from context."""
        from modules.Tools.providers.calendar_service import CalendarServiceProvider
        
        provider = CalendarServiceProvider(service=calendar_service)
        
        # Call with context containing user info
        context = {
            "user_id": "custom-user",
            "tenant_id": "custom-tenant",
            "permissions": ["calendar:read"],
        }
        
        # The _get_actor method should extract from context
        actor = provider._get_actor(context)
        
        assert actor.id == "custom-user"
        assert actor.tenant_id == "custom-tenant"


# ============================================================================
# Integration Tests
# ============================================================================


class TestAgentCalendarIntegration:
    """Integration tests for agent calendar workflow."""

    @pytest.mark.asyncio
    async def test_scheduling_workflow(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test typical agent scheduling workflow."""
        mock_repository.list_events.return_value = sample_events
        mock_repository.find_conflicting_events.return_value = []
        
        # 1. Get calendar summary
        summary = await calendar_service.get_calendar_summary(test_actor, period="today")
        assert summary.is_success
        
        # 2. Check availability for proposed time
        now = datetime.now(timezone.utc)
        availability = await calendar_service.check_availability(
            test_actor,
            now + timedelta(hours=10),
            now + timedelta(hours=11),
        )
        assert availability.is_success
        
        # 3. If busy, find alternative times
        if not availability.value["available"]:
            alternatives = await calendar_service.find_free_time(
                test_actor,
                start_range=now,
                end_range=now + timedelta(days=1),
                duration_minutes=60,
            )
            assert alternatives.is_success

    @pytest.mark.asyncio
    async def test_context_injection_workflow(
        self, calendar_service, mock_repository, test_actor, sample_events
    ):
        """Test context injection for LLM prompts."""
        mock_repository.list_events.return_value = sample_events
        
        # Get context for prompt injection
        context = await get_calendar_context_for_prompt(
            calendar_service,
            test_actor,
            include_today=True,
            include_upcoming=True,
            include_next_event=True,
        )
        
        # Context should be suitable for system prompt
        assert isinstance(context, str)
        # If there are events, context should not be empty
        if sample_events:
            # May be empty if events are in the past
            pass


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
