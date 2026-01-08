"""
Tests for calendar event validation.

Tests the comprehensive validation logic that ensures calendar events
meet business rules and data integrity requirements.

Author: ATLAS Team
Date: Jan 7, 2026
"""

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from core.services.calendar.validation import CalendarEventValidator
from core.services.calendar.types import CalendarEventCreate, CalendarEventUpdate, CalendarEvent, EventStatus
from modules.calendar_store.dataclasses import EventVisibility, BusyStatus


def make_test_event(
    event_id: str = "evt-123",
    title: str = "Test Event",
    **kwargs
) -> CalendarEvent:
    """Helper to create CalendarEvent instances with all required fields."""
    defaults = {
        "event_id": event_id,
        "title": title,
        "description": "Test description",
        "start_time": datetime(2026, 2, 1, 14, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
        "timezone_name": "UTC",
        "location": "Room B",
        "status": EventStatus.CONFIRMED,
        "visibility": EventVisibility.PRIVATE,
        "busy_status": BusyStatus.BUSY,
        "all_day": False,
        "category_id": None,
        "created_by": "user-123",
        "tenant_id": "tenant-1",
        "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "is_recurring": False,
        "recurrence_pattern": None,
    }
    defaults.update(kwargs)
    return CalendarEvent(**defaults)


@pytest.fixture
def validator():
    """Create a CalendarEventValidator for testing."""
    return CalendarEventValidator()


@pytest.fixture
def valid_event_data():
    """Valid event data for testing."""
    return CalendarEventCreate(
        title="Team Meeting",
        description="Weekly team standup",
        start_time=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc),
        all_day=False,
        timezone_name="UTC",
        location="Conference Room A",
        is_recurring=False,
        recurrence_pattern=None,
    )


@pytest.fixture  
def existing_event():
    """An existing event for update testing."""
    return make_test_event(
        event_id="evt-123",
        title="Existing Meeting",
        description="Existing description",
    )


class TestCreateValidation:
    """Test calendar event creation validation."""
    
    @pytest.mark.asyncio
    async def test_valid_event_passes(self, validator, valid_event_data):
        """Valid event data should pass validation."""
        result = await validator.validate_create(valid_event_data)
        assert result.is_success
        assert result.value is None
    
    @pytest.mark.asyncio 
    async def test_empty_title_fails(self, validator, valid_event_data):
        """Events with empty titles should fail validation."""
        valid_event_data.title = ""
        result = await validator.validate_create(valid_event_data)
        
        assert not result.is_success
        assert "Title is required" in result.error
        assert result.error_code == "VALIDATION_FAILED"
    
    @pytest.mark.asyncio
    async def test_whitespace_title_fails(self, validator, valid_event_data):
        """Events with whitespace-only titles should fail."""
        valid_event_data.title = "   "
        result = await validator.validate_create(valid_event_data)
        
        assert not result.is_success
        assert "cannot be empty or whitespace only" in result.error
    
    @pytest.mark.asyncio
    async def test_long_title_fails(self, validator, valid_event_data):
        """Titles exceeding max length should fail validation."""
        valid_event_data.title = "x" * 256  # Exceeds 255 limit
        result = await validator.validate_create(valid_event_data)
        
        assert not result.is_success
        assert "cannot exceed 255 characters" in result.error
    
    @pytest.mark.asyncio
    async def test_long_description_fails(self, validator, valid_event_data):
        """Descriptions exceeding max length should fail."""
        valid_event_data.description = "x" * 10001  # Exceeds 10000 limit
        result = await validator.validate_create(valid_event_data)
        
        assert not result.is_success
        assert "cannot exceed 10000 characters" in result.error
    
    @pytest.mark.asyncio
    async def test_start_after_end_fails(self, validator, valid_event_data):
        """Events with start time after end time should fail."""
        # Swap start and end times
        valid_event_data.start_time, valid_event_data.end_time = (
            valid_event_data.end_time, valid_event_data.start_time
        )
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "Start time must be before end time" in result.error
    
    @pytest.mark.asyncio
    async def test_same_start_end_time_fails(self, validator, valid_event_data):
        """Events with identical start and end times should fail."""
        valid_event_data.end_time = valid_event_data.start_time
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "Start time must be before end time" in result.error
    
    @pytest.mark.asyncio
    async def test_too_long_event_fails(self, validator, valid_event_data):
        """Events longer than max duration should fail."""
        # Create an event lasting more than 365 days
        valid_event_data.end_time = valid_event_data.start_time.replace(year=2027, month=3)
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "cannot exceed 365 days" in result.error
    
    @pytest.mark.asyncio
    async def test_too_short_event_fails(self, validator, valid_event_data):
        """Non-all-day events shorter than minimum duration should fail."""
        # Create a 30-second event (less than 1 minute minimum)
        valid_event_data.end_time = valid_event_data.start_time.replace(second=30)
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "must be at least 1 minute" in result.error
    
    @pytest.mark.asyncio
    async def test_all_day_non_midnight_warns(self, validator, valid_event_data):
        """All-day events with non-midnight times should fail."""
        valid_event_data.all_day = True
        # Keep non-midnight times
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "should have times set to midnight" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_timezone_fails(self, validator, valid_event_data):
        """Invalid timezone names should fail validation."""
        valid_event_data.timezone_name = "Invalid/Timezone"
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "Invalid timezone" in result.error
    
    @pytest.mark.asyncio
    async def test_recurring_without_pattern_fails(self, validator, valid_event_data):
        """Recurring events without recurrence patterns should fail."""
        valid_event_data.is_recurring = True
        valid_event_data.recurrence_pattern = None
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "must have a recurrence pattern" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_recurrence_pattern_fails(self, validator, valid_event_data):
        """Invalid recurrence patterns should fail validation."""
        valid_event_data.is_recurring = True
        valid_event_data.recurrence_pattern = "INVALID_PATTERN"
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "must be a valid RRULE" in result.error
    
    @pytest.mark.asyncio
    async def test_valid_recurrence_pattern_passes(self, validator, valid_event_data):
        """Valid recurrence patterns should pass validation."""
        valid_event_data.is_recurring = True
        valid_event_data.recurrence_pattern = "FREQ=WEEKLY;BYDAY=MO,WE,FR"
        
        result = await validator.validate_create(valid_event_data)
        assert result.is_success


class TestUpdateValidation:
    """Test calendar event update validation."""
    
    @pytest.mark.asyncio
    async def test_valid_update_passes(self, validator, existing_event):
        """Valid update data should pass validation."""
        update_data = CalendarEventUpdate(
            title="Updated Meeting",
            description="Updated description"
        )
        
        result = await validator.validate_update(update_data, existing_event)
        assert result.is_success
    
    @pytest.mark.asyncio
    async def test_update_title_validation(self, validator, existing_event):
        """Title updates should be validated."""
        update_data = CalendarEventUpdate(title="")
        
        result = await validator.validate_update(update_data, existing_event)
        assert not result.is_success
        assert "Title is required" in result.error
    
    @pytest.mark.asyncio
    async def test_update_time_validation(self, validator, existing_event):
        """Time updates should be validated."""
        # Update with invalid time range
        update_data = CalendarEventUpdate(
            start_time=datetime(2026, 2, 1, 16, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 2, 1, 14, 0, tzinfo=timezone.utc)  # Before start
        )
        
        result = await validator.validate_update(update_data, existing_event)
        assert not result.is_success
        assert "Start time must be before end time" in result.error
    
    @pytest.mark.asyncio
    async def test_partial_time_update_uses_existing(self, validator, existing_event):
        """Partial time updates should combine with existing data."""
        # Only update start time - should validate against existing end time
        update_data = CalendarEventUpdate(
            start_time=datetime(2026, 2, 1, 16, 0, tzinfo=timezone.utc)  # After existing end
        )
        
        result = await validator.validate_update(update_data, existing_event)
        assert not result.is_success
        assert "Start time must be before end time" in result.error


class TestConflictValidation:
    """Test conflict resolution validation."""
    
    def test_valid_strategy_passes(self, validator):
        """Valid conflict resolution strategies should pass."""
        result = validator.validate_event_conflict_resolution([], "ignore")
        assert result.is_success
        
        result = validator.validate_event_conflict_resolution([], "warn")
        assert result.is_success
        
        result = validator.validate_event_conflict_resolution([], "block")
        assert result.is_success
        
        result = validator.validate_event_conflict_resolution([], "auto_reschedule")
        assert result.is_success
    
    def test_invalid_strategy_fails(self, validator):
        """Invalid strategies should fail validation."""
        result = validator.validate_event_conflict_resolution([], "invalid_strategy")
        
        assert not result.is_success
        assert "Invalid conflict resolution strategy" in result.error
        assert result.error_code == "INVALID_STRATEGY"
    
    def test_too_many_conflicts_for_auto_reschedule(self, validator, existing_event):
        """Auto-reschedule should fail with too many conflicts."""
        many_conflicts = [existing_event] * 6  # More than 5
        
        result = validator.validate_event_conflict_resolution(
            many_conflicts, 
            "auto_reschedule"
        )
        
        assert not result.is_success
        assert "more than 5 conflicts" in result.error
        assert result.error_code == "TOO_MANY_CONFLICTS"


class TestBulkValidation:
    """Test bulk operation validation."""
    
    def test_valid_bulk_operation_passes(self, validator):
        """Valid bulk operations should pass."""
        event_ids = ["evt-1", "evt-2", "evt-3"]
        
        result = validator.validate_bulk_operation(event_ids, "delete")
        assert result.is_success
        
        result = validator.validate_bulk_operation(event_ids, "update_category")
        assert result.is_success
    
    def test_empty_event_list_fails(self, validator):
        """Empty event ID lists should fail."""
        result = validator.validate_bulk_operation([], "delete")
        
        assert not result.is_success
        assert "cannot be empty" in result.error
        assert result.error_code == "EMPTY_EVENT_LIST"
    
    def test_too_many_events_fails(self, validator):
        """Bulk operations with too many events should fail."""
        too_many_events = [f"evt-{i}" for i in range(101)]  # More than 100
        
        result = validator.validate_bulk_operation(too_many_events, "delete")
        
        assert not result.is_success
        assert "limited to 100 events" in result.error
        assert result.error_code == "TOO_MANY_EVENTS"
    
    def test_invalid_operation_fails(self, validator):
        """Invalid bulk operations should fail."""
        event_ids = ["evt-1"]
        
        result = validator.validate_bulk_operation(event_ids, "invalid_operation")
        
        assert not result.is_success
        assert "Invalid bulk operation" in result.error
        assert result.error_code == "INVALID_OPERATION"


class TestTimezoneHandling:
    """Test timezone-specific validation."""
    
    @pytest.mark.asyncio
    async def test_common_timezones_pass(self, validator, valid_event_data):
        """Common timezone names should pass validation."""
        timezones = [
            "UTC",
            "America/New_York", 
            "Europe/London",
            "Asia/Tokyo",
            "Australia/Sydney"
        ]
        
        for tz in timezones:
            valid_event_data.timezone_name = tz
            result = await validator.validate_create(valid_event_data)
            assert result.is_success, f"Timezone {tz} should be valid"
    
    @pytest.mark.asyncio
    async def test_empty_timezone_fails(self, validator, valid_event_data):
        """Empty timezone should fail validation."""
        valid_event_data.timezone_name = ""
        
        result = await validator.validate_create(valid_event_data)
        assert not result.is_success
        assert "Timezone is required" in result.error