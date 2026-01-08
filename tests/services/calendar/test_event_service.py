"""
Tests for calendar event service.

Tests the main CalendarEventService with comprehensive CRUD operations,
validation, conflict detection, and event publishing.

Author: ATLAS Team
Date: Jan 7, 2026
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, call

from core.services.common import OperationResult, ValidationError, NotFoundError, ConflictError
from core.services.calendar.event_service import CalendarEventService
from core.services.calendar.types import CalendarEvent, CalendarEventCreate, CalendarEventUpdate, EventStatus
from core.services.calendar.events import CalendarEventCreated, CalendarEventUpdated, CalendarEventDeleted
from modules.calendar_store.dataclasses import EventVisibility, BusyStatus


def make_test_event(
    event_id: str = "00000000-0000-0000-0000-000000000123",
    title: str = "Test Event",
    description: str = "Test description",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    created_by: str = "user-123",
    tenant_id: str = "tenant-1",
    **kwargs
) -> CalendarEvent:
    """Helper to create CalendarEvent instances with all required fields."""
    if start_time is None:
        start_time = datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc)
    if end_time is None:
        end_time = datetime(2026, 2, 1, 11, 0, tzinfo=timezone.utc)
    
    defaults = {
        "event_id": event_id,
        "title": title,
        "description": description,
        "start_time": start_time,
        "end_time": end_time,
        "timezone_name": "UTC",
        "location": None,
        "status": EventStatus.CONFIRMED,
        "visibility": EventVisibility.PRIVATE,
        "busy_status": BusyStatus.BUSY,
        "all_day": False,
        "category_id": None,
        "created_by": created_by,
        "tenant_id": tenant_id,
        "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        "is_recurring": False,
        "recurrence_pattern": None,
    }
    defaults.update(kwargs)
    return CalendarEvent(**defaults)


@pytest.fixture
def mock_repository():
    """Mock calendar repository."""
    repo = Mock()
    repo.create_event = AsyncMock()
    repo.get_event_by_id = AsyncMock()
    repo.get_events_by_ids = AsyncMock()
    repo.update_event = AsyncMock()
    repo.delete_event = AsyncMock()
    repo.bulk_delete_events = AsyncMock()
    repo.list_events = AsyncMock()
    repo.find_conflicting_events = AsyncMock()
    repo.search_events = AsyncMock()
    return repo


@pytest.fixture
def mock_permission_checker():
    """Mock permission checker."""
    checker = Mock()
    checker.require_read_permission = AsyncMock()
    checker.require_write_permission = AsyncMock()
    checker.require_event_read_permission = AsyncMock()
    checker.require_event_edit_permission = AsyncMock()
    checker.require_event_delete_permission = AsyncMock()
    checker.can_read_event = AsyncMock(return_value=True)
    checker.can_delete_event = AsyncMock(return_value=True)
    checker.filter_events_by_permissions = AsyncMock()
    return checker


@pytest.fixture
def mock_validator():
    """Mock event validator."""
    validator = Mock()
    validator.validate_create = AsyncMock(return_value=OperationResult.success(None))
    validator.validate_update = AsyncMock(return_value=OperationResult.success(None))
    validator.validate_event_conflict_resolution = Mock(return_value=OperationResult.success(None))
    validator.validate_bulk_operation = Mock(return_value=OperationResult.success(None))
    return validator


@pytest.fixture
def mock_event_publisher():
    """Mock domain event publisher."""
    publisher = Mock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def calendar_service(mock_repository, mock_permission_checker, mock_validator, mock_event_publisher):
    """Create CalendarEventService for testing."""
    return CalendarEventService(
        repository=mock_repository,
        permission_checker=mock_permission_checker,
        validator=mock_validator,
        event_publisher=mock_event_publisher,
    )


@pytest.fixture
def actor():
    """Standard test actor."""
    from core.services.common import Actor
    return Actor(type="user", id="user-123", tenant_id="tenant-1", permissions={"calendar.read", "calendar.write"})


@pytest.fixture
def sample_event_data():
    """Sample event creation data."""
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
def sample_event():
    """Sample calendar event."""
    return make_test_event(
        event_id="00000000-0000-0000-0000-000000000123",
        title="Team Meeting",
        description="Weekly team standup",
        location="Conference Room A",
    )


class TestCreateEvent:
    """Test event creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_event_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        mock_validator,
        mock_event_publisher,
        actor,
        sample_event_data,
        sample_event
    ):
        """Successful event creation should work end-to-end."""
        # Setup mocks
        mock_repository.find_conflicting_events.return_value = []
        mock_repository.create_event.return_value = sample_event
        
        # Execute
        result = await calendar_service.create_event(actor, sample_event_data)
        
        # Verify success
        assert result.is_success
        assert result.value == sample_event
        
        # Verify permission check
        mock_permission_checker.require_write_permission.assert_called_once_with(actor)
        
        # Verify validation
        mock_validator.validate_create.assert_called_once_with(sample_event_data)
        
        # Verify conflict check
        mock_repository.find_conflicting_events.assert_called_once()
        
        # Verify repository call
        mock_repository.create_event.assert_called_once()
        
        # Verify event published
        mock_event_publisher.publish.assert_called_once()
        published_event = mock_event_publisher.publish.call_args[0][0]
        assert isinstance(published_event, CalendarEventCreated)
        assert published_event.event_id == sample_event.event_id
    
    @pytest.mark.asyncio
    async def test_create_event_permission_denied(
        self,
        calendar_service,
        mock_permission_checker,
        actor,
        sample_event_data
    ):
        """Event creation should fail without write permission."""
        from core.services.common import PermissionDeniedError
        mock_permission_checker.require_write_permission.side_effect = PermissionDeniedError("Access denied")
        
        result = await calendar_service.create_event(actor, sample_event_data)
        
        assert not result.is_success
        assert "Access denied" in result.error
        assert result.error_code == "PERMISSION_DENIED"
    
    @pytest.mark.asyncio
    async def test_create_event_validation_failure(
        self,
        calendar_service,
        mock_validator,
        actor,
        sample_event_data
    ):
        """Event creation should fail with validation errors."""
        mock_validator.validate_create.return_value = OperationResult.failure(
            "Title is required", "VALIDATION_FAILED"
        )
        
        result = await calendar_service.create_event(actor, sample_event_data)
        
        assert not result.is_success
        assert result.error == "Title is required"
        assert result.error_code == "VALIDATION_FAILED"
    
    @pytest.mark.asyncio
    async def test_create_event_with_conflicts_block_strategy(
        self,
        calendar_service,
        mock_repository,
        mock_validator,
        actor,
        sample_event_data,
        sample_event
    ):
        """Event creation should fail when conflicts exist and strategy is block."""
        # Setup conflicting event
        conflicting_event = make_test_event(
            event_id="evt-conflict",
            title="Conflicting Meeting",
            start_time=sample_event_data.start_time,
            end_time=sample_event_data.end_time,
            created_by="other-user",
        )
        
        mock_repository.find_conflicting_events.return_value = [conflicting_event]
        
        result = await calendar_service.create_event(
            actor, 
            sample_event_data, 
            conflict_resolution="block"
        )
        
        assert not result.is_success
        assert "scheduling conflicts" in result.error
        assert result.error_code == "SCHEDULING_CONFLICT"
    
    @pytest.mark.asyncio
    async def test_create_event_with_conflicts_ignore_strategy(
        self,
        calendar_service,
        mock_repository,
        actor,
        sample_event_data,
        sample_event
    ):
        """Event creation should succeed when conflicts exist but strategy is ignore."""
        # Setup conflicting event
        conflicting_event = make_test_event(
            event_id="evt-conflict",
            title="Conflicting Meeting",
            start_time=sample_event_data.start_time,
            end_time=sample_event_data.end_time,
            created_by="other-user",
        )
        
        mock_repository.find_conflicting_events.return_value = [conflicting_event]
        mock_repository.create_event.return_value = sample_event
        
        result = await calendar_service.create_event(
            actor,
            sample_event_data,
            conflict_resolution="ignore"
        )
        
        assert result.is_success
        assert result.value == sample_event


class TestGetEvent:
    """Test event retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_event_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        actor,
        sample_event
    ):
        """Successful event retrieval should work."""
        mock_repository.get_event_by_id.return_value = sample_event
        
        result = await calendar_service.get_event(actor, "evt-123")
        
        assert result.is_success
        assert result.value == sample_event
        
        # Verify permission check
        mock_permission_checker.require_event_read_permission.assert_called_once_with(
            actor, sample_event
        )
    
    @pytest.mark.asyncio
    async def test_get_event_not_found(
        self,
        calendar_service,
        mock_repository,
        actor
    ):
        """Getting non-existent event should return not found error."""
        mock_repository.get_event_by_id.return_value = None
        
        result = await calendar_service.get_event(actor, "evt-nonexistent")
        
        assert not result.is_success
        assert result.error_code == "NOT_FOUND"
        assert "Event not found" in result.error
    
    @pytest.mark.asyncio
    async def test_get_event_permission_denied(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        actor,
        sample_event
    ):
        """Getting event without permission should fail."""
        from core.services.common import PermissionDeniedError
        mock_repository.get_event_by_id.return_value = sample_event
        mock_permission_checker.require_event_read_permission.side_effect = PermissionDeniedError("Access denied")
        
        result = await calendar_service.get_event(actor, "evt-123")
        
        assert not result.is_success
        assert result.error_code == "PERMISSION_DENIED"


class TestUpdateEvent:
    """Test event update functionality."""
    
    @pytest.mark.asyncio
    async def test_update_event_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        mock_validator,
        mock_event_publisher,
        actor,
        sample_event
    ):
        """Successful event update should work."""
        update_data = CalendarEventUpdate(
            title="Updated Meeting",
            description="Updated description"
        )
        
        updated_event = make_test_event(
            event_id=sample_event.event_id,
            title=update_data.title or sample_event.title,
            description=update_data.description or sample_event.description,
            start_time=sample_event.start_time,
            end_time=sample_event.end_time,
            location=sample_event.location,
            created_by=sample_event.created_by,
        )
        
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.find_conflicting_events.return_value = []
        mock_repository.update_event.return_value = updated_event
        
        result = await calendar_service.update_event(actor, "evt-123", update_data)
        
        assert result.is_success
        assert result.value == updated_event
        
        # Verify permission check  
        mock_permission_checker.require_event_edit_permission.assert_called_once_with(
            actor, sample_event
        )
        
        # Verify validation
        mock_validator.validate_update.assert_called_once_with(update_data, sample_event)
        
        # Verify repository update
        mock_repository.update_event.assert_called_once()
        
        # Verify event published
        mock_event_publisher.publish.assert_called_once()
        published_event = mock_event_publisher.publish.call_args[0][0]
        assert isinstance(published_event, CalendarEventUpdated)
    
    @pytest.mark.asyncio
    async def test_update_event_not_found(
        self,
        calendar_service,
        mock_repository,
        actor
    ):
        """Updating non-existent event should fail."""
        mock_repository.get_event_by_id.return_value = None
        update_data = CalendarEventUpdate(title="Updated")
        
        result = await calendar_service.update_event(actor, "evt-nonexistent", update_data)
        
        assert not result.is_success
        assert result.error_code == "NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_update_event_with_time_conflicts(
        self,
        calendar_service,
        mock_repository,
        mock_validator,
        actor,
        sample_event
    ):
        """Event updates that create conflicts should fail with block strategy."""
        update_data = CalendarEventUpdate(
            start_time=datetime(2026, 2, 1, 14, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc)
        )
        
        conflicting_event = make_test_event(
            event_id="evt-other",
            title="Other Meeting",
            start_time=update_data.start_time,
            end_time=update_data.end_time,
            created_by="other-user",
        )
        
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.find_conflicting_events.return_value = [conflicting_event]
        
        result = await calendar_service.update_event(
            actor, 
            "evt-123", 
            update_data, 
            conflict_resolution="block"
        )
        
        assert not result.is_success
        assert result.error_code == "SCHEDULING_CONFLICT"


class TestDeleteEvent:
    """Test event deletion functionality."""
    
    @pytest.mark.asyncio
    async def test_delete_event_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        mock_event_publisher,
        actor,
        sample_event
    ):
        """Successful event deletion should work."""
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.delete_event.return_value = True
        
        result = await calendar_service.delete_event(actor, sample_event.event_id)
        
        assert result.is_success
        assert result.value is True
        
        # Verify permission check
        mock_permission_checker.require_event_delete_permission.assert_called_once_with(
            actor, sample_event
        )
        
        # Verify repository deletion
        mock_repository.delete_event.assert_called_once_with(sample_event.event_id)
        
        # Verify event published
        mock_event_publisher.publish.assert_called_once()
        published_event = mock_event_publisher.publish.call_args[0][0]
        assert isinstance(published_event, CalendarEventDeleted)
        assert published_event.event_id == sample_event.event_id
    
    @pytest.mark.asyncio
    async def test_delete_event_not_found(
        self,
        calendar_service,
        mock_repository,
        actor
    ):
        """Deleting non-existent event should fail."""
        mock_repository.get_event_by_id.return_value = None
        
        result = await calendar_service.delete_event(actor, "evt-nonexistent")
        
        assert not result.is_success
        assert result.error_code == "NOT_FOUND"


class TestListEvents:
    """Test event listing functionality."""
    
    @pytest.mark.asyncio
    async def test_list_events_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        actor,
        sample_event
    ):
        """Successful event listing should work."""
        all_events = [sample_event]
        filtered_events = [sample_event]
        
        mock_repository.list_events.return_value = all_events
        mock_permission_checker.filter_events_by_permissions.return_value = filtered_events
        
        result = await calendar_service.list_events(actor)
        
        assert result.is_success
        assert result.value == filtered_events
        
        # Verify permission check
        mock_permission_checker.require_read_permission.assert_called_once_with(actor)
        
        # Verify filtering
        mock_permission_checker.filter_events_by_permissions.assert_called_once_with(
            actor, all_events
        )
    
    @pytest.mark.asyncio
    async def test_list_events_with_filters(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        actor,
        sample_event
    ):
        """Event listing with date filters should work."""
        start_date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 2, 28, tzinfo=timezone.utc)
        
        mock_repository.list_events.return_value = [sample_event]
        mock_permission_checker.filter_events_by_permissions.return_value = [sample_event]
        
        result = await calendar_service.list_events(
            actor,
            start_date=start_date,
            end_date=end_date
        )
        
        assert result.is_success
        
        # Verify repository called with filters
        mock_repository.list_events.assert_called_once_with(
            start_date=start_date,
            end_date=end_date,
            status=None,
            limit=None,
            offset=None
        )


class TestBulkOperations:
    """Test bulk operations on events."""
    
    @pytest.mark.asyncio
    async def test_bulk_delete_success(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        mock_validator,
        mock_event_publisher,
        actor,
        sample_event
    ):
        """Bulk delete should work for authorized events."""
        event_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002", 
            "00000000-0000-0000-0000-000000000003"
        ] 
        events = [
            make_test_event(event_id=eid, title=f"Event {i+1}", created_by=actor.id)
            for i, eid in enumerate(event_ids)
        ]
        
        mock_repository.get_events_by_ids.return_value = events
        mock_permission_checker.can_delete_event.return_value = True
        mock_repository.bulk_delete_events.return_value = len(event_ids)
        
        result = await calendar_service.bulk_delete_events(actor, event_ids)
        
        assert result.is_success
        assert result.value == len(event_ids)
        
        # Verify validation
        mock_validator.validate_bulk_operation.assert_called_once_with(event_ids, "delete")
        
        # Verify permission checks for each event
        assert mock_permission_checker.can_delete_event.call_count == len(event_ids)
        
        # Verify events published
        assert mock_event_publisher.publish.call_count == len(event_ids)
    
    @pytest.mark.asyncio
    async def test_bulk_delete_partial_permissions(
        self,
        calendar_service,
        mock_repository,
        mock_permission_checker,
        mock_validator,
        actor
    ):
        """Bulk delete should only delete events user has permission for."""
        event_id_1 = "00000000-0000-0000-0000-000000000001"
        event_id_2 = "00000000-0000-0000-0000-000000000002"
        event_id_3 = "00000000-0000-0000-0000-000000000003"
        event_ids = [event_id_1, event_id_2, event_id_3]
        events = [
            make_test_event(event_id=event_id_1, title="Event 1", created_by=actor.id),
            make_test_event(event_id=event_id_2, title="Event 2", created_by="other-user"),
            make_test_event(event_id=event_id_3, title="Event 3", created_by=actor.id),
        ]
        
        mock_repository.get_events_by_ids.return_value = events
        # User can delete their own events but not others
        mock_permission_checker.can_delete_event.side_effect = lambda actor, event: event.created_by == actor.user_id
        mock_repository.bulk_delete_events.return_value = 2  # Only 2 deleted
        
        result = await calendar_service.bulk_delete_events(actor, event_ids)
        
        assert result.is_success
        assert result.value == 2
        
        # Should have called delete with only the 2 events user can delete
        expected_deletable_ids = [event_id_1, event_id_3]
        mock_repository.bulk_delete_events.assert_called_once_with(expected_deletable_ids)