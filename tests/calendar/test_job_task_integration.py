"""Integration tests for calendar event linking to jobs and tasks.

Tests the following features:
- Link models (JobEventLink, TaskEventLink)
- Event service linking methods (link_to_job, link_to_task, etc.)
- Event creation from jobs/tasks
- JobTaskEventHandler auto-sync

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from modules.calendar_store.link_models import (
    LinkType,
    SyncBehavior,
    JobEventLink,
    TaskEventLink,
)
from core.services.calendar.event_service import CalendarEventService
from core.services.calendar.job_task_integration import (
    JobTaskEventHandler,
    create_job_task_handler,
)
from core.services.calendar.types import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventUpdate,
    EventStatus,
    EventVisibility,
    BusyStatus,
)
from core.services.common import Actor, OperationResult
from core.services.common.types import DomainEvent


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_repository():
    """Create a mock repository with async methods."""
    repo = MagicMock()
    
    # Event CRUD
    repo.get_event_by_id = AsyncMock(return_value=None)
    repo.create_event = AsyncMock()
    repo.update_event = AsyncMock()
    repo.delete_event = AsyncMock()
    repo.list_events = AsyncMock(return_value=[])
    repo.get_events_by_ids = AsyncMock(return_value=[])
    
    # Conflict detection
    repo.find_conflicting_events = AsyncMock(return_value=[])
    
    # Link operations
    repo.create_job_link = AsyncMock()
    repo.create_task_link = AsyncMock()
    repo.delete_job_link = AsyncMock(return_value=True)
    repo.delete_task_link = AsyncMock(return_value=True)
    repo.get_job_links_for_event = AsyncMock(return_value=[])
    repo.get_task_links_for_event = AsyncMock(return_value=[])
    repo.get_events_for_job = AsyncMock(return_value=[])
    repo.get_events_for_task = AsyncMock(return_value=[])
    
    return repo


@pytest.fixture
def mock_permissions():
    """Create a mock permission checker."""
    perms = MagicMock()
    perms.require_read_permission = AsyncMock()
    perms.require_write_permission = AsyncMock()
    perms.require_event_edit_permission = AsyncMock()
    perms.require_event_delete_permission = AsyncMock()
    perms.can_delete_event = AsyncMock(return_value=True)
    perms.filter_events_by_permissions = AsyncMock(side_effect=lambda actor, events: events)
    return perms


@pytest.fixture
def mock_publisher():
    """Create a mock event publisher."""
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def event_service(mock_repository, mock_permissions, mock_publisher):
    """Create a CalendarEventService with mocked dependencies."""
    return CalendarEventService(
        repository=mock_repository,
        permission_checker=mock_permissions,
        event_publisher=mock_publisher,
    )


@pytest.fixture
def test_actor():
    """Create a test actor."""
    return Actor(
        type="user",
        id="user-123",
        tenant_id="tenant-abc",
        permissions=["calendar.read", "calendar.write"],
    )


@pytest.fixture
def sample_event():
    """Create a sample calendar event."""
    return CalendarEvent(
        event_id=str(uuid4()),
        title="Test Event",
        description="Test Description",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        status=EventStatus.CONFIRMED,
        visibility=EventVisibility.PRIVATE,
        busy_status=BusyStatus.BUSY,
        timezone_name="UTC",
        tenant_id="tenant-abc",
    )


@pytest.fixture
def sample_job():
    """Create a sample job data dict."""
    return {
        "id": str(uuid4()),
        "name": "Daily Backup Job",
        "description": "Runs daily backup of all data",
        "status": "scheduled",
        "scheduled_at": datetime.now(timezone.utc) + timedelta(hours=2),
        "estimated_duration_minutes": 30,
    }


@pytest.fixture
def sample_task():
    """Create a sample task data dict."""
    return {
        "id": str(uuid4()),
        "title": "Review PR #123",
        "description": "Code review for feature branch",
        "status": "ready",
        "priority": 2,
        "due_at": datetime.now(timezone.utc) + timedelta(days=1),
        "estimated_minutes": 45,
    }


# ============================================================================
# Link Type Tests
# ============================================================================

class TestLinkTypes:
    """Tests for link type and sync behavior enums."""
    
    def test_link_type_values(self):
        """Verify all link types are defined."""
        assert LinkType.AUTO_CREATED.value == "auto_created"
        assert LinkType.MANUAL.value == "manual"
        assert LinkType.DEADLINE.value == "deadline"
        assert LinkType.WORK_BLOCK.value == "work_block"
        assert LinkType.MILESTONE.value == "milestone"
        assert LinkType.REVIEW.value == "review"
    
    def test_sync_behavior_values(self):
        """Verify all sync behaviors are defined."""
        assert SyncBehavior.NONE.value == "none"
        assert SyncBehavior.FROM_SOURCE.value == "from_source"
        assert SyncBehavior.FROM_EVENT.value == "from_event"
        assert SyncBehavior.BIDIRECTIONAL.value == "bidirectional"


# ============================================================================
# Service Linking Method Tests
# ============================================================================

class TestLinkToJob:
    """Tests for CalendarEventService.link_to_job."""
    
    @pytest.mark.asyncio
    async def test_link_to_job_success(
        self, event_service, mock_repository, mock_permissions, test_actor, sample_event
    ):
        """Successfully link an event to a job."""
        job_id = str(uuid4())
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.create_job_link.return_value = {
            "id": str(uuid4()),
            "event_id": sample_event.event_id,
            "job_id": job_id,
            "link_type": LinkType.MANUAL.value,
            "sync_behavior": SyncBehavior.FROM_SOURCE.value,
        }
        
        result = await event_service.link_to_job(
            test_actor,
            sample_event.event_id,
            job_id,
            link_type=LinkType.MANUAL,
            sync_behavior=SyncBehavior.FROM_SOURCE,
        )
        
        assert result.is_success
        assert result.value["job_id"] == job_id
        mock_repository.create_job_link.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_link_to_job_event_not_found(
        self, event_service, mock_repository, test_actor
    ):
        """Return failure when event doesn't exist."""
        mock_repository.get_event_by_id.return_value = None
        
        result = await event_service.link_to_job(
            test_actor,
            "nonexistent-event",
            str(uuid4()),
        )
        
        assert result.is_failure
        assert result.error_code == "NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_link_to_job_with_notes(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Link with custom notes."""
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.create_job_link.return_value = {"id": str(uuid4())}
        
        result = await event_service.link_to_job(
            test_actor,
            sample_event.event_id,
            str(uuid4()),
            notes="Linked for tracking purposes",
        )
        
        assert result.is_success
        call_kwargs = mock_repository.create_job_link.call_args.kwargs
        assert call_kwargs["notes"] == "Linked for tracking purposes"


class TestLinkToTask:
    """Tests for CalendarEventService.link_to_task."""
    
    @pytest.mark.asyncio
    async def test_link_to_task_success(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Successfully link an event to a task."""
        task_id = str(uuid4())
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.create_task_link.return_value = {
            "id": str(uuid4()),
            "event_id": sample_event.event_id,
            "task_id": task_id,
            "link_type": LinkType.DEADLINE.value,
        }
        
        result = await event_service.link_to_task(
            test_actor,
            sample_event.event_id,
            task_id,
            link_type=LinkType.DEADLINE,
        )
        
        assert result.is_success
        assert result.value["task_id"] == task_id


class TestUnlinkMethods:
    """Tests for unlink methods."""
    
    @pytest.mark.asyncio
    async def test_unlink_from_job(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Successfully unlink an event from a job."""
        job_id = str(uuid4())
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.delete_job_link.return_value = True
        
        result = await event_service.unlink_from_job(
            test_actor,
            sample_event.event_id,
            job_id,
        )
        
        assert result.is_success
        assert result.value is True
    
    @pytest.mark.asyncio
    async def test_unlink_from_task(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Successfully unlink an event from a task."""
        task_id = str(uuid4())
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.delete_task_link.return_value = True
        
        result = await event_service.unlink_from_task(
            test_actor,
            sample_event.event_id,
            task_id,
        )
        
        assert result.is_success
        assert result.value is True


class TestGetLinkedEntities:
    """Tests for retrieving linked jobs/tasks."""
    
    @pytest.mark.asyncio
    async def test_get_linked_jobs(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Get jobs linked to an event."""
        mock_repository.get_event_by_id.return_value = sample_event
        mock_repository.get_job_links_for_event.return_value = [
            {"job_id": str(uuid4()), "link_type": "manual"},
            {"job_id": str(uuid4()), "link_type": "auto_created"},
        ]
        
        result = await event_service.get_linked_jobs(
            test_actor,
            sample_event.event_id,
        )
        
        assert result.is_success
        assert len(result.value) == 2
    
    @pytest.mark.asyncio
    async def test_get_events_for_job(
        self, event_service, mock_repository, test_actor, sample_event
    ):
        """Get events linked to a job."""
        job_id = str(uuid4())
        mock_repository.get_events_for_job.return_value = [sample_event]
        
        result = await event_service.get_events_for_job(test_actor, job_id)
        
        assert result.is_success
        assert len(result.value) == 1
        assert result.value[0].event_id == sample_event.event_id


# ============================================================================
# Create Event From Job/Task Tests
# ============================================================================

class TestCreateEventFromJob:
    """Tests for creating calendar events from jobs."""
    
    @pytest.mark.asyncio
    async def test_create_event_from_job_success(
        self, event_service, mock_repository, mock_permissions, test_actor, sample_job
    ):
        """Successfully create an event from a job."""
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title=f"Job: {sample_job['name']}",
            description=sample_job["description"],
            start_time=sample_job["scheduled_at"],
            end_time=sample_job["scheduled_at"] + timedelta(minutes=30),
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="tenant-abc",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_job_link.return_value = {"id": str(uuid4())}
        
        result = await event_service.create_event_from_job(
            test_actor,
            sample_job,
        )
        
        assert result.is_success
        assert "Job:" in result.value.title
        mock_repository.create_event.assert_called_once()
        mock_repository.create_job_link.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_event_from_job_with_overrides(
        self, event_service, mock_repository, test_actor, sample_job
    ):
        """Create event with custom overrides."""
        custom_title = "Custom Job Event"
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title=custom_title,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="tenant-abc",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_job_link.return_value = {"id": str(uuid4())}
        
        result = await event_service.create_event_from_job(
            test_actor,
            sample_job,
            event_overrides={"title": custom_title},
        )
        
        assert result.is_success
    
    @pytest.mark.asyncio
    async def test_create_event_from_job_missing_id(
        self, event_service, test_actor
    ):
        """Fail when job has no ID."""
        job_without_id = {"name": "Test Job"}
        
        result = await event_service.create_event_from_job(
            test_actor,
            job_without_id,
        )
        
        assert result.is_failure
        assert result.error_code == "INVALID_INPUT"


class TestCreateEventFromTask:
    """Tests for creating calendar events from tasks."""
    
    @pytest.mark.asyncio
    async def test_create_event_from_task_with_due_date(
        self, event_service, mock_repository, test_actor, sample_task
    ):
        """Create event from task with due date (uses DEADLINE link type)."""
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title=f"Task: {sample_task['title']}",
            start_time=sample_task["due_at"] - timedelta(minutes=45),
            end_time=sample_task["due_at"],
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="tenant-abc",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_task_link.return_value = {"id": str(uuid4())}
        
        result = await event_service.create_event_from_task(
            test_actor,
            sample_task,
        )
        
        assert result.is_success
        assert "Task:" in result.value.title
        
        # Verify DEADLINE link type was used for task with due date
        call_kwargs = mock_repository.create_task_link.call_args.kwargs
        assert call_kwargs["link_type"] == LinkType.DEADLINE
    
    @pytest.mark.asyncio
    async def test_create_event_from_task_no_due_date(
        self, event_service, mock_repository, test_actor
    ):
        """Create event from task without due date."""
        task_no_due = {
            "id": str(uuid4()),
            "title": "Task Without Due Date",
            "status": "ready",
        }
        
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title="Task: Task Without Due Date",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="tenant-abc",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_task_link.return_value = {"id": str(uuid4())}
        
        result = await event_service.create_event_from_task(
            test_actor,
            task_no_due,
        )
        
        assert result.is_success


# ============================================================================
# JobTaskEventHandler Tests
# ============================================================================

class TestJobTaskEventHandler:
    """Tests for the JobTaskEventHandler."""
    
    @pytest.fixture
    def handler(self, event_service):
        """Create a JobTaskEventHandler."""
        return JobTaskEventHandler(
            event_service,
            auto_create_job_events=True,
            auto_create_task_events=True,
        )
    
    def test_subscribed_event_types(self, handler):
        """Verify handler subscribes to expected events."""
        types = handler.get_subscribed_event_types()
        
        assert "job.created" in types
        assert "job.updated" in types
        assert "job.completed" in types
        assert "task.created" in types
        assert "task.updated" in types
        assert "task.completed" in types
    
    @pytest.mark.asyncio
    async def test_handle_job_created_auto_creates_event(
        self, handler, event_service, mock_repository
    ):
        """Auto-create event when job.created is received."""
        job_id = str(uuid4())
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title="Job: Test Job",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="system",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_job_link.return_value = {"id": str(uuid4())}
        
        event = DomainEvent.create(
            event_type="job.created",
            entity_id=job_id,
            tenant_id="tenant-abc",
            actor="system",
            metadata={
                "name": "Test Job",
                "scheduled_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        await handler.handle_event(event)
        
        mock_repository.create_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_job_created_skips_without_schedule(
        self, handler, mock_repository
    ):
        """Skip auto-creation when job has no schedule."""
        event = DomainEvent.create(
            event_type="job.created",
            entity_id=str(uuid4()),
            tenant_id="tenant-abc",
            actor="system",
            metadata={"name": "Unscheduled Job"},
        )
        
        await handler.handle_event(event)
        
        mock_repository.create_event.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_handle_task_created_auto_creates_event(
        self, handler, event_service, mock_repository
    ):
        """Auto-create event when task.created has due date."""
        task_id = str(uuid4())
        created_event = CalendarEvent(
            event_id=str(uuid4()),
            title="Task: Test Task",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            status=EventStatus.CONFIRMED,
            visibility=EventVisibility.PRIVATE,
            busy_status=BusyStatus.BUSY,
            timezone_name="UTC",
            tenant_id="system",
        )
        
        mock_repository.create_event.return_value = created_event
        mock_repository.get_event_by_id.return_value = created_event
        mock_repository.create_task_link.return_value = {"id": str(uuid4())}
        
        event = DomainEvent.create(
            event_type="task.created",
            entity_id=task_id,
            tenant_id="tenant-abc",
            actor="system",
            metadata={
                "title": "Test Task",
                "due_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            },
        )
        
        await handler.handle_event(event)
        
        mock_repository.create_event.assert_called_once()
    
    def test_job_status_mapping(self, handler):
        """Verify job status to calendar status mapping."""
        assert handler.JOB_STATUS_MAP["draft"] == "tentative"
        assert handler.JOB_STATUS_MAP["scheduled"] == "confirmed"
        assert handler.JOB_STATUS_MAP["running"] == "confirmed"
        assert handler.JOB_STATUS_MAP["cancelled"] == "cancelled"
    
    def test_task_status_mapping(self, handler):
        """Verify task status to calendar status mapping."""
        assert handler.TASK_STATUS_MAP["draft"] == "tentative"
        assert handler.TASK_STATUS_MAP["done"] == "confirmed"
        assert handler.TASK_STATUS_MAP["cancelled"] == "cancelled"


class TestCreateJobTaskHandlerFactory:
    """Tests for the factory function."""
    
    def test_create_handler_default_options(self, event_service):
        """Create handler with default options."""
        handler = create_job_task_handler(event_service)
        
        assert handler._auto_create_job_events is False
        assert handler._auto_create_task_events is False
    
    def test_create_handler_with_auto_create(self, event_service):
        """Create handler with auto-create enabled."""
        handler = create_job_task_handler(
            event_service,
            auto_create_job_events=True,
            auto_create_task_events=True,
        )
        
        assert handler._auto_create_job_events is True
        assert handler._auto_create_task_events is True
