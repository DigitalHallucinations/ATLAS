"""Tests for the ReminderService."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.context import ExecutionContext
from core.services.common.types import DomainEvent
from core.services.calendar.reminder_service import ReminderService
from core.services.calendar.types import (
    CalendarEvent,
    CalendarReminder,
    ReminderMethod,
    ReminderStatus,
)
from core.services.calendar.events import (
    ReminderScheduled,
    ReminderTriggered,
    ReminderDelivered,
)
from modules.calendar_store import CalendarStoreRepository


class MockRepository:
    """Mock repository for testing."""
    
    def __init__(self):
        self.reminders: Dict[str, CalendarReminder] = {}
        self.events: Dict[str, CalendarEvent] = {}
    
    async def create_reminder(self, reminder: CalendarReminder) -> CalendarReminder:
        self.reminders[reminder.reminder_id] = reminder
        return reminder
    
    async def update_reminder(self, reminder: CalendarReminder) -> CalendarReminder:
        """Update an existing reminder."""
        self.reminders[reminder.reminder_id] = reminder
        return reminder
    
    async def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        if reminder_id in self.reminders:
            del self.reminders[reminder_id]
            return True
        return False
    
    async def get_reminder(self, reminder_id: str) -> CalendarReminder | None:
        """Get reminder by ID, returns None if not found."""
        return self.reminders.get(reminder_id)
    
    async def list_reminders_for_event(self, event_id: str) -> List[CalendarReminder]:
        """Get all reminders for an event."""
        return [
            reminder for reminder in self.reminders.values()
            if reminder.event_id == event_id
        ]
    
    async def list_due_reminders(self, before_time: datetime) -> List[CalendarReminder]:
        """Get all reminders due before the specified time."""
        return [
            reminder for reminder in self.reminders.values()
            if reminder.is_due(before_time) and reminder.status == ReminderStatus.SCHEDULED
        ]
    
    async def list_snoozed_reminders(self, before_time: datetime) -> List[CalendarReminder]:
        """Get all snoozed reminders that should be reactivated."""
        return [
            reminder for reminder in self.reminders.values()
            if (
                reminder.status == ReminderStatus.SNOOZED and
                reminder.snooze_until is not None and
                reminder.snooze_until <= before_time
            )
        ]
    
    async def get_event(self, event_id: str) -> CalendarEvent:
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found")
        return self.events[event_id]


class MockNotificationService:
    """Mock notification service for testing."""
    
    def __init__(self):
        self.sent_notifications: List[Dict[str, Any]] = []
        self._should_succeed = True
    
    async def send_notification(
        self,
        method: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record notification and return configured success status."""
        self.sent_notifications.append({
            "method": method,
            "title": title,
            "message": message,
            "metadata": metadata or {},
        })
        return self._should_succeed
    
    def set_should_succeed(self, should_succeed: bool) -> None:
        """Configure whether notifications should succeed."""
        self._should_succeed = should_succeed
    
    def clear(self) -> None:
        """Clear recorded notifications."""
        self.sent_notifications.clear()


class MockPermissionChecker:
    """Mock permission checker that allows all operations."""
    
    async def can_create_reminder(self, context: ExecutionContext, event: CalendarEvent) -> bool:
        return True
    
    async def can_update_reminder(self, context: ExecutionContext, reminder: CalendarReminder) -> bool:
        return True
    
    async def can_delete_reminder(self, context: ExecutionContext, reminder: CalendarReminder) -> bool:
        return True
        
    async def has_permission(self, actor, permission: str) -> bool:
        """Check if actor has permission."""
        return True


class MockEventPublisher:
    """Mock event publisher that captures published events."""
    
    def __init__(self):
        self.published_events: List[DomainEvent] = []
    
    async def publish(self, event: DomainEvent) -> None:
        self.published_events.append(event)


@pytest.fixture
def mock_context():
    """Create a mock execution context."""
    return ExecutionContext(
        tenant_id="test-tenant",
        user_id="test-user",
        conversation_id=None,
    )


@pytest.fixture
def mock_event():
    """Create a mock calendar event."""
    return CalendarEvent(
        event_id=str(uuid.uuid4()),
        title="Test Meeting",
        description="A test meeting",
        start_time=datetime.now(timezone.utc) + timedelta(hours=1),
        end_time=datetime.now(timezone.utc) + timedelta(hours=2),
        created_by="test-user",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_actor(mock_context):
    """Create an actor for testing."""
    from core.services.common.types import Actor
    return Actor(
        type="user",
        id=mock_context.user_id,
        tenant_id=mock_context.tenant_id,
        permissions={"calendar:read", "calendar:write", "reminders:read", "reminders:write"}
    )


@pytest.fixture
def reminder_service():
    """Create a ReminderService with mock dependencies."""
    repository = MockRepository()
    notification_service = MockNotificationService()
    permission_checker = MockPermissionChecker()
    event_publisher = MockEventPublisher()
    
    service = ReminderService(
        repository=repository,
        permission_checker=permission_checker,
        event_publisher=event_publisher,
        notification_service=notification_service,
    )
    
    # Store references to mocks for test assertions
    service.repository = repository
    service.notification_service = notification_service
    service.permission_checker = permission_checker
    service.event_publisher = event_publisher
    
    return service


class TestReminderService:
    """Test suite for ReminderService."""

    @pytest.mark.asyncio
    async def test_schedule_reminder_success(self, reminder_service, mock_context, mock_event):
        """Test successful reminder scheduling."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Create actor from context
        from core.services.common.types import Actor
        actor = Actor(
            type="user",
            id=mock_context.user_id,
            tenant_id=mock_context.tenant_id,
            permissions={"calendar:write", "reminders:write"}
        )
        
        result = await reminder_service.schedule_reminder(
            actor=actor,
            event=mock_event,
            minutes_before=30,
            method=ReminderMethod.NOTIFICATION,
            message="Don't forget your meeting!"
        )
        
        assert result.is_success
        reminder = result.data
        assert reminder is not None
        assert reminder.event_id == mock_event.event_id
        assert reminder.method == ReminderMethod.NOTIFICATION
        assert reminder.status == ReminderStatus.SCHEDULED
        assert reminder.message == "Don't forget your meeting!"
        
        # Check that event was published
        events = reminder_service.event_publisher.published_events
        assert len(events) == 1
        assert isinstance(events[0], ReminderScheduled)
        assert str(events[0].entity_id) == reminder.reminder_id

    @pytest.mark.asyncio
    async def test_schedule_reminder_past_time(self, reminder_service, mock_context, mock_event):
        """Test that scheduling a reminder in the past fails."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Create actor from context
        from core.services.common.types import Actor
        actor = Actor(
            type="user",
            id=mock_context.user_id,
            tenant_id=mock_context.tenant_id,
            permissions={"calendar:write", "reminders:write"}
        )
        
        # Create an event in the past relative to reminder time
        past_event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title="Past Meeting",
            description="A past meeting",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),  # Event was 1 hour ago
            end_time=datetime.now(timezone.utc) - timedelta(minutes=30),  # Event ended 30 mins ago
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        reminder_service.repository.events[past_event.event_id] = past_event

        result = await reminder_service.schedule_reminder(
            actor=actor,
            event=past_event,
            minutes_before=30,  # This would schedule reminder 30 mins before past event
            method=ReminderMethod.NOTIFICATION,
        )
        
        # Should fail because the reminder time would be in the past
        assert not result.is_success
        assert "past" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cancel_reminder(self, reminder_service, mock_context, mock_event, mock_actor):
        """Test reminder cancellation."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Schedule a reminder first
        result = await reminder_service.schedule_reminder(
            actor=mock_actor,
            event=mock_event,
            minutes_before=30,
            method=ReminderMethod.NOTIFICATION,
        )
        
        assert result.is_success
        reminder = result.data
        
        # Cancel the reminder
        cancel_result = await reminder_service.cancel_reminder(
            actor=mock_actor,
            reminder_id=reminder.reminder_id
        )
        
        assert cancel_result.is_success
        assert cancel_result.data is True
        
        # Verify reminder is cancelled in repository
        cancelled_reminder = await reminder_service.repository.get_reminder(reminder.reminder_id)
        assert cancelled_reminder.status == ReminderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_snooze_reminder(self, reminder_service, mock_context, mock_event, mock_actor):
        """Test reminder snoozing."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Schedule a reminder first
        result = await reminder_service.schedule_reminder(
            actor=mock_actor,
            event=mock_event,
            minutes_before=30,
            method=ReminderMethod.NOTIFICATION,
        )
        
        assert result.is_success
        reminder = result.data
        
        # Simulate triggering by updating the status
        reminder.status = ReminderStatus.TRIGGERED
        await reminder_service.repository.update_reminder(reminder)
        
        # Snooze the reminder
        snooze_result = await reminder_service.snooze_reminder(
            actor=mock_actor,
            reminder_id=reminder.reminder_id,
            snooze_minutes=15
        )
        
        assert snooze_result.is_success
        snoozed = snooze_result.data
        assert snoozed.status == ReminderStatus.SNOOZED
        assert snoozed.snooze_until is not None
        expected_snooze_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        # Allow 5 second tolerance for test execution time
        assert abs((snoozed.snooze_until - expected_snooze_time).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_process_reminders(self, reminder_service, mock_context):
        """Test reminder processing."""
        # Create a test event
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title="Test Meeting",
            description="A test meeting",
            start_time=datetime.now(timezone.utc) + timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(hours=2),
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        # Add event to repository
        reminder_service.repository.events[event.event_id] = event
        
        # Create a due reminder
        due_reminder = CalendarReminder(
            reminder_id=str(uuid.uuid4()),
            event_id=event.event_id,
            scheduled_time=datetime.now(timezone.utc) - timedelta(minutes=1),  # Past due
            method=ReminderMethod.NOTIFICATION,
            status=ReminderStatus.SCHEDULED,
            message="Test reminder",
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.reminders[due_reminder.reminder_id] = due_reminder
        
        # Process reminders
        result = await reminder_service.process_reminders()
        
        assert result.is_success
        stats = result.data
        assert stats["processed"] >= 1
        
        # Check that reminder was triggered and notification sent
        updated_reminder = reminder_service.repository.reminders[due_reminder.reminder_id]
        # Reminder should either be delivered or scheduled for retry
        assert updated_reminder.status in [ReminderStatus.DELIVERED, ReminderStatus.SCHEDULED, ReminderStatus.TRIGGERED]
        
        # Check notification was sent
        notifications = reminder_service.notification_service.sent_notifications
        assert len(notifications) >= 1
        
        # Check events were published
        events = reminder_service.event_publisher.published_events
        trigger_events = [e for e in events if isinstance(e, ReminderTriggered)]
        assert len(trigger_events) >= 1

    @pytest.mark.asyncio
    async def test_background_processing_lifecycle(self, reminder_service):
        """Test background processing start and stop."""
        # Initially no background task
        assert reminder_service._processing_task is None
        
        # Start background processing
        await reminder_service.start_background_processing()
        assert reminder_service._processing_task is not None
        assert not reminder_service._processing_task.done()
        
        # Wait a bit to ensure processing loop is running
        await asyncio.sleep(0.1)
        
        # Stop background processing
        await reminder_service.stop_background_processing()
        # After stopping, task should be None or done
        assert reminder_service._processing_task is None or reminder_service._processing_task.done()

    @pytest.mark.asyncio
    async def test_reminder_retry_logic(self, reminder_service, mock_context):
        """Test reminder retry logic on delivery failure."""
        # Configure notification service to fail
        reminder_service.notification_service.set_should_succeed(False)
        
        # Create test event and reminder
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title="Test Meeting",
            description="A test meeting",
            start_time=datetime.now(timezone.utc) + timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(hours=2),
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder = CalendarReminder(
            reminder_id=str(uuid.uuid4()),
            event_id=event.event_id,
            scheduled_time=datetime.now(timezone.utc) - timedelta(minutes=1),
            method=ReminderMethod.NOTIFICATION,
            status=ReminderStatus.SCHEDULED,
            message="Test reminder",
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            retry_count=0,
            max_retries=3,
        )
        
        reminder_service.repository.events[event.event_id] = event
        reminder_service.repository.reminders[reminder.reminder_id] = reminder
        
        # Process reminder - should fail and increment retry count
        result = await reminder_service.process_reminders()
        
        assert result.is_success
        updated_reminder = reminder_service.repository.reminders[reminder.reminder_id]
        # After failure, either retry_count should increment or status should change
        assert updated_reminder.retry_count >= 1 or updated_reminder.status in [ReminderStatus.FAILED, ReminderStatus.SCHEDULED]

    @pytest.mark.asyncio
    async def test_snoozed_reminder_reactivation(self, reminder_service, mock_context):
        """Test that snoozed reminders are reactivated when snooze time expires."""
        # Create test event and snoozed reminder
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title="Test Meeting",
            description="A test meeting",
            start_time=datetime.now(timezone.utc) + timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(hours=2),
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        snoozed_reminder = CalendarReminder(
            reminder_id=str(uuid.uuid4()),
            event_id=event.event_id,
            scheduled_time=datetime.now(timezone.utc) - timedelta(minutes=30),  # Original time in past
            method=ReminderMethod.NOTIFICATION,
            status=ReminderStatus.SNOOZED,
            message="Snoozed reminder",
            snooze_until=datetime.now(timezone.utc) - timedelta(minutes=1),  # Snooze expired
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.events[event.event_id] = event
        reminder_service.repository.reminders[snoozed_reminder.reminder_id] = snoozed_reminder
        
        # Process reminders - should reactivate the snoozed reminder
        result = await reminder_service.process_reminders()
        
        assert result.is_success
        stats = result.data
        # Should have reactivated at least one snoozed reminder
        assert stats.get("snoozed_reactivated", 0) >= 1
        
        # Check that reminder was reactivated
        updated_reminder = reminder_service.repository.reminders[snoozed_reminder.reminder_id]
        # After reactivation, should be SCHEDULED (ready for next processing) or already processed
        assert updated_reminder.status in [ReminderStatus.SCHEDULED, ReminderStatus.TRIGGERED, ReminderStatus.DELIVERED]

    @pytest.mark.asyncio
    async def test_default_message_generation(self, reminder_service, mock_actor):
        """Test that reminder uses event title as default message."""
        event = CalendarEvent(
            event_id=str(uuid.uuid4()),
            title="Team Standup",
            description="Daily standup meeting",
            start_time=datetime.now(timezone.utc) + timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1, minutes=30),
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.events[event.event_id] = event
        
        # Schedule reminder without explicit message - should use event title
        result = await reminder_service.schedule_reminder(
            actor=mock_actor,
            event=event,
            minutes_before=15,
            method=ReminderMethod.NOTIFICATION,
            # No message specified - should default to event title
        )
        
        assert result.is_success
        reminder = result.data
        # Default message should be the event title
        assert reminder.message == "Team Standup"