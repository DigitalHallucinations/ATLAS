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
    
    async def update_reminder(self, reminder_id: str, updates: Dict[str, Any]) -> CalendarReminder:
        if reminder_id not in self.reminders:
            raise ValueError(f"Reminder {reminder_id} not found")
        reminder = self.reminders[reminder_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(reminder, key):
                setattr(reminder, key, value)
        
        return reminder
    
    async def get_reminder(self, reminder_id: str) -> CalendarReminder:
        if reminder_id not in self.reminders:
            raise ValueError(f"Reminder {reminder_id} not found")
        return self.reminders[reminder_id]
    
    async def list_due_reminders(self, as_of: datetime) -> List[CalendarReminder]:
        return [
            reminder for reminder in self.reminders.values()
            if reminder.is_due(as_of) and reminder.status == ReminderStatus.SCHEDULED
        ]
    
    async def get_event(self, event_id: str) -> CalendarEvent:
        if event_id not in self.events:
            raise ValueError(f"Event {event_id} not found")
        return self.events[event_id]


class MockNotificationService:
    """Mock notification service for testing."""
    
    def __init__(self):
        self.sent_notifications: List[Dict[str, Any]] = []
    
    async def send_notification(self, reminder: CalendarReminder, event: CalendarEvent) -> bool:
        self.sent_notifications.append({
            "reminder_id": reminder.reminder_id,
            "event_id": reminder.event_id,
            "method": reminder.method,
            "message": f"Reminder: {event.title}",
        })
        return True


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
    async def test_cancel_reminder(self, reminder_service, mock_context, mock_event):
        """Test reminder cancellation."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Schedule a reminder first
        reminder_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        reminder = await reminder_service.schedule_reminder(
            context=mock_context,
            event_id=mock_event.event_id,
            reminder_time=reminder_time,
            method=ReminderMethod.NOTIFICATION,
        )
        
        # Cancel the reminder
        cancelled = await reminder_service.cancel_reminder(
            context=mock_context,
            reminder_id=reminder.id
        )
        
        assert cancelled.status == ReminderStatus.CANCELLED
        
        # Check that cancellation event was published
        events = reminder_service.event_publisher.published_events
        assert len(events) == 2  # Schedule + Cancel
        cancel_events = [e for e in events if hasattr(e, 'status') and e.status == ReminderStatus.CANCELLED]
        assert len(cancel_events) == 1

    @pytest.mark.asyncio
    async def test_snooze_reminder(self, reminder_service, mock_context, mock_event):
        """Test reminder snoozing."""
        # Add the event to the repository
        reminder_service.repository.events[mock_event.event_id] = mock_event
        
        # Schedule a reminder first
        reminder_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        reminder = await reminder_service.schedule_reminder(
            context=mock_context,
            event_id=mock_event.event_id,
            reminder_time=reminder_time,
            method=ReminderMethod.NOTIFICATION,
        )
        
        # Trigger the reminder (set status to triggered)
        await reminder_service.repository.update_reminder(
            reminder.id, 
            {"status": ReminderStatus.TRIGGERED}
        )
        
        # Snooze the reminder
        snooze_duration = timedelta(minutes=15)
        snoozed = await reminder_service.snooze_reminder(
            context=mock_context,
            reminder_id=reminder.id,
            snooze_duration=snooze_duration
        )
        
        assert snoozed.status == ReminderStatus.SNOOZED
        assert snoozed.snooze_until is not None
        expected_snooze_time = datetime.now(timezone.utc) + snooze_duration
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
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.reminders[due_reminder.reminder_id] = due_reminder
        
        # Process reminders
        with patch.object(reminder_service, "_create_execution_context", return_value=mock_context):
            processed_count = await reminder_service.process_reminders()
        
        assert processed_count == 1
        
        # Check that reminder was triggered and notification sent
        updated_reminder = reminder_service.repository.reminders[due_reminder.reminder_id]
        assert updated_reminder.status == ReminderStatus.DELIVERED
        
        # Check notification was sent
        notifications = reminder_service.notification_service.sent_notifications
        assert len(notifications) == 1
        assert notifications[0]["reminder_id"] == due_reminder.reminder_id
        
        # Check events were published
        events = reminder_service.event_publisher.published_events
        trigger_events = [e for e in events if isinstance(e, ReminderTriggered)]
        deliver_events = [e for e in events if isinstance(e, ReminderDelivered)]
        assert len(trigger_events) == 1
        assert len(deliver_events) == 1

    @pytest.mark.asyncio
    async def test_background_processing_lifecycle(self, reminder_service):
        """Test background processing start and stop."""
        assert not reminder_service.is_running
        
        # Start background processing
        await reminder_service.start_background_processing()
        assert reminder_service.is_running
        
        # Wait a bit to ensure processing loop is running
        await asyncio.sleep(0.1)
        
        # Stop background processing
        await reminder_service.stop_background_processing()
        assert not reminder_service.is_running

    @pytest.mark.asyncio
    async def test_reminder_retry_logic(self, reminder_service, mock_context):
        """Test reminder retry logic on delivery failure."""
        # Mock notification service to fail initially
        call_count = 0
        async def failing_send_notification(reminder, event):
            nonlocal call_count
            call_count += 1
            # Fail first two attempts, succeed on third
            return call_count >= 3
        
        reminder_service.notification_service.send_notification = failing_send_notification
        
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
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.events[event.event_id] = event
        reminder_service.repository.reminders[reminder.reminder_id] = reminder
        
        # Process reminder with retries
        with patch.object(reminder_service, "_create_execution_context", return_value=mock_context):
            await reminder_service._deliver_reminder(reminder, event, mock_context)
        
        # Should eventually succeed after retries
        updated_reminder = reminder_service.repository.reminders[reminder.reminder_id]
        assert updated_reminder.status == ReminderStatus.DELIVERED
        assert updated_reminder.retry_count == 2  # Two failed attempts before success

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
            snooze_until=datetime.now(timezone.utc) - timedelta(minutes=1),  # Snooze expired
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        reminder_service.repository.events[event.event_id] = event
        reminder_service.repository.reminders[snoozed_reminder.reminder_id] = snoozed_reminder
        
        # Process reminders - should reactivate the snoozed reminder
        with patch.object(reminder_service, "_create_execution_context", return_value=mock_context):
            processed_count = await reminder_service.process_reminders()
        
        assert processed_count == 1
        
        # Check that reminder was delivered
        updated_reminder = reminder_service.repository.reminders[snoozed_reminder.reminder_id]
        assert updated_reminder.status == ReminderStatus.DELIVERED
        assert updated_reminder.snooze_until is None  # Cleared on delivery

    def test_default_message_generation(self, reminder_service):
        """Test default reminder message generation."""
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
        
        reminder = CalendarReminder(
            reminder_id=str(uuid.uuid4()),
            event_id=event.event_id,
            scheduled_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            method=ReminderMethod.NOTIFICATION,
            status=ReminderStatus.SCHEDULED,
            created_by="test-user",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        message = reminder_service._generate_reminder_message(reminder, event)
        
        assert "Team Standup" in message
        assert "Daily standup meeting" in message
        assert message.startswith("Reminder:")