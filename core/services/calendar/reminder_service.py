"""
Calendar reminder service.

Handles scheduling, triggering, and delivery of calendar event reminders
with support for multiple notification methods and retry logic.

Author: ATLAS Team
Date: Jan 7, 2026
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol
from uuid import uuid4

from core.services.common import (
    Actor,
    BusinessRuleError,
    DomainEvent,
    DomainEventPublisher,
    NotFoundError,
    OperationResult,
    PermissionChecker,
    PermissionDeniedError,
    Service,
    ServiceError,
    ValidationError,
)

from .events import ReminderScheduled, ReminderTriggered, ReminderDelivered
from .types import CalendarEvent, CalendarReminder, ReminderMethod, ReminderStatus


logger = logging.getLogger(__name__)


class ReminderRepository(Protocol):
    """Repository interface for reminder persistence."""
    
    async def create_reminder(self, reminder: CalendarReminder) -> CalendarReminder:
        """Create a new reminder."""
        ...
    
    async def get_reminder(self, reminder_id: str) -> CalendarReminder | None:
        """Get reminder by ID."""
        ...
    
    async def update_reminder(self, reminder: CalendarReminder) -> CalendarReminder:
        """Update an existing reminder."""
        ...
    
    async def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        ...
    
    async def list_reminders_for_event(self, event_id: str) -> List[CalendarReminder]:
        """Get all reminders for an event."""
        ...
    
    async def list_due_reminders(self, before_time: datetime) -> List[CalendarReminder]:
        """Get all reminders due before the specified time."""
        ...
    
    async def list_snoozed_reminders(self, before_time: datetime) -> List[CalendarReminder]:
        """Get all snoozed reminders that should be reactivated."""
        ...


class NotificationService(Protocol):
    """Service interface for delivering notifications."""
    
    async def send_notification(
        self, 
        method: ReminderMethod,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a notification via the specified method."""
        ...


class ReminderService(Service):
    """Service for managing calendar event reminders."""
    
    def __init__(
        self,
        repository: ReminderRepository,
        permission_checker: PermissionChecker,
        event_publisher: DomainEventPublisher,
        notification_service: Optional[NotificationService] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the reminder service.
        
        Args:
            repository: Reminder persistence layer
            permission_checker: Permission validation service
            event_publisher: Domain event publisher for notifications
            notification_service: Service for delivering notifications
            logger: Service logger
        """
        self._repository = repository
        self._permission_checker = permission_checker
        self._event_publisher = event_publisher
        self._notification_service = notification_service
        self._logger = logger or logging.getLogger(__name__)
        
        # Background processing
        self._processing_task: Optional[asyncio.Task] = None
        self._processing_interval = 30  # seconds
        self._shutdown_event = asyncio.Event()
    
    async def schedule_reminder(
        self,
        actor: Actor,
        event: CalendarEvent,
        minutes_before: int = 15,
        method: ReminderMethod = ReminderMethod.NOTIFICATION,
        message: Optional[str] = None,
    ) -> OperationResult[CalendarReminder]:
        """Schedule a reminder for a calendar event.
        
        Args:
            actor: Actor scheduling the reminder
            event: Calendar event to remind about
            minutes_before: Minutes before event start to trigger reminder
            method: Delivery method for the reminder
            message: Custom reminder message (defaults to event title)
            
        Returns:
            OperationResult containing the created reminder or error
        """
        try:
            # Validate permissions
            await self._require_calendar_write_permission(actor)
            
            # Validate inputs
            validation_result = self._validate_reminder_request(
                event, minutes_before, method
            )
            if not validation_result.is_success:
                return OperationResult.failure(validation_result.error or "Validation failed")
            
            # Calculate scheduled time
            if not event.start_time:
                return OperationResult.failure(
                    "Cannot schedule reminder for event without start time"
                )
            
            scheduled_time = event.start_time - timedelta(minutes=minutes_before)
            
            # Create reminder
            reminder = CalendarReminder(
                reminder_id=str(uuid4()),
                event_id=event.event_id,
                method=method,
                minutes_before=minutes_before,
                message=message or event.title,
                status=ReminderStatus.SCHEDULED,
                scheduled_time=scheduled_time,
                tenant_id=actor.tenant_id,
                created_by=actor.id,
                created_at=datetime.now(timezone.utc),
            )
            
            # Persist reminder
            saved_reminder = await self._repository.create_reminder(reminder)
            
            # Publish domain event
            await self._event_publisher.publish(
                ReminderScheduled.create_for_reminder(
                    reminder_id=saved_reminder.reminder_id,
                    event_id=saved_reminder.event_id,
                    tenant_id=actor.tenant_id,
                    actor_type=actor.type,
                    scheduled_time=saved_reminder.scheduled_time or datetime.now(timezone.utc),
                    method=saved_reminder.method,
                )
            )
            
            self._logger.info(
                f"Reminder scheduled: {saved_reminder.reminder_id} for event {event.event_id}"
            )
            
            return OperationResult.success(saved_reminder)
            
        except Exception as e:
            self._logger.error(f"Failed to schedule reminder: {e}")
            return OperationResult.failure(f"Failed to schedule reminder: {e}")
    
    async def cancel_reminder(
        self,
        actor: Actor,
        reminder_id: str
    ) -> OperationResult[bool]:
        """Cancel a scheduled reminder.
        
        Args:
            actor: Actor cancelling the reminder
            reminder_id: ID of reminder to cancel
            
        Returns:
            OperationResult indicating success or failure
        """
        try:
            # Validate permissions
            await self._require_calendar_write_permission(actor)
            
            # Get existing reminder
            reminder = await self._repository.get_reminder(reminder_id)
            if not reminder:
                return OperationResult.failure(f"Reminder {reminder_id} not found")
            
            # Check if cancellable
            if reminder.status in [ReminderStatus.DELIVERED, ReminderStatus.CANCELLED]:
                return OperationResult.failure(
                    "Cannot cancel reminder that is already delivered or cancelled"
                )
            
            # Update status
            reminder.status = ReminderStatus.CANCELLED
            reminder.updated_at = datetime.now(timezone.utc)
            
            await self._repository.update_reminder(reminder)
            
            self._logger.info(f"Reminder cancelled: {reminder_id}")
            return OperationResult.success(True)
            
        except Exception as e:
            self._logger.error(f"Failed to cancel reminder: {e}")
            return OperationResult.failure(f"Failed to cancel reminder: {e}")
    
    async def snooze_reminder(
        self,
        actor: Actor,
        reminder_id: str,
        snooze_minutes: int = 5
    ) -> OperationResult[CalendarReminder]:
        """Snooze a triggered reminder for later delivery.
        
        Args:
            actor: Actor snoozing the reminder
            reminder_id: ID of reminder to snooze
            snooze_minutes: Minutes to snooze the reminder
            
        Returns:
            OperationResult containing updated reminder or error
        """
        try:
            # Validate permissions
            await self._require_calendar_write_permission(actor)
            
            # Get existing reminder
            reminder = await self._repository.get_reminder(reminder_id)
            if not reminder:
                return OperationResult.failure(f"Reminder {reminder_id} not found")
            
            # Check if snoozeable
            if reminder.status != ReminderStatus.TRIGGERED:
                return OperationResult.failure(
                    "Can only snooze triggered reminders"
                )
            
            # Update snooze time
            reminder.status = ReminderStatus.SNOOZED
            reminder.snooze_until = datetime.now(timezone.utc) + timedelta(minutes=snooze_minutes)
            reminder.updated_at = datetime.now(timezone.utc)
            
            updated_reminder = await self._repository.update_reminder(reminder)
            
            self._logger.info(f"Reminder snoozed: {reminder_id} for {snooze_minutes} minutes")
            return OperationResult.success(updated_reminder)
            
        except Exception as e:
            self._logger.error(f"Failed to snooze reminder: {e}")
            return OperationResult.failure(f"Failed to snooze reminder: {e}")
    
    async def process_reminders(self) -> OperationResult[Dict[str, int]]:
        """Process all due reminders.
        
        Returns:
            OperationResult with statistics about processed reminders
        """
        try:
            now = datetime.now(timezone.utc)
            stats = {
                "processed": 0,
                "delivered": 0,
                "failed": 0,
                "snoozed_reactivated": 0
            }
            
            # Process due reminders
            due_reminders = await self._repository.list_due_reminders(now)
            for reminder in due_reminders:
                result = await self._process_reminder(reminder)
                stats["processed"] += 1
                if result:
                    stats["delivered"] += 1
                else:
                    stats["failed"] += 1
            
            # Reactivate snoozed reminders
            snoozed_reminders = await self._repository.list_snoozed_reminders(now)
            for reminder in snoozed_reminders:
                reminder.status = ReminderStatus.SCHEDULED
                reminder.snooze_until = None
                reminder.updated_at = now
                await self._repository.update_reminder(reminder)
                stats["snoozed_reactivated"] += 1
            
            return OperationResult.success(stats)
            
        except Exception as e:
            self._logger.error(f"Failed to process reminders: {e}")
            return OperationResult.failure(f"Failed to process reminders: {e}")
    
    async def list_reminders_for_event(
        self,
        actor: Actor,
        event_id: str
    ) -> OperationResult[List[CalendarReminder]]:
        """List all reminders for a specific event.
        
        Args:
            actor: Actor requesting reminders
            event_id: ID of event to get reminders for
            
        Returns:
            OperationResult containing list of reminders
        """
        try:
            # Validate permissions
            await self._require_calendar_read_permission(actor)
            
            reminders = await self._repository.list_reminders_for_event(event_id)
            return OperationResult.success(reminders)
            
        except Exception as e:
            self._logger.error(f"Failed to list reminders: {e}")
            return OperationResult.failure(f"Failed to list reminders: {e}")
    
    async def start_background_processing(self) -> None:
        """Start background task for processing reminders."""
        if self._processing_task and not self._processing_task.done():
            return
        
        self._shutdown_event.clear()
        self._processing_task = asyncio.create_task(self._background_processor())
        self._logger.info("Started reminder background processing")
    
    async def stop_background_processing(self) -> None:
        """Stop background reminder processing."""
        if self._processing_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._processing_task.cancel()
            self._processing_task = None
            
        self._logger.info("Stopped reminder background processing")
    
    async def _background_processor(self) -> None:
        """Background task for processing reminders."""
        while not self._shutdown_event.is_set():
            try:
                await self.process_reminders()
            except Exception as e:
                self._logger.error(f"Error in background reminder processing: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), 
                    timeout=self._processing_interval
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Timeout means continue processing
    
    async def _process_reminder(self, reminder: CalendarReminder) -> bool:
        """Process a single reminder.
        
        Args:
            reminder: Reminder to process
            
        Returns:
            True if successfully delivered, False otherwise
        """
        try:
            # Mark as triggered
            reminder.status = ReminderStatus.TRIGGERED
            reminder.triggered_at = datetime.now(timezone.utc)
            await self._repository.update_reminder(reminder)
            
            # Publish triggered event
            await self._event_publisher.publish(
                ReminderTriggered.create_for_reminder(
                    reminder_id=reminder.reminder_id,
                    event_id=reminder.event_id,
                    tenant_id=reminder.tenant_id or "default",
                    actor_type="system",
                    method=reminder.method,
                    message=reminder.message or "",
                )
            )
            
            # Attempt delivery
            success = False
            if self._notification_service:
                success = await self._notification_service.send_notification(
                    method=reminder.method,
                    title=f"Event Reminder: {reminder.message}",
                    message=f"Your event starts in {reminder.minutes_before} minutes",
                    metadata={
                        "event_id": reminder.event_id,
                        "reminder_id": reminder.reminder_id,
                    }
                )
            
            if success:
                # Mark as delivered
                reminder.status = ReminderStatus.DELIVERED
                reminder.delivered_at = datetime.now(timezone.utc)
                
                # Publish delivered event
                await self._event_publisher.publish(
                    ReminderDelivered.create_for_reminder(
                        reminder_id=reminder.reminder_id,
                        event_id=reminder.event_id,
                        tenant_id=reminder.tenant_id or "default",
                        actor_type="system",
                        method=reminder.method,
                        delivery_success=True,
                    )
                )
            else:
                # Mark as failed, potentially retry
                reminder.retry_count += 1
                if reminder.can_retry():
                    reminder.status = ReminderStatus.SCHEDULED
                    reminder.scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=2)
                else:
                    reminder.status = ReminderStatus.FAILED
                
                # Publish delivered event with failure
                await self._event_publisher.publish(
                    ReminderDelivered.create_for_reminder(
                        reminder_id=reminder.reminder_id,
                        event_id=reminder.event_id,
                        tenant_id=reminder.tenant_id or "default",
                        actor_type="system",
                        method=reminder.method,
                        delivery_success=False,
                    )
                )
            
            await self._repository.update_reminder(reminder)
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to process reminder {reminder.reminder_id}: {e}")
            return False
    
    def _validate_reminder_request(
        self,
        event: CalendarEvent,
        minutes_before: int,
        method: ReminderMethod
    ) -> OperationResult[None]:
        """Validate reminder scheduling parameters."""
        errors = []
        
        if minutes_before < 0:
            errors.append("Minutes before must be non-negative")
        
        if minutes_before > 10080:  # 1 week
            errors.append("Reminder cannot be scheduled more than 1 week in advance")
        
        if not event.start_time:
            errors.append("Event must have a start time")
        
        if event.start_time and event.start_time <= datetime.now(timezone.utc):
            errors.append("Cannot schedule reminder for past events")
        
        if errors:
            return OperationResult.failure("; ".join(errors))
        
        return OperationResult.success(None)
    
    async def _require_calendar_read_permission(self, actor: Actor) -> None:
        """Require calendar read permission."""
        has_permission = await self._permission_checker.has_permission(
            actor, "calendar:read"
        )
        if not has_permission:
            raise PermissionDeniedError("Calendar read permission required")
    
    async def _require_calendar_write_permission(self, actor: Actor) -> None:
        """Require calendar write permission."""
        has_permission = await self._permission_checker.has_permission(
            actor, "calendar:write"
        )
        if not has_permission:
            raise PermissionDeniedError("Calendar write permission required")