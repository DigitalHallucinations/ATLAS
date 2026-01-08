"""Integration between calendar events and job/task lifecycle.

Subscribes to job and task domain events and automatically updates
linked calendar events when the source entities change.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.services.common.messaging import DomainEventSubscriber
from core.services.common.types import DomainEvent
from core.services.common import Actor

from modules.calendar_store.link_models import LinkType, SyncBehavior

if TYPE_CHECKING:
    from .event_service import CalendarEventService

logger = logging.getLogger(__name__)


class JobTaskEventHandler(DomainEventSubscriber):
    """
    Handles job and task domain events to update linked calendar events.
    
    Subscribed events:
    - job.created: Optionally auto-create calendar events for new jobs
    - job.updated: Update linked event details (title, description)
    - job.status_changed: Update event status/styling based on job state
    - job.scheduled: Create or update event timing
    - job.completed: Mark linked events as complete
    - job.cancelled: Update or remove linked events
    
    - task.created: Optionally auto-create calendar events for tasks with due dates
    - task.updated: Update linked event details
    - task.status_changed: Update event status
    - task.completed: Mark linked events as complete
    - task.cancelled: Update or remove linked events
    
    Example:
        handler = JobTaskEventHandler(
            event_service=calendar_service,
            auto_create_job_events=True,
            auto_create_task_events=True,
        )
        
        # Register with the agent bus
        await agent_bus.subscribe(TASK_CREATE.name, handler.handle_agent_message)
        await agent_bus.subscribe(TASK_UPDATE.name, handler.handle_agent_message)
    """
    
    # Job status to calendar event status mapping
    JOB_STATUS_MAP = {
        "draft": "tentative",
        "scheduled": "confirmed",
        "running": "confirmed",
        "succeeded": "confirmed",
        "failed": "cancelled",
        "cancelled": "cancelled",
    }
    
    # Task status to calendar event status mapping
    TASK_STATUS_MAP = {
        "draft": "tentative",
        "ready": "confirmed",
        "in_progress": "confirmed",
        "review": "confirmed",
        "done": "confirmed",
        "cancelled": "cancelled",
    }
    
    def __init__(
        self,
        event_service: "CalendarEventService",
        *,
        auto_create_job_events: bool = False,
        auto_create_task_events: bool = False,
        default_job_category_id: Optional[str] = None,
        default_task_category_id: Optional[str] = None,
        system_actor: Optional[Actor] = None,
    ) -> None:
        """
        Initialize the job/task event handler.
        
        Parameters
        ----------
        event_service
            The calendar event service to use for operations
        auto_create_job_events
            If True, automatically create calendar events for new jobs
        auto_create_task_events
            If True, automatically create calendar events for tasks with due dates
        default_job_category_id
            Default category for auto-created job events
        default_task_category_id
            Default category for auto-created task events
        system_actor
            Actor to use for system-initiated operations
        """
        super().__init__()
        self._event_service = event_service
        self._auto_create_job_events = auto_create_job_events
        self._auto_create_task_events = auto_create_task_events
        self._default_job_category_id = default_job_category_id
        self._default_task_category_id = default_task_category_id
        self._system_actor = system_actor or Actor(
            type="system",
            id="system",
            tenant_id="system",
            permissions=["calendar.write", "calendar.read"],
        )
    
    def get_subscribed_event_types(self) -> List[str]:
        """Return event types this handler subscribes to."""
        return [
            # Job events
            "job.created",
            "job.updated",
            "job.status_changed",
            "job.scheduled",
            "job.completed",
            "job.cancelled",
            # Task events
            "task.created",
            "task.updated",
            "task.status_changed",
            "task.completed",
            "task.cancelled",
        ]
    
    async def handle_event(self, event: DomainEvent) -> None:
        """
        Handle a domain event and update linked calendar events.
        
        Parameters
        ----------
        event
            The domain event to handle
        """
        try:
            event_type = event.event_type
            entity_id = str(event.entity_id)
            # DomainEvent uses 'metadata' for additional event data
            payload = event.metadata or {}
            
            logger.debug(
                f"Handling {event_type} for entity {entity_id}",
                extra={"event_type": event_type, "entity_id": entity_id}
            )
            
            # Route to appropriate handler
            if event_type == "job.created":
                await self._handle_job_created(entity_id, payload)
            elif event_type == "job.updated":
                await self._handle_job_updated(entity_id, payload)
            elif event_type == "job.status_changed":
                await self._handle_job_status_changed(entity_id, payload)
            elif event_type == "job.scheduled":
                await self._handle_job_scheduled(entity_id, payload)
            elif event_type in ("job.completed", "job.cancelled"):
                await self._handle_job_finished(entity_id, payload, event_type)
            elif event_type == "task.created":
                await self._handle_task_created(entity_id, payload)
            elif event_type == "task.updated":
                await self._handle_task_updated(entity_id, payload)
            elif event_type == "task.status_changed":
                await self._handle_task_status_changed(entity_id, payload)
            elif event_type in ("task.completed", "task.cancelled"):
                await self._handle_task_finished(entity_id, payload, event_type)
                
        except Exception as e:
            logger.error(
                f"Error handling {event.event_type}: {e}",
                extra={"event_type": event.event_type, "entity_id": str(event.entity_id)},
                exc_info=True
            )
    
    # ========================================================================
    # Job event handlers
    # ========================================================================
    
    async def _handle_job_created(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Handle job.created event."""
        if not self._auto_create_job_events:
            return
        
        # Only auto-create if job has scheduling info
        if not payload.get("scheduled_at") and not payload.get("next_run_at"):
            logger.debug(f"Skipping auto-create for job {job_id}: no schedule")
            return
        
        job_data = {
            "id": job_id,
            "name": payload.get("name", "Unnamed Job"),
            "description": payload.get("description"),
            "scheduled_at": payload.get("scheduled_at") or payload.get("next_run_at"),
            "estimated_duration_minutes": payload.get("estimated_duration_minutes", 60),
            "status": payload.get("status", "scheduled"),
        }
        
        overrides = {}
        if self._default_job_category_id:
            overrides["category_id"] = self._default_job_category_id
        
        result = await self._event_service.create_event_from_job(
            self._system_actor,
            job_data,
            event_overrides=overrides if overrides else None,
            link_type=LinkType.AUTO_CREATED,
            sync_behavior=SyncBehavior.FROM_SOURCE,
        )
        
        if result.is_failure:
            logger.warning(f"Failed to auto-create event for job {job_id}: {result.error}")
        else:
            logger.info(f"Auto-created calendar event for job {job_id}")
    
    async def _handle_job_updated(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Handle job.updated event - update linked calendar events."""
        await self._update_linked_events_for_job(
            job_id,
            payload,
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _handle_job_status_changed(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Handle job.status_changed event - update event status/styling."""
        new_status = payload.get("new_status") or payload.get("status")
        if not new_status:
            return
        
        calendar_status = self.JOB_STATUS_MAP.get(str(new_status).lower(), "confirmed")
        
        await self._update_linked_events_for_job(
            job_id,
            {"status": calendar_status, "job_status": new_status},
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _handle_job_scheduled(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Handle job.scheduled event - update or create event timing."""
        # Check if we already have a linked event
        result = await self._event_service.get_events_for_job(self._system_actor, job_id)
        
        if result.is_success and result.value:
            # Update existing event timing
            scheduled_at = payload.get("scheduled_at") or payload.get("next_run_at")
            if scheduled_at:
                await self._update_linked_events_for_job(
                    job_id,
                    {"scheduled_at": scheduled_at},
                    sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
                )
        elif self._auto_create_job_events:
            # Create new event
            await self._handle_job_created(job_id, payload)
    
    async def _handle_job_finished(
        self, job_id: str, payload: Dict[str, Any], event_type: str
    ) -> None:
        """Handle job.completed or job.cancelled event."""
        is_cancelled = event_type == "job.cancelled"
        
        updates = {
            "status": "cancelled" if is_cancelled else "confirmed",
            "job_status": "cancelled" if is_cancelled else "succeeded",
        }
        
        # Add completion time to metadata
        if payload.get("finished_at"):
            updates["finished_at"] = payload["finished_at"]
        
        await self._update_linked_events_for_job(
            job_id,
            updates,
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _update_linked_events_for_job(
        self,
        job_id: str,
        updates: Dict[str, Any],
        sync_behaviors: List[SyncBehavior],
    ) -> None:
        """Update all calendar events linked to a job with specified sync behaviors."""
        # Get linked events
        result = await self._event_service.get_events_for_job(self._system_actor, job_id)
        if result.is_failure or not result.value:
            return
        
        # Get links to check sync behavior
        for event in result.value:
            links_result = await self._event_service.get_linked_jobs(
                self._system_actor, event.event_id
            )
            if links_result.is_failure or links_result.value is None:
                continue
            
            # Find the link for this job
            for link in links_result.value:
                if str(link.get("job_id")) == job_id:
                    link_sync = link.get("sync_behavior")
                    if link_sync in [b.value for b in sync_behaviors]:
                        await self._apply_updates_to_event(event.event_id, updates)
                    break
    
    # ========================================================================
    # Task event handlers
    # ========================================================================
    
    async def _handle_task_created(self, task_id: str, payload: Dict[str, Any]) -> None:
        """Handle task.created event."""
        if not self._auto_create_task_events:
            return
        
        # Only auto-create if task has a due date
        if not payload.get("due_at"):
            logger.debug(f"Skipping auto-create for task {task_id}: no due date")
            return
        
        task_data = {
            "id": task_id,
            "title": payload.get("title", "Unnamed Task"),
            "description": payload.get("description"),
            "due_at": payload.get("due_at"),
            "estimated_minutes": payload.get("estimated_minutes", 60),
            "status": payload.get("status", "ready"),
            "priority": payload.get("priority", 0),
        }
        
        overrides = {}
        if self._default_task_category_id:
            overrides["category_id"] = self._default_task_category_id
        
        result = await self._event_service.create_event_from_task(
            self._system_actor,
            task_data,
            event_overrides=overrides if overrides else None,
            link_type=LinkType.DEADLINE,
            sync_behavior=SyncBehavior.FROM_SOURCE,
        )
        
        if result.is_failure:
            logger.warning(f"Failed to auto-create event for task {task_id}: {result.error}")
        else:
            logger.info(f"Auto-created calendar event for task {task_id}")
    
    async def _handle_task_updated(self, task_id: str, payload: Dict[str, Any]) -> None:
        """Handle task.updated event - update linked calendar events."""
        await self._update_linked_events_for_task(
            task_id,
            payload,
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _handle_task_status_changed(self, task_id: str, payload: Dict[str, Any]) -> None:
        """Handle task.status_changed event - update event status/styling."""
        new_status = payload.get("new_status") or payload.get("status")
        if not new_status:
            return
        
        calendar_status = self.TASK_STATUS_MAP.get(str(new_status).lower(), "confirmed")
        
        await self._update_linked_events_for_task(
            task_id,
            {"status": calendar_status, "task_status": new_status},
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _handle_task_finished(
        self, task_id: str, payload: Dict[str, Any], event_type: str
    ) -> None:
        """Handle task.completed or task.cancelled event."""
        is_cancelled = event_type == "task.cancelled"
        
        updates = {
            "status": "cancelled" if is_cancelled else "confirmed",
            "task_status": "cancelled" if is_cancelled else "done",
        }
        
        await self._update_linked_events_for_task(
            task_id,
            updates,
            sync_behaviors=[SyncBehavior.FROM_SOURCE, SyncBehavior.BIDIRECTIONAL],
        )
    
    async def _update_linked_events_for_task(
        self,
        task_id: str,
        updates: Dict[str, Any],
        sync_behaviors: List[SyncBehavior],
    ) -> None:
        """Update all calendar events linked to a task with specified sync behaviors."""
        # Get linked events
        result = await self._event_service.get_events_for_task(self._system_actor, task_id)
        if result.is_failure or not result.value:
            return
        
        # Get links to check sync behavior
        for event in result.value:
            links_result = await self._event_service.get_linked_tasks(
                self._system_actor, event.event_id
            )
            if links_result.is_failure or links_result.value is None:
                continue
            
            # Find the link for this task
            for link in links_result.value:
                if str(link.get("task_id")) == task_id:
                    link_sync = link.get("sync_behavior")
                    if link_sync in [b.value for b in sync_behaviors]:
                        await self._apply_updates_to_event(event.event_id, updates)
                    break
    
    # ========================================================================
    # Common helpers
    # ========================================================================
    
    async def _apply_updates_to_event(
        self, event_id: str, updates: Dict[str, Any]
    ) -> None:
        """Apply updates to a calendar event."""
        from .types import CalendarEventUpdate
        
        # Build update object from updates dict
        update_fields = {}
        metadata_updates = {}
        
        for key, value in updates.items():
            if key in ("title", "description", "location", "status"):
                update_fields[key] = value
            elif key in ("start_time", "end_time", "scheduled_at"):
                if key == "scheduled_at":
                    # Convert scheduled_at to start_time
                    if isinstance(value, str):
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    update_fields["start_time"] = value
                    # Also update end_time to maintain duration (default 1 hour)
                    update_fields["end_time"] = value + timedelta(hours=1)
                else:
                    update_fields[key] = value
            else:
                # Put in metadata
                metadata_updates[key] = value
        
        if metadata_updates:
            update_fields["metadata"] = metadata_updates
        
        if not update_fields:
            return
        
        event_update = CalendarEventUpdate(**update_fields)
        
        result = await self._event_service.update_event(
            self._system_actor, event_id, event_update
        )
        
        if result.is_failure:
            logger.warning(f"Failed to update event {event_id}: {result.error}")
        else:
            logger.debug(f"Updated calendar event {event_id}")


def create_job_task_handler(
    event_service: "CalendarEventService",
    *,
    auto_create_job_events: bool = False,
    auto_create_task_events: bool = False,
) -> JobTaskEventHandler:
    """
    Factory function to create a JobTaskEventHandler.
    
    Parameters
    ----------
    event_service
        The calendar event service to use
    auto_create_job_events
        Whether to auto-create events for new scheduled jobs
    auto_create_task_events
        Whether to auto-create events for tasks with due dates
        
    Returns
    -------
    Configured JobTaskEventHandler instance
    """
    return JobTaskEventHandler(
        event_service,
        auto_create_job_events=auto_create_job_events,
        auto_create_task_events=auto_create_task_events,
    )


__all__ = [
    "JobTaskEventHandler",
    "create_job_task_handler",
]
