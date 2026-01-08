"""Calendar event service implementation aligned with the service pattern tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.services.common import (
    Actor,
    OperationResult,
    PermissionDeniedError,
    Service,
)

from .events import CalendarEventCreated, CalendarEventDeleted, CalendarEventUpdated
from .types import CalendarEvent, CalendarEventCreate, CalendarEventUpdate
from .validation import CalendarEventValidator

# Import link types for type hints
from modules.calendar_store.link_models import LinkType, SyncBehavior


class CalendarEventService(Service):
    """Service façade that coordinates calendar event operations."""

    def __init__(
        self,
        *,
        repository,
        permission_checker,
        validator: CalendarEventValidator | None = None,
        event_publisher = None,
        default_conflict_resolution: str = "block",
    ) -> None:
        self._repository = repository
        self._permissions = permission_checker
        self._validator = validator or CalendarEventValidator()
        self._publisher = event_publisher
        self._default_conflict_resolution = default_conflict_resolution

    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    async def health_check(self) -> OperationResult[dict[str, Any]]:
        """Check service health."""
        return OperationResult.success({"status": "healthy"})

    async def cleanup(self) -> None:
        """Clean up service resources."""
        pass

    async def create_event(
        self,
        actor: Actor,
        event_data: CalendarEventCreate,
        *,
        conflict_resolution: str | None = None,
    ) -> OperationResult[CalendarEvent]:
        """Create a calendar event after validation and permission checks."""
        strategy = conflict_resolution or self._default_conflict_resolution

        try:
            await self._permissions.require_write_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        validation = await self._validator.validate_create(event_data)
        if validation.is_failure:
            return OperationResult.failure(validation.error or "Validation failed", validation.error_code)

        conflict_result = await self._check_conflicts(
            actor=actor,
            start_time=event_data.start_time,
            end_time=event_data.end_time,
            resolution=strategy,
        )
        if conflict_result.is_failure:
            return OperationResult.failure(conflict_result.error or "Conflict detected", conflict_result.error_code)

        try:
            created_event = await self._repository.create_event(actor, event_data)
        except Exception as exc:  # pragma: no cover - defensive
            return OperationResult.failure(str(exc), "CREATION_FAILED")

        await self._publish_event_created(created_event, actor)
        return OperationResult.success(created_event)

    async def get_event(self, actor: Actor, event_id: str) -> OperationResult[CalendarEvent]:
        """Return the requested event when the actor has access."""
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_read_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        return OperationResult.success(event)

    async def update_event(
        self,
        actor: Actor,
        event_id: str,
        update_data: CalendarEventUpdate,
        *,
        conflict_resolution: str | None = None,
    ) -> OperationResult[CalendarEvent]:
        """Update a calendar event when authorised and valid."""
        strategy = conflict_resolution or self._default_conflict_resolution

        existing_event = await self._repository.get_event_by_id(event_id)
        if not existing_event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_edit_permission(actor, existing_event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        validation = await self._validator.validate_update(update_data, existing_event)
        if validation.is_failure:
            return OperationResult.failure(validation.error or "Validation failed", validation.error_code)

        start_time = update_data.start_time or existing_event.start_time
        end_time = update_data.end_time or existing_event.end_time

        conflict_result = await self._check_conflicts(
            actor=actor,
            start_time=start_time,
            end_time=end_time,
            resolution=strategy,
            exclude_event_id=existing_event.event_id,
        )
        if conflict_result.is_failure:
            return OperationResult.failure(conflict_result.error or "Conflict detected", conflict_result.error_code)

        try:
            updated_event = await self._repository.update_event(event_id, update_data)
        except Exception as exc:  # pragma: no cover - defensive
            return OperationResult.failure(str(exc), "UPDATE_FAILED")

        changed_fields = [
            field
            for field in (
                "title",
                "description",
                "start_time",
                "end_time",
                "timezone_name",
                "location",
                "status",
                "visibility",
                "busy_status",
                "all_day",
                "category_id",
                "is_recurring",
                "recurrence_pattern",
                "metadata",
            )
            if getattr(update_data, field) is not None
        ]

        await self._publish_event_updated(updated_event, actor, changed_fields)
        return OperationResult.success(updated_event)

    async def delete_event(self, actor: Actor, event_id: str) -> OperationResult[bool]:
        """Delete a calendar event when the actor is authorised."""
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_delete_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        deleted = await self._repository.delete_event(event_id)
        if not deleted:
            return OperationResult.failure("Failed to delete event", "DELETE_FAILED")

        await self._publish_event_deleted(event, actor)
        return OperationResult.success(True)

    async def list_events(
        self,
        actor: Actor,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        status: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> OperationResult[List[CalendarEvent]]:
        """List events using the repository and filter by permissions."""
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        events = await self._repository.list_events(
            start_date=start_date,
            end_date=end_date,
            status=status,
            limit=limit,
            offset=offset,
        )

        filtered = await self._permissions.filter_events_by_permissions(actor, events)
        return OperationResult.success(filtered)

    async def bulk_delete_events(
        self,
        actor: Actor,
        event_ids: Iterable[str],
    ) -> OperationResult[int]:
        """Delete multiple events, skipping those the actor cannot manage."""
        event_ids = list(event_ids)
        validation = self._validator.validate_bulk_operation(event_ids, "delete")
        if validation.is_failure:
            return OperationResult.failure(validation.error or "Validation failed", validation.error_code)

        events = await self._repository.get_events_by_ids(event_ids)
        deletable: List[CalendarEvent] = []
        for event in events:
            if await self._permissions.can_delete_event(actor, event):
                deletable.append(event)

        if not deletable:
            return OperationResult.success(0)

        deletable_ids = [event.event_id for event in deletable]
        deleted_count = await self._repository.bulk_delete_events(deletable_ids)

        for event in deletable:
            await self._publish_event_deleted(event, actor)

        return OperationResult.success(deleted_count)

    async def _check_conflicts(
        self,
        *,
        actor: Actor,
        start_time: datetime | None,
        end_time: datetime | None,
        resolution: str,
        exclude_event_id: str | None = None,
    ) -> OperationResult[None]:
        """Validate conflict strategy and enforce blocking rules."""
        if not start_time or not end_time:
            return OperationResult.success(None)

        conflicts = await self._repository.find_conflicting_events(
            actor.user_id,
            start_time=start_time,
            end_time=end_time,
            exclude_event_id=exclude_event_id,
        )

        validation = self._validator.validate_event_conflict_resolution(conflicts, resolution)
        if validation.is_failure:
            return OperationResult.failure(validation.error or "Validation failed", validation.error_code)

        if conflicts and resolution == "block":
            return OperationResult.failure(
                "Event cannot be saved due to scheduling conflicts",
                "SCHEDULING_CONFLICT",
            )

        return OperationResult.success(None)

    async def _publish_event_created(self, event: CalendarEvent, actor: Actor) -> None:
        if not self._publisher:
            return

        domain_event = CalendarEventCreated.create_for_event(
            event_id=event.event_id,
            tenant_id=event.tenant_id or actor.tenant_id,
            actor_type=actor.type,
            event_title=event.title,
            event_start=event.start_time or datetime.utcnow(),
            event_end=event.end_time or event.start_time or datetime.utcnow(),
            category_id=event.category_id,
        )
        await self._publisher.publish(domain_event)

    async def _publish_event_updated(
        self,
        event: CalendarEvent,
        actor: Actor,
        changed_fields: List[str],
    ) -> None:
        if not self._publisher:
            return

        domain_event = CalendarEventUpdated.create_for_event(
            event_id=event.event_id,
            tenant_id=event.tenant_id or actor.tenant_id,
            actor_type=actor.type,
            event_title=event.title,
            changed_fields=changed_fields,
            new_start=event.start_time,
            new_end=event.end_time,
        )
        await self._publisher.publish(domain_event)

    async def _publish_event_deleted(self, event: CalendarEvent, actor: Actor) -> None:
        if not self._publisher:
            return

        domain_event = CalendarEventDeleted.create_for_event(
            event_id=event.event_id,
            tenant_id=event.tenant_id or actor.tenant_id,
            actor_type=actor.type,
            event_title=event.title,
            event_start=event.start_time or datetime.utcnow(),
            event_end=event.end_time or event.start_time or datetime.utcnow(),
        )
        await self._publisher.publish(domain_event)

    # ========================================================================
    # Job/Task linking methods
    # ========================================================================

    async def link_to_job(
        self,
        actor: Actor,
        event_id: str,
        job_id: str,
        *,
        link_type: LinkType = LinkType.MANUAL,
        sync_behavior: SyncBehavior = SyncBehavior.NONE,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[Dict[str, Any]]:
        """Link a calendar event to a job.

        Parameters
        ----------
        actor
            The actor performing the operation
        event_id
            The calendar event to link
        job_id
            The job to link to
        link_type
            Type of link relationship (default: MANUAL)
        sync_behavior
            How changes should propagate (default: NONE)
        notes
            Optional description of why the link was created
        metadata
            Additional link metadata

        Returns
        -------
        OperationResult containing the created link or an error
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_edit_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            link = await self._repository.create_job_link(
                event_id=event_id,
                job_id=job_id,
                link_type=link_type,
                sync_behavior=sync_behavior,
                notes=notes,
                metadata=metadata,
                created_by=actor.user_id,
            )
            return OperationResult.success(link)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "LINK_FAILED")

    async def link_to_task(
        self,
        actor: Actor,
        event_id: str,
        task_id: str,
        *,
        link_type: LinkType = LinkType.MANUAL,
        sync_behavior: SyncBehavior = SyncBehavior.NONE,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[Dict[str, Any]]:
        """Link a calendar event to a task.

        Parameters
        ----------
        actor
            The actor performing the operation
        event_id
            The calendar event to link
        task_id
            The task to link to
        link_type
            Type of link relationship (default: MANUAL)
        sync_behavior
            How changes should propagate (default: NONE)
        notes
            Optional description of why the link was created
        metadata
            Additional link metadata

        Returns
        -------
        OperationResult containing the created link or an error
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_edit_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            link = await self._repository.create_task_link(
                event_id=event_id,
                task_id=task_id,
                link_type=link_type,
                sync_behavior=sync_behavior,
                notes=notes,
                metadata=metadata,
                created_by=actor.user_id,
            )
            return OperationResult.success(link)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "LINK_FAILED")

    async def unlink_from_job(
        self,
        actor: Actor,
        event_id: str,
        job_id: str,
    ) -> OperationResult[bool]:
        """Remove a link between a calendar event and a job.

        Returns
        -------
        OperationResult with True if the link was removed, False if it didn't exist
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_edit_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            deleted = await self._repository.delete_job_link(event_id, job_id)
            return OperationResult.success(deleted)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "UNLINK_FAILED")

    async def unlink_from_task(
        self,
        actor: Actor,
        event_id: str,
        task_id: str,
    ) -> OperationResult[bool]:
        """Remove a link between a calendar event and a task.

        Returns
        -------
        OperationResult with True if the link was removed, False if it didn't exist
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_event_edit_permission(actor, event)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            deleted = await self._repository.delete_task_link(event_id, task_id)
            return OperationResult.success(deleted)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "UNLINK_FAILED")

    async def get_linked_jobs(
        self,
        actor: Actor,
        event_id: str,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Get all jobs linked to a calendar event.

        Returns
        -------
        OperationResult containing list of job link records
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            links = await self._repository.get_job_links_for_event(event_id)
            return OperationResult.success(links)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def get_linked_tasks(
        self,
        actor: Actor,
        event_id: str,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Get all tasks linked to a calendar event.

        Returns
        -------
        OperationResult containing list of task link records
        """
        event = await self._repository.get_event_by_id(event_id)
        if not event:
            return OperationResult.failure("Event not found", "NOT_FOUND")

        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            links = await self._repository.get_task_links_for_event(event_id)
            return OperationResult.success(links)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def get_events_for_job(
        self,
        actor: Actor,
        job_id: str,
    ) -> OperationResult[List[CalendarEvent]]:
        """Get all calendar events linked to a job.

        Returns
        -------
        OperationResult containing list of linked calendar events
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            events = await self._repository.get_events_for_job(job_id)
            # Filter by permissions
            filtered = await self._permissions.filter_events_by_permissions(actor, events)
            return OperationResult.success(filtered)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def get_events_for_task(
        self,
        actor: Actor,
        task_id: str,
    ) -> OperationResult[List[CalendarEvent]]:
        """Get all calendar events linked to a task.

        Returns
        -------
        OperationResult containing list of linked calendar events
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        try:
            events = await self._repository.get_events_for_task(task_id)
            # Filter by permissions
            filtered = await self._permissions.filter_events_by_permissions(actor, events)
            return OperationResult.success(filtered)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    # ========================================================================
    # Event creation from jobs/tasks
    # ========================================================================

    async def create_event_from_job(
        self,
        actor: Actor,
        job: Dict[str, Any],
        *,
        event_overrides: Optional[Dict[str, Any]] = None,
        link_type: LinkType = LinkType.AUTO_CREATED,
        sync_behavior: SyncBehavior = SyncBehavior.FROM_SOURCE,
    ) -> OperationResult[CalendarEvent]:
        """Create a calendar event from a job and automatically link them.

        Parameters
        ----------
        actor
            The actor performing the operation
        job
            Job data dict with at minimum 'id' and 'name' keys.
            Optional: 'description', 'scheduled_at', 'estimated_duration_minutes'
        event_overrides
            Optional overrides for event fields (title, start_time, end_time, etc.)
        link_type
            Type of link relationship (default: AUTO_CREATED)
        sync_behavior
            How changes should propagate (default: FROM_SOURCE)

        Returns
        -------
        OperationResult containing the created calendar event
        """
        overrides = event_overrides or {}
        
        # Build event data from job
        job_id = str(job.get("id", ""))
        if not job_id:
            return OperationResult.failure("Job ID is required", "INVALID_INPUT")
        
        title = overrides.get("title") or f"Job: {job.get('name', 'Unnamed Job')}"
        description = overrides.get("description") or job.get("description")
        
        # Determine timing
        start_time = overrides.get("start_time")
        end_time = overrides.get("end_time")
        
        if not start_time:
            # Try to get scheduled time from job
            scheduled_at = job.get("scheduled_at") or job.get("next_run_at")
            if scheduled_at:
                if isinstance(scheduled_at, str):
                    start_time = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
                else:
                    start_time = scheduled_at
            else:
                start_time = datetime.now(timezone.utc)
        
        if not end_time:
            # Estimate duration or default to 1 hour
            duration_minutes = job.get("estimated_duration_minutes", 60)
            end_time = start_time + timedelta(minutes=duration_minutes)
        
        event_data = CalendarEventCreate(
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            timezone_name=overrides.get("timezone_name", "UTC"),
            location=overrides.get("location"),
            status=overrides.get("status", "confirmed"),
            visibility=overrides.get("visibility", "private"),
            busy_status=overrides.get("busy_status", "busy"),
            all_day=overrides.get("all_day", False),
            category_id=overrides.get("category_id"),
            metadata={
                "source": "job",
                "job_id": job_id,
                "job_name": job.get("name"),
                "job_status": str(job.get("status", "")),
                **(overrides.get("metadata") or {}),
            },
        )
        
        # Create the event
        result = await self.create_event(actor, event_data)
        if result.is_failure:
            return result
        
        created_event = result.value
        
        # Link the event to the job
        link_result = await self.link_to_job(
            actor,
            created_event.event_id,
            job_id,
            link_type=link_type,
            sync_behavior=sync_behavior,
            notes=f"Auto-created from job '{job.get('name')}'",
        )
        
        if link_result.is_failure:
            # Event was created but link failed - log warning but return success
            # The event is still valid, just not linked
            pass
        
        return OperationResult.success(created_event)

    async def create_event_from_task(
        self,
        actor: Actor,
        task: Dict[str, Any],
        *,
        event_overrides: Optional[Dict[str, Any]] = None,
        link_type: LinkType = LinkType.AUTO_CREATED,
        sync_behavior: SyncBehavior = SyncBehavior.FROM_SOURCE,
    ) -> OperationResult[CalendarEvent]:
        """Create a calendar event from a task and automatically link them.

        Parameters
        ----------
        actor
            The actor performing the operation
        task
            Task data dict with at minimum 'id' and 'title' keys.
            Optional: 'description', 'due_at', 'estimated_minutes'
        event_overrides
            Optional overrides for event fields (title, start_time, end_time, etc.)
        link_type
            Type of link relationship (default: AUTO_CREATED)
        sync_behavior
            How changes should propagate (default: FROM_SOURCE)

        Returns
        -------
        OperationResult containing the created calendar event
        """
        overrides = event_overrides or {}
        
        # Build event data from task
        task_id = str(task.get("id", ""))
        if not task_id:
            return OperationResult.failure("Task ID is required", "INVALID_INPUT")
        
        title = overrides.get("title") or f"Task: {task.get('title', 'Unnamed Task')}"
        description = overrides.get("description") or task.get("description")
        
        # Determine timing
        start_time = overrides.get("start_time")
        end_time = overrides.get("end_time")
        
        if not start_time and not end_time:
            # If task has a due date, create event ending at due date
            due_at = task.get("due_at")
            if due_at:
                if isinstance(due_at, str):
                    end_time = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
                else:
                    end_time = due_at
                # Start 1 hour before due (or estimated time)
                duration_minutes = task.get("estimated_minutes", 60)
                start_time = end_time - timedelta(minutes=duration_minutes)
            else:
                # No due date - schedule for now
                start_time = datetime.now(timezone.utc)
                duration_minutes = task.get("estimated_minutes", 60)
                end_time = start_time + timedelta(minutes=duration_minutes)
        elif start_time and not end_time:
            duration_minutes = task.get("estimated_minutes", 60)
            end_time = start_time + timedelta(minutes=duration_minutes)
        elif end_time and not start_time:
            duration_minutes = task.get("estimated_minutes", 60)
            start_time = end_time - timedelta(minutes=duration_minutes)
        
        # Determine link type based on task context
        actual_link_type = link_type
        if task.get("due_at") and link_type == LinkType.AUTO_CREATED:
            actual_link_type = LinkType.DEADLINE
        
        event_data = CalendarEventCreate(
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            timezone_name=overrides.get("timezone_name", "UTC"),
            location=overrides.get("location"),
            status=overrides.get("status", "confirmed"),
            visibility=overrides.get("visibility", "private"),
            busy_status=overrides.get("busy_status", "busy"),
            all_day=overrides.get("all_day", False),
            category_id=overrides.get("category_id"),
            metadata={
                "source": "task",
                "task_id": task_id,
                "task_title": task.get("title"),
                "task_status": str(task.get("status", "")),
                "task_priority": task.get("priority"),
                **(overrides.get("metadata") or {}),
            },
        )
        
        # Create the event
        result = await self.create_event(actor, event_data)
        if result.is_failure:
            return result
        
        created_event = result.value
        
        # Link the event to the task
        link_result = await self.link_to_task(
            actor,
            created_event.event_id,
            task_id,
            link_type=actual_link_type,
            sync_behavior=sync_behavior,
            notes=f"Auto-created from task '{task.get('title')}'",
        )
        
        if link_result.is_failure:
            # Event was created but link failed - log warning but return success
            pass
        
        return OperationResult.success(created_event)

    # ========================================================================
    # Agent-focused schedule methods
    # ========================================================================

    async def search_events(
        self,
        actor: Actor,
        query: str,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        category_id: str | None = None,
        limit: int = 50,
    ) -> OperationResult[List[CalendarEvent]]:
        """Full-text search for calendar events.

        Uses PostgreSQL full-text search when available, falls back to
        LIKE queries for other databases. Results are sorted by relevance.

        Parameters
        ----------
        actor
            The actor performing the search
        query
            Search query string
        start_date
            Filter events after this time
        end_date
            Filter events before this time
        category_id
            Filter by category
        limit
            Maximum results to return

        Returns
        -------
        OperationResult containing list of matching events
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        if not query or not query.strip():
            return OperationResult.success([])

        try:
            events = await self._repository.search_events(
                query=query,
                start=start_date,
                end=end_date,
                category_id=category_id,
                limit=limit,
            )
            # Filter by permissions
            filtered = await self._permissions.filter_events_by_permissions(actor, events)
            return OperationResult.success(filtered)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "SEARCH_FAILED")

    async def get_upcoming_events(
        self,
        actor: Actor,
        *,
        hours_ahead: int = 24,
        limit: int = 10,
    ) -> OperationResult[List[CalendarEvent]]:
        """Get upcoming events within the specified time window.

        Useful for agents to understand what's happening soon.

        Parameters
        ----------
        actor
            The actor requesting upcoming events
        hours_ahead
            Number of hours to look ahead (default: 24)
        limit
            Maximum number of events to return (default: 10)

        Returns
        -------
        OperationResult containing list of upcoming events sorted by start time
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        now = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=hours_ahead)

        try:
            events = await self._repository.list_events(
                start_date=now,
                end_date=end_time,
                limit=limit,
            )
            # Filter by permissions
            filtered = await self._permissions.filter_events_by_permissions(actor, events)
            return OperationResult.success(filtered)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def check_availability(
        self,
        actor: Actor,
        start_time: datetime,
        end_time: datetime,
    ) -> OperationResult[Dict[str, Any]]:
        """Check if a time range is available (no conflicting events).

        Parameters
        ----------
        actor
            The actor to check availability for
        start_time
            Start of the time range to check
        end_time
            End of the time range to check

        Returns
        -------
        OperationResult containing availability info:
            - available: bool indicating if the time is free
            - conflicts: list of conflicting events if any
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        if end_time <= start_time:
            return OperationResult.failure(
                "End time must be after start time",
                "INVALID_TIME_RANGE",
            )

        try:
            conflicts = await self._repository.find_conflicting_events(
                actor.user_id,
                start_time=start_time,
                end_time=end_time,
            )

            return OperationResult.success({
                "available": len(conflicts) == 0,
                "conflicts": conflicts,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            })
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def find_free_time(
        self,
        actor: Actor,
        *,
        start_range: datetime,
        end_range: datetime,
        duration_minutes: int,
        working_hours: tuple[int, int] | None = None,
        exclude_weekends: bool = False,
        max_slots: int = 5,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Find available time slots within a date range.

        Searches for gaps in the calendar where a meeting of the
        specified duration could be scheduled.

        Parameters
        ----------
        actor
            The actor to find free time for
        start_range
            Start of the search range
        end_range
            End of the search range
        duration_minutes
            Required duration in minutes
        working_hours
            Optional tuple of (start_hour, end_hour) in 24h format
            e.g., (9, 17) for 9 AM to 5 PM
        exclude_weekends
            If True, skip Saturday and Sunday
        max_slots
            Maximum number of slots to return (default: 5)

        Returns
        -------
        OperationResult containing list of free time slots:
            - start: datetime
            - end: datetime
            - duration_minutes: int
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        if end_range <= start_range:
            return OperationResult.failure(
                "End range must be after start range",
                "INVALID_TIME_RANGE",
            )

        if duration_minutes <= 0:
            return OperationResult.failure(
                "Duration must be positive",
                "INVALID_DURATION",
            )

        duration = timedelta(minutes=duration_minutes)

        try:
            # Get all events in the range
            events = await self._repository.list_events(
                start_date=start_range,
                end_date=end_range,
            )
            # Filter by permissions
            events = await self._permissions.filter_events_by_permissions(actor, events)

            # Sort events by start time
            events = sorted(events, key=lambda e: e.start_time or start_range)

            free_slots: List[Dict[str, Any]] = []
            current_time = start_range

            # Iterate through time and find gaps
            for event in events:
                event_start = event.start_time or start_range
                event_end = event.end_time or event_start

                # Check gap before this event
                gap_start = current_time
                gap_end = event_start

                # Find slots in this gap
                slots_in_gap = self._find_slots_in_gap(
                    gap_start=gap_start,
                    gap_end=gap_end,
                    duration=duration,
                    working_hours=working_hours,
                    exclude_weekends=exclude_weekends,
                )
                free_slots.extend(slots_in_gap)

                if len(free_slots) >= max_slots:
                    break

                # Move current time past this event
                if event_end > current_time:
                    current_time = event_end

            # Check gap after last event
            if len(free_slots) < max_slots and current_time < end_range:
                slots_in_gap = self._find_slots_in_gap(
                    gap_start=current_time,
                    gap_end=end_range,
                    duration=duration,
                    working_hours=working_hours,
                    exclude_weekends=exclude_weekends,
                )
                free_slots.extend(slots_in_gap)

            # Limit results
            free_slots = free_slots[:max_slots]

            return OperationResult.success(free_slots)
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    def _find_slots_in_gap(
        self,
        *,
        gap_start: datetime,
        gap_end: datetime,
        duration: timedelta,
        working_hours: tuple[int, int] | None,
        exclude_weekends: bool,
    ) -> List[Dict[str, Any]]:
        """Find valid time slots within a gap between events."""
        slots: List[Dict[str, Any]] = []

        current = gap_start

        while current + duration <= gap_end:
            # Check weekend exclusion
            if exclude_weekends and current.weekday() >= 5:  # Saturday=5, Sunday=6
                # Skip to Monday
                days_until_monday = 7 - current.weekday()
                current = current.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_until_monday)
                continue

            # Check working hours
            if working_hours:
                start_hour, end_hour = working_hours
                current_hour = current.hour

                if current_hour < start_hour:
                    # Move to start of working hours
                    current = current.replace(
                        hour=start_hour, minute=0, second=0, microsecond=0
                    )
                    continue
                elif current_hour >= end_hour:
                    # Move to next day's working hours
                    current = (current + timedelta(days=1)).replace(
                        hour=start_hour, minute=0, second=0, microsecond=0
                    )
                    continue

                # Check if slot fits within working hours
                slot_end = current + duration
                if slot_end.hour > end_hour or (
                    slot_end.hour == end_hour and slot_end.minute > 0
                ):
                    # Slot would extend past working hours, move to next day
                    current = (current + timedelta(days=1)).replace(
                        hour=start_hour, minute=0, second=0, microsecond=0
                    )
                    continue

            # Valid slot found
            slot_end = current + duration
            slots.append({
                "start": current,
                "end": slot_end,
                "duration_minutes": int(duration.total_seconds() / 60),
            })

            # Move to end of this slot for next potential slot
            current = slot_end

        return slots

    async def get_calendar_summary(
        self,
        actor: Actor,
        *,
        period: str = "today",
    ) -> OperationResult[Dict[str, Any]]:
        """Get a summary of calendar events for agent context.

        Provides a quick overview of the calendar state useful for
        agent decision-making.

        Parameters
        ----------
        actor
            The actor requesting the summary
        period
            Time period: "today", "tomorrow", "week", "month"

        Returns
        -------
        OperationResult containing calendar summary:
            - period: str
            - start: datetime
            - end: datetime
            - total_events: int
            - events: list of event summaries
            - busy_hours: float (estimated hours of commitments)
            - next_event: dict or None
        """
        try:
            await self._permissions.require_read_permission(actor)
        except PermissionDeniedError as exc:
            return self._permission_failure(str(exc))

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Determine date range based on period
        if period == "today":
            start_date = today_start
            end_date = today_start + timedelta(days=1)
        elif period == "tomorrow":
            start_date = today_start + timedelta(days=1)
            end_date = today_start + timedelta(days=2)
        elif period == "week":
            start_date = today_start
            end_date = today_start + timedelta(days=7)
        elif period == "month":
            start_date = today_start
            end_date = today_start + timedelta(days=30)
        else:
            return OperationResult.failure(
                f"Invalid period: {period}. Use 'today', 'tomorrow', 'week', or 'month'",
                "INVALID_PERIOD",
            )

        try:
            events = await self._repository.list_events(
                start_date=start_date,
                end_date=end_date,
            )
            # Filter by permissions
            events = await self._permissions.filter_events_by_permissions(actor, events)

            # Sort by start time
            events = sorted(events, key=lambda e: e.start_time or start_date)

            # Calculate busy hours
            busy_minutes = 0
            for event in events:
                if event.start_time and event.end_time:
                    duration = (event.end_time - event.start_time).total_seconds() / 60
                    busy_minutes += duration

            # Find next upcoming event
            next_event = None
            for event in events:
                if event.start_time and event.start_time > now:
                    next_event = {
                        "event_id": event.event_id,
                        "title": event.title,
                        "start_time": event.start_time.isoformat(),
                        "end_time": event.end_time.isoformat() if event.end_time else None,
                        "location": event.location,
                    }
                    break

            # Build event summaries
            event_summaries = [
                {
                    "event_id": e.event_id,
                    "title": e.title,
                    "start_time": e.start_time.isoformat() if e.start_time else None,
                    "end_time": e.end_time.isoformat() if e.end_time else None,
                    "all_day": e.all_day,
                    "location": e.location,
                    "status": e.status,
                }
                for e in events
            ]

            return OperationResult.success({
                "period": period,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "total_events": len(events),
                "events": event_summaries,
                "busy_hours": round(busy_minutes / 60, 2),
                "next_event": next_event,
            })
        except Exception as exc:  # pragma: no cover
            return OperationResult.failure(str(exc), "QUERY_FAILED")

    async def suggest_meeting_times(
        self,
        actor: Actor,
        *,
        duration_minutes: int,
        preferred_start: datetime | None = None,
        preferred_end: datetime | None = None,
        working_hours: tuple[int, int] | None = (9, 17),
        exclude_weekends: bool = True,
        max_suggestions: int = 3,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """Suggest optimal meeting times based on calendar availability.

        This is a convenience wrapper around find_free_time that provides
        sensible defaults for meeting scheduling.

        Parameters
        ----------
        actor
            The actor to find meeting times for
        duration_minutes
            Required meeting duration in minutes
        preferred_start
            Earliest time to consider (default: now)
        preferred_end
            Latest time to consider (default: 7 days from now)
        working_hours
            Tuple of (start_hour, end_hour), default: (9, 17)
        exclude_weekends
            Skip weekends, default: True
        max_suggestions
            Maximum number of suggestions (default: 3)

        Returns
        -------
        OperationResult containing list of suggested meeting times
        """
        now = datetime.now(timezone.utc)

        start_range = preferred_start or now
        end_range = preferred_end or (now + timedelta(days=7))

        return await self.find_free_time(
            actor,
            start_range=start_range,
            end_range=end_range,
            duration_minutes=duration_minutes,
            working_hours=working_hours,
            exclude_weekends=exclude_weekends,
            max_slots=max_suggestions,
        )

    @staticmethod
    def _permission_failure(message: str) -> OperationResult:
        return OperationResult.failure(message, "PERMISSION_DENIED")
