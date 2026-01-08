"""Calendar event service implementation aligned with the service pattern tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, List, Sequence

from core.services.common import (
    Actor,
    OperationResult,
    PermissionDeniedError,
    Service,
)

from .events import CalendarEventCreated, CalendarEventDeleted, CalendarEventUpdated
from .types import CalendarEvent, CalendarEventCreate, CalendarEventUpdate
from .validation import CalendarEventValidator


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

    @staticmethod
    def _permission_failure(message: str) -> OperationResult:
        return OperationResult.failure(message, "PERMISSION_DENIED")
