"""Domain service encapsulating task lifecycle orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from modules.analytics.persona_metrics import record_task_lifecycle_event
from modules.store_common.service_utils import (
    LifecycleServiceBase,
    coerce_enum_value,
    parse_timestamp as _parse_timestamp,
)

from .models import TaskStatus
from .repository import (
    TaskConcurrencyError,
    TaskNotFoundError,
    TaskStoreRepository,
)


class TaskServiceError(RuntimeError):
    """Base class for task service errors."""


class TaskTransitionError(TaskServiceError):
    """Raised when an invalid lifecycle transition is requested."""


class TaskDependencyError(TaskServiceError):
    """Raised when dependencies prevent a transition."""


def _coerce_status(value: Any) -> TaskStatus:
    return coerce_enum_value(value, TaskStatus)


def _extract_persona(payload: Mapping[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    persona = metadata.get("persona") or metadata.get("persona_name")
    if persona is None:
        return None
    text = str(persona).strip()
    return text or None


_ALLOWED_TRANSITIONS: Dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.CANCELLED},
    TaskStatus.REVIEW: {TaskStatus.DONE, TaskStatus.CANCELLED},
    TaskStatus.DONE: set(),
    TaskStatus.CANCELLED: set(),
}


_COMPLETE_STATUSES = {TaskStatus.DONE, TaskStatus.CANCELLED}


class TaskService(LifecycleServiceBase):
    """Facade that orchestrates repository operations and lifecycle rules."""

    def __init__(
        self,
        repository: TaskStoreRepository,
        *,
        event_emitter: Optional[Callable[[str, Mapping[str, Any]], Any]] = None,
    ) -> None:
        super().__init__(event_emitter=event_emitter)
        self._repository = repository

    # -- Repository proxies -------------------------------------------------

    def list_tasks(
        self,
        *,
        tenant_id: Any,
        status: Any | Sequence[Any] | None = None,
        owner_id: Any | None = None,
        conversation_id: Any | None = None,
        limit: int | None = None,
        cursor: tuple[datetime, uuid.UUID] | None = None,
    ) -> list[Dict[str, Any]]:
        return self._repository.list_tasks(
            tenant_id=tenant_id,
            status=status,
            owner_id=owner_id,
            conversation_id=conversation_id,
            limit=limit,
            cursor=cursor,
        )

    def get_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        with_events: bool = False,
    ) -> Dict[str, Any]:
        return self._repository.get_task(task_id, tenant_id=tenant_id, with_events=with_events)

    # -- Lifecycle orchestration --------------------------------------------

    def create_task(
        self,
        title: Any,
        *,
        tenant_id: Any,
        description: Any | None = None,
        status: Any | None = None,
        priority: Any | None = None,
        owner_id: Any | None = None,
        session_id: Any | None = None,
        conversation_id: Any,
        due_at: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        record = self._repository.create_task(
            title,
            tenant_id=tenant_id,
            description=description,
            status=status,
            priority=priority,
            owner_id=owner_id,
            session_id=session_id,
            conversation_id=conversation_id,
            due_at=due_at,
            metadata=metadata,
        )
        self._emit(
            "task.created",
            {"task_id": record["id"], "tenant_id": record.get("tenant_id"), "status": record["status"]},
        )
        created_at = _parse_timestamp(record.get("created_at")) or datetime.now(timezone.utc)
        record_task_lifecycle_event(
            task_id=record.get("id"),
            event="created",
            persona=_extract_persona(record),
            tenant_id=record.get("tenant_id"),
            from_status=None,
            to_status=record.get("status"),
            success=None,
            latency_ms=None,
            reassignments=0,
            timestamp=created_at,
            metadata={
                "priority": record.get("priority"),
                "conversation_id": record.get("conversation_id"),
            },
        )
        return record

    def update_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        change_payload = dict(changes)
        if not change_payload:
            raise ValueError("At least one field must be provided for update")

        snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)

        change_payload, owner_changed = self.apply_owner_change(
            snapshot=snapshot,
            changes=change_payload,
        )

        if not change_payload:
            return self.prepare_noop_update(
                snapshot=snapshot,
                expected_updated_at=expected_updated_at,
                concurrency_error=TaskConcurrencyError,
                error_message="Task was modified by another transaction",
                events_field="events",
            )

        applied_changes = dict(change_payload)
        record = self._repository.update_task(
            task_id,
            tenant_id=tenant_id,
            changes=change_payload,
            expected_updated_at=expected_updated_at,
        )
        self._emit(
            "task.updated",
            {
                "task_id": record["id"],
                "tenant_id": record.get("tenant_id"),
                "changes": applied_changes,
            },
        )
        if owner_changed:
            timestamp = _parse_timestamp(record.get("updated_at")) or datetime.now(timezone.utc)
            owner_serialized = record.get("owner_id") or None
            record_task_lifecycle_event(
                task_id=record.get("id"),
                event="reassigned",
                persona=_extract_persona(record),
                tenant_id=record.get("tenant_id"),
                from_status=record.get("status"),
                to_status=record.get("status"),
                success=None,
                latency_ms=None,
                reassignments=1,
                timestamp=timestamp,
                metadata={
                    "owner_id": owner_serialized,
                },
            )
        return record

    def transition_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        target_status: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        snapshot = self._repository.get_task(task_id, tenant_id=tenant_id)
        current_status = _coerce_status(snapshot["status"])
        desired_status = _coerce_status(target_status)

        if desired_status == current_status:
            return snapshot

        allowed = _ALLOWED_TRANSITIONS.get(current_status, set())
        if desired_status not in allowed:
            raise TaskTransitionError(
                f"Cannot transition task {task_id} from {current_status.value} to {desired_status.value}"
            )

        if desired_status not in {TaskStatus.CANCELLED, TaskStatus.DRAFT}:
            dependency_statuses = self._repository.dependency_statuses(task_id, tenant_id=tenant_id)
            incomplete = [status for status in dependency_statuses if _coerce_status(status) not in _COMPLETE_STATUSES]
            if incomplete:
                raise TaskDependencyError(
                    "Cannot advance task because dependencies are incomplete",
                )

        reference_timestamp = expected_updated_at or snapshot.get("updated_at")

        updated = self._repository.update_task(
            task_id,
            tenant_id=tenant_id,
            changes={"status": desired_status},
            expected_updated_at=reference_timestamp,
        )

        self._emit(
            "task.status_changed",
            {
                "task_id": updated["id"],
                "tenant_id": updated.get("tenant_id"),
                "from": current_status.value,
                "to": desired_status.value,
            },
        )
        previous_timestamp = _parse_timestamp(snapshot.get("updated_at"))
        current_timestamp = _parse_timestamp(updated.get("updated_at")) or datetime.now(timezone.utc)
        latency_ms: Optional[float] = None
        if previous_timestamp is not None and current_timestamp is not None:
            latency_ms = (current_timestamp - previous_timestamp).total_seconds() * 1000.0
        if desired_status == TaskStatus.DONE:
            lifecycle_event = "completed"
            success_flag: Optional[bool] = True
        elif desired_status == TaskStatus.CANCELLED:
            lifecycle_event = "cancelled"
            success_flag = False
        else:
            lifecycle_event = "status_changed"
            success_flag = None
        record_task_lifecycle_event(
            task_id=updated.get("id"),
            event=lifecycle_event,
            persona=_extract_persona(updated),
            tenant_id=updated.get("tenant_id"),
            from_status=current_status.value,
            to_status=desired_status.value,
            success=success_flag,
            latency_ms=latency_ms,
            reassignments=0,
            timestamp=current_timestamp,
            metadata={"requested_status": desired_status.value},
        )
        return updated

    # -- Dependency helpers -------------------------------------------------

    def dependencies_complete(self, task_id: Any, *, tenant_id: Any) -> bool:
        statuses = self._repository.dependency_statuses(task_id, tenant_id=tenant_id)
        return all(_coerce_status(status) in _COMPLETE_STATUSES for status in statuses)


__all__ = [
    "TaskService",
    "TaskServiceError",
    "TaskTransitionError",
    "TaskDependencyError",
    "TaskConcurrencyError",
    "TaskNotFoundError",
]
