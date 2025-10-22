"""Domain service encapsulating task lifecycle orchestration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from modules.Tools.tool_event_system import publish_bus_event

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
    if isinstance(value, TaskStatus):
        return value
    text = str(value).strip().lower()
    return TaskStatus(text)


_ALLOWED_TRANSITIONS: Dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.CANCELLED},
    TaskStatus.REVIEW: {TaskStatus.DONE, TaskStatus.CANCELLED},
    TaskStatus.DONE: set(),
    TaskStatus.CANCELLED: set(),
}


_COMPLETE_STATUSES = {TaskStatus.DONE, TaskStatus.CANCELLED}


class TaskService:
    """Facade that orchestrates repository operations and lifecycle rules."""

    def __init__(
        self,
        repository: TaskStoreRepository,
        *,
        event_emitter: Optional[Callable[[str, Mapping[str, Any]], Any]] = None,
    ) -> None:
        self._repository = repository
        if event_emitter is None:
            self._emit: Callable[[str, Mapping[str, Any]], Any] = self._default_emit
        else:
            self._emit = event_emitter

    @staticmethod
    def _default_emit(event_name: str, payload: Mapping[str, Any]) -> None:
        publish_bus_event(event_name, dict(payload))

    # -- Repository proxies -------------------------------------------------

    def list_tasks(
        self,
        *,
        tenant_id: Any,
        status: Any | Sequence[Any] | None = None,
        owner_id: Any | None = None,
        conversation_id: Any | None = None,
    ) -> list[Dict[str, Any]]:
        return self._repository.list_tasks(
            tenant_id=tenant_id,
            status=status,
            owner_id=owner_id,
            conversation_id=conversation_id,
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
        return record

    def update_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        record = self._repository.update_task(
            task_id,
            tenant_id=tenant_id,
            changes=changes,
            expected_updated_at=expected_updated_at,
        )
        self._emit(
            "task.updated",
            {
                "task_id": record["id"],
                "tenant_id": record.get("tenant_id"),
                "changes": dict(changes),
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
