"""Repository helpers for working with task persistence."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from modules.conversation_store.models import Conversation
from modules.store_common.repository_utils import (
    BaseStoreRepository,
    Session,
    _coerce_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_enum_value,
    _normalize_meta,
    _normalize_tenant_id,
    and_,
    joinedload,
    or_,
    select,
    sessionmaker,
)
from .models import (
    Base,
    Task,
    TaskDependency,
    TaskEvent,
    TaskEventType,
    TaskStatus,
    ensure_task_schema,
)


class TaskStoreError(RuntimeError):
    """Base class for repository level errors."""


class TaskNotFoundError(TaskStoreError):
    """Raised when a task cannot be located for the active tenant."""


class TaskConcurrencyError(TaskStoreError):
    """Raised when optimistic concurrency checks fail."""


def _normalize_status(value: Any | None) -> TaskStatus:
    return _normalize_enum_value(value, TaskStatus, TaskStatus.DRAFT)


def _normalize_title(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Task title must not be empty")
    return text


def _normalize_priority(value: Any | None) -> int:
    if value is None:
        return 0
    return int(value)


def _normalize_due_at(value: Any | None) -> Optional[datetime]:
    if value is None or value == "":
        return None
    return _coerce_dt(value)


def _serialize_task(task: Task) -> Dict[str, Any]:
    return {
        "id": str(task.id),
        "title": task.title,
        "description": task.description,
        "status": task.status.value,
        "priority": int(task.priority or 0),
        "owner_id": str(task.owner_id) if task.owner_id else None,
        "session_id": str(task.session_id) if task.session_id else None,
        "conversation_id": str(task.conversation_id) if task.conversation_id else None,
        "tenant_id": task.conversation.tenant_id if task.conversation else None,
        "metadata": dict(task.meta or {}),
        "due_at": _dt_to_iso(task.due_at),
        "created_at": _dt_to_iso(task.created_at),
        "updated_at": _dt_to_iso(task.updated_at),
    }


def _serialize_event(event: TaskEvent) -> Dict[str, Any]:
    return {
        "id": str(event.id),
        "task_id": str(event.task_id),
        "event_type": event.event_type.value,
        "triggered_by_id": str(event.triggered_by_id) if event.triggered_by_id else None,
        "session_id": str(event.session_id) if event.session_id else None,
        "payload": dict(event.payload or {}),
        "created_at": _dt_to_iso(event.created_at),
    }


class TaskStoreRepository(BaseStoreRepository):
    """Persistence helper around :mod:`modules.task_store` models."""

    def __init__(self, session_factory: sessionmaker) -> None:
        super().__init__(session_factory)

    # -- schema helpers -------------------------------------------------

    def _get_schema_metadata(self) -> Any:
        return Base.metadata

    def _ensure_additional_schema(self, engine: Any) -> None:
        ensure_task_schema(engine)
        from modules.job_store import ensure_job_schema  # local import to avoid circular dependency

        ensure_job_schema(engine)

    # -- query helpers --------------------------------------------------

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
        tenant_key = _normalize_tenant_id(tenant_id)
        status_filters: Iterable[TaskStatus]
        if status is None:
            status_filters = ()
        elif isinstance(status, Sequence) and not isinstance(status, (str, bytes)):
            status_filters = tuple(_normalize_status(value) for value in status)
        else:
            status_filters = (_normalize_status(status),)

        owner_uuid = _coerce_uuid(owner_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        cursor_moment: Optional[datetime]
        cursor_task_id: Optional[uuid.UUID]
        if cursor is None:
            cursor_moment = None
            cursor_task_id = None
        else:
            cutoff_time, cutoff_id = cursor
            cursor_moment = _coerce_dt(cutoff_time)
            cursor_task_id = _coerce_uuid(cutoff_id)

        with self._session_scope() as session:
            stmt = (
                select(Task)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .options(joinedload(Task.conversation))
                .where(Conversation.tenant_id == tenant_key)
                .order_by(Task.created_at.desc(), Task.id.desc())
            )

            if status_filters:
                stmt = stmt.where(Task.status.in_([value for value in status_filters]))
            if owner_uuid is not None:
                stmt = stmt.where(Task.owner_id == owner_uuid)
            if conversation_uuid is not None:
                stmt = stmt.where(Task.conversation_id == conversation_uuid)
            if cursor_moment is not None and cursor_task_id is not None:
                stmt = stmt.where(
                    or_(
                        Task.created_at < cursor_moment,
                        and_(Task.created_at == cursor_moment, Task.id < cursor_task_id),
                    )
                )
            if limit is not None:
                stmt = stmt.limit(int(limit))

            tasks = session.execute(stmt).scalars().all()
            return [_serialize_task(task) for task in tasks]

    def get_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        with_events: bool = False,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        task_uuid = _coerce_uuid(task_id)
        if task_uuid is None:
            raise TaskNotFoundError("Task identifier is required")

        load_options = [joinedload(Task.conversation)]
        if with_events:
            load_options.append(joinedload(Task.events))

        with self._session_scope() as session:
            stmt = (
                select(Task)
                .options(*load_options)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(Task.id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                raise TaskNotFoundError("Task not found for tenant")
            payload = _serialize_task(record)
            if with_events:
                payload["events"] = [_serialize_event(event) for event in sorted(record.events, key=lambda item: item.created_at)]
            return payload

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
        tenant_key = _normalize_tenant_id(tenant_id)
        title_text = _normalize_title(title)
        task_status = _normalize_status(status)
        priority_value = _normalize_priority(priority)
        owner_uuid = _coerce_uuid(owner_id)
        session_uuid = _coerce_uuid(session_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        due_at_value = _normalize_due_at(due_at)
        metadata_dict = _normalize_meta(metadata, error_message="Task metadata must be a mapping")
        return self._create_with_created_event(
            model_cls=Task,
            model_kwargs={
                "title": title_text,
                "description": str(description).strip() if description is not None else None,
                "status": task_status,
                "priority": priority_value,
                "owner_id": owner_uuid,
                "session_id": session_uuid,
                "conversation_id": conversation_uuid,
                "meta": metadata_dict,
                "due_at": due_at_value,
            },
            serializer=_serialize_task,
            event_model_cls=TaskEvent,
            event_enum=TaskEventType,
            event_serializer=_serialize_event,
            event_foreign_key="task_id",
            tenant_id=tenant_key,
            conversation_id=conversation_uuid,
            require_conversation=True,
            event_payload_factory=lambda record: {
                "status": record.status.value
                if isinstance(record.status, TaskStatus)
                else str(record.status)
            },
        )

    def update_task(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("At least one field must be provided for update")

        tenant_key = _normalize_tenant_id(tenant_id)
        task_uuid = _coerce_uuid(task_id)
        if task_uuid is None:
            raise TaskNotFoundError("Task identifier is required")

        def load_record(session: Session) -> Task:
            stmt = (
                select(Task)
                .options(joinedload(Task.conversation))
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(Task.id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
                .with_for_update()
            )
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                raise TaskNotFoundError("Task not found for tenant")
            return record

        def context_factory(record: Task) -> Dict[str, Any]:
            return {"status_before": record.status, "status_changed": False}

        def mutate_title(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.title = _normalize_title(value)

        def mutate_description(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.description = str(value).strip() if value is not None else None

        def mutate_status(
            record: Task, value: Any, _session: Session, context: Dict[str, Any]
        ) -> None:
            new_status = _normalize_status(value)
            if new_status != record.status:
                record.status = new_status
                context["status_changed"] = True

        def mutate_priority(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.priority = _normalize_priority(value)

        def mutate_owner(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.owner_id = _coerce_uuid(value)

        def mutate_session(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.session_id = _coerce_uuid(value)

        def mutate_metadata(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.meta = _normalize_meta(
                value, error_message="Task metadata must be a mapping"
            )

        def mutate_due_at(
            record: Task, value: Any, _session: Session, _context: Dict[str, Any]
        ) -> None:
            record.due_at = _normalize_due_at(value)

        def build_change_event(
            record: Task, update_changes: Mapping[str, Any], _context: Dict[str, Any]
        ) -> Sequence[TaskEvent]:
            return [
                TaskEvent(
                    task_id=record.id,
                    event_type=TaskEventType.UPDATED,
                    payload={"changes": dict(update_changes)},
                )
            ]

        def build_status_event(
            record: Task, _update_changes: Mapping[str, Any], context: Dict[str, Any]
        ) -> Sequence[TaskEvent]:
            if not context.get("status_changed"):
                return []
            status_before = context.get("status_before")
            if status_before is None:
                return []
            return [
                TaskEvent(
                    task_id=record.id,
                    event_type=TaskEventType.STATUS_CHANGED,
                    payload={
                        "from": status_before.value,
                        "to": record.status.value,
                    },
                )
            ]

        field_mutators = {
            "title": mutate_title,
            "description": mutate_description,
            "status": mutate_status,
            "priority": mutate_priority,
            "owner_id": mutate_owner,
            "session_id": mutate_session,
            "metadata": mutate_metadata,
            "due_at": mutate_due_at,
        }

        return self._update_with_optimistic_lock(
            load_record=load_record,
            expected_updated_at=expected_updated_at,
            concurrency_error_factory=lambda: TaskConcurrencyError(
                "Task was modified by another transaction"
            ),
            changes=changes,
            field_mutators=field_mutators,
            serializer=_serialize_task,
            event_factories=[build_change_event, build_status_event],
            event_serializer=_serialize_event,
            unknown_field_error_factory=lambda field: ValueError(
                f"Unsupported task attribute: {field}"
            ),
            context_factory=context_factory,
        )

    def dependency_statuses(self, task_id: Any, *, tenant_id: Any) -> list[str]:
        tenant_key = _normalize_tenant_id(tenant_id)
        task_uuid = _coerce_uuid(task_id)
        if task_uuid is None:
            return []

        with self._session_scope() as session:
            stmt = (
                select(Task.status)
                .join(TaskDependency, Task.id == TaskDependency.depends_on_id)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(TaskDependency.task_id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )
            rows = session.execute(stmt).scalars().all()
            return [status.value if isinstance(status, TaskStatus) else str(status) for status in rows]

    def record_event(
        self,
        task_id: Any,
        *,
        tenant_id: Any,
        event_type: TaskEventType,
        payload: Mapping[str, Any],
        triggered_by_id: Any | None = None,
        session_id: Any | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        task_uuid = _coerce_uuid(task_id)
        if task_uuid is None:
            raise TaskNotFoundError("Task identifier is required")

        triggered_uuid = _coerce_uuid(triggered_by_id)
        session_uuid = _coerce_uuid(session_id)

        return self._record_event(
            aggregate_loader=lambda session: session.execute(
                select(Task)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(Task.id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
            ).scalar_one_or_none(),
            not_found_error_factory=lambda: TaskNotFoundError("Task not found for tenant"),
            event_model_cls=TaskEvent,
            event_serializer=_serialize_event,
            foreign_key_field="task_id",
            event_type=event_type,
            payload=payload,
            triggered_by_id=triggered_uuid,
            session_id=session_uuid,
        )
