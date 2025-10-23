"""Repository helpers for working with task persistence."""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy import select
except Exception:  # pragma: no cover - fallback when SQLAlchemy is absent
    def select(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy select is unavailable in this environment")


try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.orm import Session, joinedload, sessionmaker
except Exception:  # pragma: no cover - fallback when SQLAlchemy is absent
    class _Session:  # pragma: no cover - lightweight placeholder type
        pass

    class _Sessionmaker:  # pragma: no cover - lightweight placeholder type
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

    def joinedload(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy joinedload is unavailable in this environment")

    Session = _Session  # type: ignore[assignment]
    sessionmaker = _Sessionmaker  # type: ignore[assignment]
else:  # pragma: no cover - sanitize stubbed implementations
    if not isinstance(sessionmaker, type):  # pragma: no cover - test stub compatibility
        class _Sessionmaker:  # lightweight placeholder mirroring SQLAlchemy API
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

        sessionmaker = _Sessionmaker  # type: ignore[assignment]

from modules.conversation_store.models import Conversation
from modules.job_store import ensure_job_schema

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


def _coerce_uuid(value: Any | None) -> Optional[uuid.UUID]:
    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.UUID(hex=text.replace("-", ""))


def _normalize_tenant_id(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("Tenant identifier must be provided")
    return text


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        candidate = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Datetime value cannot be empty")
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            candidate = datetime.fromisoformat(normalized)
        except ValueError:
            candidate = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _dt_to_iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    normalized = moment.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _normalize_status(value: Any | None) -> TaskStatus:
    if value is None:
        return TaskStatus.DRAFT
    if isinstance(value, TaskStatus):
        return value
    text = str(value).strip().lower()
    if not text:
        return TaskStatus.DRAFT
    return TaskStatus(text)


def _normalize_meta(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("Task metadata must be a mapping")
    return dict(metadata)


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


class TaskStoreRepository:
    """Persistence helper around :mod:`modules.task_store` models."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    @contextlib.contextmanager
    def _session_scope(self) -> Iterator[Session]:
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- schema helpers -------------------------------------------------

    def create_schema(self) -> None:
        engine = getattr(self._session_factory, "bind", None)
        if engine is None:
            with contextlib.ExitStack() as stack:
                try:
                    session = stack.enter_context(self._session_factory())
                except Exception as exc:  # pragma: no cover - defensive fallback
                    raise RuntimeError("Session factory must be bound to an engine") from exc
                engine = session.get_bind()
        if engine is None:
            raise RuntimeError("Session factory must be bound to an engine")
        Base.metadata.create_all(engine)
        ensure_task_schema(engine)
        ensure_job_schema(engine)

    # -- query helpers --------------------------------------------------

    def list_tasks(
        self,
        *,
        tenant_id: Any,
        status: Any | Sequence[Any] | None = None,
        owner_id: Any | None = None,
        conversation_id: Any | None = None,
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

        with self._session_scope() as session:
            stmt = (
                select(Task)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .options(joinedload(Task.conversation))
                .where(Conversation.tenant_id == tenant_key)
                .order_by(Task.created_at.desc())
            )

            if status_filters:
                stmt = stmt.where(Task.status.in_([value for value in status_filters]))
            if owner_uuid is not None:
                stmt = stmt.where(Task.owner_id == owner_uuid)
            if conversation_uuid is not None:
                stmt = stmt.where(Task.conversation_id == conversation_uuid)

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
        if conversation_uuid is None:
            raise ValueError("Conversation identifier is required")
        due_at_value = _normalize_due_at(due_at)
        metadata_dict = _normalize_meta(metadata)

        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                raise ValueError("Conversation does not exist")
            if conversation.tenant_id != tenant_key:
                raise ValueError("Conversation belongs to a different tenant")

            record = Task(
                title=title_text,
                description=str(description).strip() if description is not None else None,
                status=task_status,
                priority=priority_value,
                owner_id=owner_uuid,
                session_id=session_uuid,
                conversation_id=conversation_uuid,
                meta=metadata_dict,
                due_at=due_at_value,
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            payload = _serialize_task(record)

            event = TaskEvent(
                task_id=record.id,
                event_type=TaskEventType.CREATED,
                payload={"status": record.status.value},
            )
            session.add(event)
            session.flush()
            payload["events"] = [_serialize_event(event)]
            return payload

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

        with self._session_scope() as session:
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

            if expected_updated_at is not None:
                expected = _coerce_dt(expected_updated_at)
                current = record.updated_at.astimezone(timezone.utc)
                if current != expected:
                    raise TaskConcurrencyError("Task was modified by another transaction")

            events: list[TaskEvent] = []
            status_changed = False
            status_before = record.status

            for field, value in changes.items():
                if field == "title":
                    record.title = _normalize_title(value)
                elif field == "description":
                    record.description = str(value).strip() if value is not None else None
                elif field == "status":
                    new_status = _normalize_status(value)
                    if new_status != record.status:
                        record.status = new_status
                        status_changed = True
                elif field == "priority":
                    record.priority = _normalize_priority(value)
                elif field == "owner_id":
                    record.owner_id = _coerce_uuid(value)
                elif field == "session_id":
                    record.session_id = _coerce_uuid(value)
                elif field == "metadata":
                    record.meta = _normalize_meta(value)
                elif field == "due_at":
                    record.due_at = _normalize_due_at(value)
                else:
                    raise ValueError(f"Unsupported task attribute: {field}")

            session.flush()
            payload = _serialize_task(record)

            event_payload = {"changes": dict(changes)}
            events.append(
                TaskEvent(
                    task_id=record.id,
                    event_type=TaskEventType.UPDATED,
                    payload=event_payload,
                )
            )
            if status_changed:
                events.append(
                    TaskEvent(
                        task_id=record.id,
                        event_type=TaskEventType.STATUS_CHANGED,
                        payload={
                            "from": status_before.value,
                            "to": record.status.value,
                        },
                    )
                )

            for event in events:
                session.add(event)
            session.flush()
            payload["events"] = [_serialize_event(event) for event in events]
            return payload

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

        with self._session_scope() as session:
            stmt = (
                select(Task)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(Task.id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )
            task = session.execute(stmt).scalar_one_or_none()
            if task is None:
                raise TaskNotFoundError("Task not found for tenant")

            event = TaskEvent(
                task_id=task.id,
                event_type=event_type,
                triggered_by_id=_coerce_uuid(triggered_by_id),
                session_id=_coerce_uuid(session_id),
                payload=dict(payload or {}),
            )
            session.add(event)
            session.flush()
            return _serialize_event(event)
