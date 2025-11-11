"""Repository helpers for working with job persistence."""

from __future__ import annotations

import contextlib
import dataclasses
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from modules.conversation_store.models import Conversation
from modules.store_common.repository_utils import (
    IntegrityError,
    Session,
    _coerce_dt,
    _coerce_optional_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_meta,
    _normalize_tenant_id,
    _session_scope,
    and_,
    joinedload,
    or_,
    select,
    sessionmaker,
)
from modules.task_store.models import Task, TaskStatus

from .models import (
    Base,
    Job,
    JobEvent,
    JobEventType,
    JobRun,
    JobRunStatus,
    JobSchedule,
    JobStatus,
    JobTaskLink,
    ensure_job_schema,
)


class JobStoreError(RuntimeError):
    """Base class for repository level errors."""


class JobNotFoundError(JobStoreError):
    """Raised when a job cannot be located for the active tenant."""


class JobConcurrencyError(JobStoreError):
    """Raised when optimistic concurrency checks fail for job entities."""


class JobTransitionError(JobStoreError):
    """Raised when an invalid state transition is attempted."""


@dataclasses.dataclass(slots=True, frozen=True)
class _Cursor:
    created_at: datetime
    job_id: uuid.UUID


def _normalize_status(value: Any | None) -> JobStatus:
    if value is None:
        return JobStatus.DRAFT
    if isinstance(value, JobStatus):
        return value
    text = str(value).strip().lower()
    if not text:
        return JobStatus.DRAFT
    return JobStatus(text)


def _normalize_run_status(value: Any | None) -> JobRunStatus:
    if value is None:
        return JobRunStatus.SCHEDULED
    if isinstance(value, JobRunStatus):
        return value
    text = str(value).strip().lower()
    if not text:
        return JobRunStatus.SCHEDULED
    return JobRunStatus(text)


def _normalize_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Job name must not be empty")
    return text


def _normalize_relationship(value: Any | None) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "relates_to"


def _encode_cursor(job: Job) -> str:
    created_iso = _dt_to_iso(job.created_at) or ""
    return f"{created_iso}|{job.id}"


def _decode_cursor(cursor: str | None) -> Optional[_Cursor]:
    if not cursor:
        return None
    if "|" not in cursor:
        raise ValueError("Cursor is malformed")
    created_at_text, job_id_text = cursor.split("|", 1)
    return _Cursor(created_at=_coerce_dt(created_at_text), job_id=_coerce_uuid(job_id_text) or uuid.uuid4())


def _serialize_job(job: Job) -> Dict[str, Any]:
    return {
        "id": str(job.id),
        "name": job.name,
        "description": job.description,
        "status": job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        "owner_id": str(job.owner_id) if job.owner_id else None,
        "conversation_id": str(job.conversation_id) if job.conversation_id else None,
        "tenant_id": job.tenant_id,
        "metadata": dict(job.meta or {}),
        "created_at": _dt_to_iso(job.created_at),
        "updated_at": _dt_to_iso(job.updated_at),
    }


def _serialize_schedule(schedule: JobSchedule) -> Dict[str, Any]:
    return {
        "id": str(schedule.id),
        "job_id": str(schedule.job_id),
        "schedule_type": schedule.schedule_type,
        "expression": schedule.expression,
        "timezone": schedule.timezone,
        "next_run_at": _dt_to_iso(schedule.next_run_at),
        "metadata": dict(schedule.meta or {}),
        "created_at": _dt_to_iso(schedule.created_at),
        "updated_at": _dt_to_iso(schedule.updated_at),
    }


def _serialize_run(run: JobRun) -> Dict[str, Any]:
    return {
        "id": str(run.id),
        "job_id": str(run.job_id),
        "run_number": int(run.run_number or 0),
        "status": run.status.value if isinstance(run.status, JobRunStatus) else str(run.status),
        "started_at": _dt_to_iso(run.started_at),
        "finished_at": _dt_to_iso(run.finished_at),
        "metadata": dict(run.meta or {}),
        "created_at": _dt_to_iso(run.created_at),
        "updated_at": _dt_to_iso(run.updated_at),
    }


def _serialize_event(event: JobEvent) -> Dict[str, Any]:
    return {
        "id": str(event.id),
        "job_id": str(event.job_id),
        "event_type": event.event_type.value if isinstance(event.event_type, JobEventType) else str(event.event_type),
        "triggered_by_id": str(event.triggered_by_id) if event.triggered_by_id else None,
        "session_id": str(event.session_id) if event.session_id else None,
        "payload": dict(event.payload or {}),
        "created_at": _dt_to_iso(event.created_at),
    }


def _serialize_task_link(link: JobTaskLink, task: Task | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(link.id),
        "job_id": str(link.job_id),
        "task_id": str(link.task_id),
        "relationship_type": link.relationship_type,
        "metadata": dict(link.meta or {}),
        "created_at": _dt_to_iso(link.created_at),
    }
    if task is not None:
        payload["task"] = {
            "id": str(task.id),
            "title": task.title,
            "status": task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
            "conversation_id": str(task.conversation_id) if task.conversation_id else None,
        }
    return payload


def _valid_job_transition(current: JobStatus, new: JobStatus) -> bool:
    if current == new:
        return True
    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
    if current in terminal:
        return False
    transitions = {
        JobStatus.DRAFT: {JobStatus.SCHEDULED, JobStatus.RUNNING, JobStatus.CANCELLED},
        JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED},
        JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED},
    }
    allowed = transitions.get(current, set())
    return new in allowed


def _valid_run_transition(current: JobRunStatus, new: JobRunStatus) -> bool:
    if current == new:
        return True
    terminal = {JobRunStatus.SUCCEEDED, JobRunStatus.FAILED, JobRunStatus.CANCELLED}
    if current in terminal:
        return False
    transitions = {
        JobRunStatus.SCHEDULED: {JobRunStatus.RUNNING, JobRunStatus.CANCELLED},
        JobRunStatus.RUNNING: {JobRunStatus.SUCCEEDED, JobRunStatus.FAILED, JobRunStatus.CANCELLED},
    }
    allowed = transitions.get(current, set())
    return new in allowed


class JobStoreRepository:
    """Persistence helper around :mod:`modules.job_store` models."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    @contextlib.contextmanager
    def _session_scope(self) -> Iterator[Session]:
        with _session_scope(self._session_factory) as session:
            yield session

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
        ensure_job_schema(engine)

    # -- job helpers ----------------------------------------------------

    def list_jobs(
        self,
        *,
        tenant_id: Any,
        status: Any | Sequence[Any] | None = None,
        owner_id: Any | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        status_filters: Iterable[JobStatus]
        if status is None:
            status_filters = ()
        elif isinstance(status, Sequence) and not isinstance(status, (str, bytes)):
            status_filters = tuple(_normalize_status(item) for item in status)
        else:
            status_filters = (_normalize_status(status),)

        owner_uuid = _coerce_uuid(owner_id)
        limit = max(1, min(int(limit or 50), 100))
        cursor_state = _decode_cursor(cursor)

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.tenant_id == tenant_key)
                .order_by(Job.created_at.desc(), Job.id.desc())
            )

            if status_filters:
                stmt = stmt.where(Job.status.in_([value for value in status_filters]))
            if owner_uuid is not None:
                stmt = stmt.where(Job.owner_id == owner_uuid)
            if cursor_state is not None:
                stmt = stmt.where(
                    or_(
                        Job.created_at < cursor_state.created_at,
                        and_(
                            Job.created_at == cursor_state.created_at,
                            Job.id < cursor_state.job_id,
                        ),
                    )
                )

            stmt = stmt.limit(limit + 1)
            jobs = session.execute(stmt).scalars().all()
            records = jobs[:limit]
            next_cursor = None
            if len(jobs) > limit and records:
                next_cursor = _encode_cursor(records[-1])

            return {
                "items": [_serialize_job(job) for job in records],
                "next_cursor": next_cursor,
            }

    def get_job_by_name(
        self,
        name: Any,
        *,
        tenant_id: Any,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        name_text = _normalize_name(name)

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.tenant_id == tenant_key)
                .where(Job.name == name_text)
            )
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                raise JobNotFoundError("Job not found for tenant")
            return _serialize_job(record)

    def get_job(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        with_schedule: bool = False,
        with_runs: bool = False,
        with_events: bool = False,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        load_options = []
        if with_schedule:
            load_options.append(joinedload(Job.schedule))
        if with_runs:
            load_options.append(joinedload(Job.runs))
        if with_events:
            load_options.append(joinedload(Job.events))

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .options(*load_options)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
            )
            result = session.execute(stmt)
            if with_runs or with_events:
                result = result.unique()
            record = result.scalar_one_or_none()
            if record is None:
                raise JobNotFoundError("Job not found for tenant")
            payload = _serialize_job(record)
            if with_schedule and record.schedule:
                payload["schedule"] = _serialize_schedule(record.schedule)
            if with_runs:
                runs = sorted(record.runs, key=lambda run: (run.run_number or 0), reverse=True)
                payload["runs"] = [_serialize_run(run) for run in runs]
            if with_events:
                events = sorted(record.events, key=lambda event: event.created_at)
                payload["events"] = [_serialize_event(event) for event in events]
            return payload

    def create_job(
        self,
        name: Any,
        *,
        tenant_id: Any,
        description: Any | None = None,
        status: Any | None = None,
        owner_id: Any | None = None,
        conversation_id: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        name_text = _normalize_name(name)
        status_value = _normalize_status(status)
        owner_uuid = _coerce_uuid(owner_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        metadata_dict = _normalize_meta(metadata)

        with self._session_scope() as session:
            if conversation_uuid is not None:
                conversation = session.get(Conversation, conversation_uuid)
                if conversation is None:
                    raise ValueError("Conversation does not exist")
                if conversation.tenant_id != tenant_key:
                    raise ValueError("Conversation belongs to a different tenant")

            record = Job(
                name=name_text,
                description=str(description).strip() if description is not None else None,
                status=status_value,
                owner_id=owner_uuid,
                conversation_id=conversation_uuid,
                tenant_id=tenant_key,
                meta=metadata_dict,
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            payload = _serialize_job(record)

            event = JobEvent(
                job_id=record.id,
                event_type=JobEventType.CREATED,
                payload={"status": record.status.value},
            )
            session.add(event)
            session.flush()
            payload["events"] = [_serialize_event(event)]
            return payload

    def update_job(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("At least one field must be provided for update")

        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
                .with_for_update()
            )
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                raise JobNotFoundError("Job not found for tenant")

            if expected_updated_at is not None:
                expected = _coerce_dt(expected_updated_at)
                current = record.updated_at.astimezone(timezone.utc)
                if current != expected:
                    raise JobConcurrencyError("Job was modified by another transaction")

            events: list[JobEvent] = []
            status_before = record.status if isinstance(record.status, JobStatus) else JobStatus(str(record.status))
            status_changed = False

            for field, value in changes.items():
                if field == "name":
                    record.name = _normalize_name(value)
                elif field == "description":
                    record.description = str(value).strip() if value is not None else None
                elif field == "status":
                    new_status = _normalize_status(value)
                    current_status = record.status if isinstance(record.status, JobStatus) else JobStatus(str(record.status))
                    if new_status != current_status:
                        if not _valid_job_transition(current_status, new_status):
                            raise JobTransitionError("Unsupported job status transition")
                        record.status = new_status
                        status_changed = True
                elif field == "owner_id":
                    record.owner_id = _coerce_uuid(value)
                elif field == "conversation_id":
                    conversation_uuid = _coerce_uuid(value)
                    if conversation_uuid is not None:
                        conversation = session.get(Conversation, conversation_uuid)
                        if conversation is None:
                            raise ValueError("Conversation does not exist")
                        if conversation.tenant_id != tenant_key:
                            raise ValueError("Conversation belongs to a different tenant")
                    record.conversation_id = conversation_uuid
                elif field == "metadata":
                    record.meta = _normalize_meta(value)
                else:
                    raise ValueError(f"Unsupported job attribute: {field}")

            session.flush()
            payload = _serialize_job(record)

            change_event = JobEvent(
                job_id=record.id,
                event_type=JobEventType.UPDATED,
                payload={"changes": dict(changes)},
            )
            session.add(change_event)
            events.append(change_event)

            if status_changed:
                status_event = JobEvent(
                    job_id=record.id,
                    event_type=JobEventType.STATUS_CHANGED,
                    payload={
                        "from": status_before.value if isinstance(status_before, JobStatus) else str(status_before),
                        "to": record.status.value if isinstance(record.status, JobStatus) else str(record.status),
                    },
                )
                session.add(status_event)
                events.append(status_event)

            session.flush()
            payload["events"] = [_serialize_event(event) for event in events]
            return payload

    # -- schedule helpers -----------------------------------------------

    def upsert_schedule(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        schedule_type: Any,
        expression: Any,
        timezone_name: Any | None = None,
        next_run_at: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        schedule_type_text = str(schedule_type).strip() or "cron"
        expression_text = str(expression).strip()
        if not expression_text:
            raise ValueError("Schedule expression must not be empty")
        timezone_text = str(timezone_name).strip() if timezone_name else "UTC"
        metadata_dict = _normalize_meta(metadata)
        next_run_dt = _coerce_optional_dt(next_run_at)

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .options(joinedload(Job.schedule))
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
                .with_for_update()
            )
            job = session.execute(stmt).unique().scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            if job.schedule is None:
                schedule = JobSchedule(
                    job_id=job.id,
                    schedule_type=schedule_type_text,
                    expression=expression_text,
                    timezone=timezone_text,
                    next_run_at=next_run_dt,
                    meta=metadata_dict,
                )
                session.add(schedule)
                session.flush()
                session.refresh(schedule)
            else:
                schedule = job.schedule
                schedule.schedule_type = schedule_type_text
                schedule.expression = expression_text
                schedule.timezone = timezone_text
                schedule.next_run_at = next_run_dt
                schedule.meta = metadata_dict
                session.flush()
                session.refresh(schedule)

            return _serialize_schedule(schedule)

    # -- run helpers ----------------------------------------------------

    def create_run(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        status: Any | None = None,
        started_at: Any | None = None,
        finished_at: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        status_value = _normalize_run_status(status)
        started_dt = _coerce_optional_dt(started_at)
        finished_dt = _coerce_optional_dt(finished_at)
        metadata_dict = _normalize_meta(metadata)

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .options(joinedload(Job.runs))
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
                .with_for_update()
            )
            job = session.execute(stmt).unique().scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            next_number = 1
            if job.runs:
                next_number = max(int(run.run_number or 0) for run in job.runs) + 1

            record = JobRun(
                job_id=job.id,
                run_number=next_number,
                status=status_value,
                started_at=started_dt,
                finished_at=finished_dt,
                meta=metadata_dict,
            )
            session.add(record)
            session.flush()
            session.refresh(record)

            event = JobEvent(
                job_id=job.id,
                event_type=JobEventType.RUN,
                payload={
                    "run_number": record.run_number,
                    "status": record.status.value,
                },
            )
            session.add(event)
            session.flush()

            return _serialize_run(record)

    def update_run(
        self,
        run_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("At least one field must be provided for update")

        tenant_key = _normalize_tenant_id(tenant_id)
        run_uuid = _coerce_uuid(run_id)
        if run_uuid is None:
            raise JobNotFoundError("Run identifier is required")

        with self._session_scope() as session:
            stmt = (
                select(JobRun)
                .join(Job, JobRun.job_id == Job.id)
                .where(JobRun.id == run_uuid)
                .where(Job.tenant_id == tenant_key)
                .with_for_update()
            )
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                raise JobNotFoundError("Job run not found for tenant")

            if expected_updated_at is not None:
                expected = _coerce_dt(expected_updated_at)
                current = record.updated_at.astimezone(timezone.utc)
                if current != expected:
                    raise JobConcurrencyError("Run was modified by another transaction")

            for field, value in changes.items():
                if field == "status":
                    new_status = _normalize_run_status(value)
                    current_status = record.status if isinstance(record.status, JobRunStatus) else JobRunStatus(str(record.status))
                    if new_status != current_status:
                        if not _valid_run_transition(current_status, new_status):
                            raise JobTransitionError("Unsupported run status transition")
                        record.status = new_status
                elif field == "started_at":
                    record.started_at = _coerce_optional_dt(value)
                elif field == "finished_at":
                    record.finished_at = _coerce_optional_dt(value)
                elif field == "metadata":
                    record.meta = _normalize_meta(value)
                else:
                    raise ValueError(f"Unsupported run attribute: {field}")

            session.flush()
            session.refresh(record)
            return _serialize_run(record)

    def list_runs(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        cursor: str | None = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")
        limit = max(1, min(int(limit or 50), 100))

        run_cursor: Optional[tuple[int, uuid.UUID]] = None
        if cursor:
            if "|" not in cursor:
                raise ValueError("Cursor is malformed")
            number_text, run_id_text = cursor.split("|", 1)
            run_cursor = (int(number_text), _coerce_uuid(run_id_text) or uuid.uuid4())

        with self._session_scope() as session:
            stmt = (
                select(JobRun)
                .join(Job, JobRun.job_id == Job.id)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
                .order_by(JobRun.run_number.desc(), JobRun.id.desc())
                .limit(limit + 1)
            )

            if run_cursor is not None:
                stmt = stmt.where(
                    or_(
                        JobRun.run_number < run_cursor[0],
                        and_(
                            JobRun.run_number == run_cursor[0],
                            JobRun.id < run_cursor[1],
                        ),
                    )
                )

            runs = session.execute(stmt).scalars().all()
            records = runs[:limit]
            next_cursor = None
            if len(runs) > limit and records:
                last = records[-1]
                next_cursor = f"{int(last.run_number or 0)}|{last.id}"

            return {
                "items": [_serialize_run(run) for run in records],
                "next_cursor": next_cursor,
            }

    # -- event helpers --------------------------------------------------

    def record_event(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        event_type: JobEventType,
        payload: Mapping[str, Any],
        triggered_by_id: Any | None = None,
        session_id: Any | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        with self._session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
            )
            job = session.execute(stmt).scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            event = JobEvent(
                job_id=job.id,
                event_type=event_type,
                triggered_by_id=_coerce_uuid(triggered_by_id),
                session_id=_coerce_uuid(session_id),
                payload=dict(payload or {}),
            )
            session.add(event)
            session.flush()
            return _serialize_event(event)

    # -- task linking helpers ------------------------------------------

    def attach_task(
        self,
        job_id: Any,
        task_id: Any,
        *,
        tenant_id: Any,
        relationship_type: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        task_uuid = _coerce_uuid(task_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")
        if task_uuid is None:
            raise ValueError("Task identifier is required")

        relationship = _normalize_relationship(relationship_type)
        metadata_dict = _normalize_meta(metadata)

        with self._session_scope() as session:
            job_stmt = (
                select(Job)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
            )
            job = session.execute(job_stmt).scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            task_stmt = (
                select(Task)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(Task.id == task_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )
            task = session.execute(task_stmt).scalar_one_or_none()
            if task is None:
                raise ValueError("Task does not exist for tenant")

            link_stmt = (
                select(JobTaskLink)
                .where(JobTaskLink.job_id == job.id)
                .where(JobTaskLink.task_id == task.id)
            )
            existing = session.execute(link_stmt).scalar_one_or_none()
            if existing is not None:
                return _serialize_task_link(existing, task)

            link = JobTaskLink(
                job_id=job.id,
                task_id=task.id,
                relationship_type=relationship,
                meta=metadata_dict,
            )
            session.add(link)
            try:
                session.flush()
            except IntegrityError as exc:  # pragma: no cover - defensive unique constraint handling
                raise ValueError("Task is already linked to this job") from exc
            session.refresh(link)
            return _serialize_task_link(link, task)

    def list_linked_tasks(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
    ) -> list[Dict[str, Any]]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        with self._session_scope() as session:
            job_stmt = (
                select(Job)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
            )
            job = session.execute(job_stmt).scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            stmt = (
                select(JobTaskLink, Task)
                .join(Task, JobTaskLink.task_id == Task.id)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(JobTaskLink.job_id == job.id)
                .where(Conversation.tenant_id == tenant_key)
                .order_by(JobTaskLink.created_at.asc())
            )
            rows = session.execute(stmt).all()
            return [_serialize_task_link(link, task) for link, task in rows]

    def detach_task(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        link_id: Any | None = None,
        task_id: Any | None = None,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            raise JobNotFoundError("Job identifier is required")

        link_uuid = _coerce_uuid(link_id) if link_id is not None else None
        task_uuid = _coerce_uuid(task_id) if task_id is not None else None
        if link_uuid is None and task_uuid is None:
            raise ValueError("A link_id or task_id must be supplied")

        with self._session_scope() as session:
            job_stmt = (
                select(Job)
                .where(Job.id == job_uuid)
                .where(Job.tenant_id == tenant_key)
            )
            job = session.execute(job_stmt).scalar_one_or_none()
            if job is None:
                raise JobNotFoundError("Job not found for tenant")

            stmt = (
                select(JobTaskLink, Task)
                .join(Task, JobTaskLink.task_id == Task.id)
                .join(Conversation, Task.conversation_id == Conversation.id)
                .where(JobTaskLink.job_id == job.id)
                .where(Conversation.tenant_id == tenant_key)
            )
            if link_uuid is not None:
                stmt = stmt.where(JobTaskLink.id == link_uuid)
            if task_uuid is not None:
                stmt = stmt.where(JobTaskLink.task_id == task_uuid)

            row = session.execute(stmt).first()
            if row is None:
                raise ValueError("Linked task was not found for this job")

            link, task = row
            payload = _serialize_task_link(link, task)
            session.delete(link)
            session.flush()
            return payload


__all__ = [
    "JobStoreRepository",
    "JobStoreError",
    "JobNotFoundError",
    "JobConcurrencyError",
    "JobTransitionError",
]
