"""Domain service orchestrating job lifecycle operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from modules.Tools.tool_event_system import publish_bus_event
from modules.analytics.persona_metrics import record_job_lifecycle_event
from modules.task_store import TaskStatus

from .models import JobStatus
from .repository import (
    JobConcurrencyError,
    JobNotFoundError,
    JobStoreRepository,
)


class JobServiceError(RuntimeError):
    """Base class for job service level errors."""


class JobTransitionError(JobServiceError):
    """Raised when an invalid lifecycle transition is requested."""


class JobDependencyError(JobServiceError):
    """Raised when linked tasks prevent the requested transition."""


_ALLOWED_TRANSITIONS: Dict[JobStatus, set[JobStatus]] = {
    JobStatus.DRAFT: {JobStatus.SCHEDULED, JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}

_TASK_COMPLETE_STATUSES = {TaskStatus.DONE, TaskStatus.CANCELLED}
_ROSTER_REQUIRED_STATUSES = {JobStatus.SCHEDULED, JobStatus.RUNNING}
_SCHEDULE_REQUIRED_STATUSES = {JobStatus.SCHEDULED, JobStatus.RUNNING}
_DEPENDENCY_REQUIRED_STATUSES = {JobStatus.RUNNING, JobStatus.SUCCEEDED}


def _coerce_status(value: Any) -> JobStatus:
    if isinstance(value, JobStatus):
        return value
    text = str(value).strip().lower()
    return JobStatus(text)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        timestamp = value
    elif value is None:
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError:
            return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def _normalize_owner_identifier(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return str(uuid.UUID(bytes=value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.UUID(hex=text.replace("-", "")))


def _extract_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if isinstance(metadata, Mapping):
        return metadata
    meta_alt = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else None
    return meta_alt or {}


def _extract_persona_roster(payload: Mapping[str, Any]) -> list[str]:
    metadata = _extract_metadata(payload)
    roster_candidates: Iterable[Any] = ()
    if "personas" in metadata:
        roster_candidates = metadata.get("personas", [])
    elif "roster" in metadata:
        roster_candidates = metadata.get("roster", [])
    elif "persona_roster" in metadata:
        roster_candidates = metadata.get("persona_roster", [])
    else:
        single = metadata.get("persona") or metadata.get("persona_name")
        if single:
            roster_candidates = [single]

    roster: list[str] = []
    for entry in roster_candidates or []:
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("persona") or entry.get("id")
            if name:
                text = str(name).strip()
                if text:
                    roster.append(text)
        elif entry is not None:
            text = str(entry).strip()
            if text:
                roster.append(text)
    return roster


def _primary_persona(payload: Mapping[str, Any]) -> Optional[str]:
    roster = _extract_persona_roster(payload)
    return roster[0] if roster else None


def _normalize_task_status(value: Any) -> Optional[TaskStatus]:
    if value is None:
        return None
    if isinstance(value, TaskStatus):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return TaskStatus(text)
    except ValueError:
        return None


class JobService:
    """Application service that coordinates job lifecycle rules."""

    def __init__(
        self,
        repository: JobStoreRepository,
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

    def list_jobs(
        self,
        *,
        tenant_id: Any,
        status: Any | Sequence[Any] | None = None,
        owner_id: Any | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        return self._repository.list_jobs(
            tenant_id=tenant_id,
            status=status,
            owner_id=owner_id,
            cursor=cursor,
            limit=limit,
        )

    def get_job(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        with_schedule: bool = False,
        with_runs: bool = False,
        with_events: bool = False,
    ) -> Dict[str, Any]:
        return self._repository.get_job(
            job_id,
            tenant_id=tenant_id,
            with_schedule=with_schedule,
            with_runs=with_runs,
            with_events=with_events,
        )

    def list_linked_tasks(self, job_id: Any, *, tenant_id: Any) -> list[Dict[str, Any]]:
        return self._repository.list_linked_tasks(job_id, tenant_id=tenant_id)

    # -- Lifecycle orchestration --------------------------------------------

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
        record = self._repository.create_job(
            name,
            tenant_id=tenant_id,
            description=description,
            status=status,
            owner_id=owner_id,
            conversation_id=conversation_id,
            metadata=metadata,
        )
        payload = {
            "job_id": record["id"],
            "tenant_id": record.get("tenant_id"),
            "status": record.get("status"),
        }
        self._emit("job.created", payload)
        created_at = _parse_timestamp(record.get("created_at")) or datetime.now(timezone.utc)
        record_job_lifecycle_event(
            job_id=record.get("id"),
            event="created",
            persona=_primary_persona(record),
            tenant_id=record.get("tenant_id"),
            from_status=None,
            to_status=record.get("status"),
            success=None,
            timestamp=created_at,
            metadata={"owner_id": record.get("owner_id")},
        )
        return record

    def update_job(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        changes: Mapping[str, Any],
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        change_payload = dict(changes)
        if "status" in change_payload:
            raise ValueError("Status updates must be performed via transition_job")

        snapshot = self._repository.get_job(job_id, tenant_id=tenant_id)

        owner_changed = False
        if "owner_id" in change_payload:
            existing_owner = _normalize_owner_identifier(snapshot.get("owner_id"))
            requested_owner = _normalize_owner_identifier(change_payload["owner_id"])
            if existing_owner == requested_owner:
                change_payload.pop("owner_id")
            else:
                owner_changed = True

        if not change_payload:
            if expected_updated_at is not None:
                expected_timestamp = _parse_timestamp(expected_updated_at)
                current_timestamp = _parse_timestamp(snapshot.get("updated_at"))
                if (
                    expected_timestamp is None
                    or current_timestamp is None
                    or current_timestamp != expected_timestamp
                ):
                    raise JobConcurrencyError("Job was modified by another transaction")
            snapshot_payload = dict(snapshot)
            snapshot_payload.setdefault("events", [])
            return snapshot_payload

        updated = self._repository.update_job(
            job_id,
            tenant_id=tenant_id,
            changes=change_payload,
            expected_updated_at=expected_updated_at,
        )
        emit_payload = {
            "job_id": updated["id"],
            "tenant_id": updated.get("tenant_id"),
            "changes": dict(change_payload),
        }
        self._emit("job.updated", emit_payload)

        if owner_changed or change_payload:
            timestamp = _parse_timestamp(updated.get("updated_at")) or datetime.now(timezone.utc)
            record_job_lifecycle_event(
                job_id=updated.get("id"),
                event="updated",
                persona=_primary_persona(updated),
                tenant_id=updated.get("tenant_id"),
                from_status=updated.get("status"),
                to_status=updated.get("status"),
                success=None,
                timestamp=timestamp,
                metadata={"changes": dict(change_payload)},
            )
        return updated

    def transition_job(
        self,
        job_id: Any,
        *,
        tenant_id: Any,
        target_status: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        snapshot = self._repository.get_job(
            job_id,
            tenant_id=tenant_id,
            with_schedule=True,
        )
        current_status = _coerce_status(snapshot["status"])
        desired_status = _coerce_status(target_status)

        if desired_status == current_status:
            return snapshot

        allowed = _ALLOWED_TRANSITIONS.get(current_status, set())
        if desired_status not in allowed:
            raise JobTransitionError(
                f"Cannot transition job {job_id} from {current_status.value} to {desired_status.value}"
            )

        roster = _extract_persona_roster(snapshot)
        if desired_status in _ROSTER_REQUIRED_STATUSES and not roster:
            raise JobTransitionError("Job must define a persona roster before activation")

        schedule = snapshot.get("schedule") if isinstance(snapshot, Mapping) else None
        if desired_status in _SCHEDULE_REQUIRED_STATUSES:
            if not isinstance(schedule, Mapping) or not schedule.get("next_run_at"):
                raise JobTransitionError("Job schedule must be configured before activation")

        if desired_status in _DEPENDENCY_REQUIRED_STATUSES:
            linked_tasks = self._repository.list_linked_tasks(job_id, tenant_id=tenant_id)
            incomplete = []
            for link in linked_tasks:
                task = link.get("task") if isinstance(link, Mapping) else None
                status_value = None
                if isinstance(task, Mapping):
                    status_value = task.get("status")
                normalized = _normalize_task_status(status_value)
                if normalized is not None and normalized not in _TASK_COMPLETE_STATUSES:
                    incomplete.append(link)
            if incomplete:
                raise JobDependencyError(
                    "Cannot advance job because linked tasks are incomplete",
                )

        reference_timestamp = expected_updated_at or snapshot.get("updated_at")

        updated = self._repository.update_job(
            job_id,
            tenant_id=tenant_id,
            changes={"status": desired_status},
            expected_updated_at=reference_timestamp,
        )

        self._emit(
            "job.status_changed",
            {
                "job_id": updated["id"],
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

        if desired_status == JobStatus.SUCCEEDED:
            lifecycle_event = "completed"
            success_flag: Optional[bool] = True
        elif desired_status == JobStatus.FAILED:
            lifecycle_event = "failed"
            success_flag = False
        elif desired_status == JobStatus.CANCELLED:
            lifecycle_event = "cancelled"
            success_flag = False
        else:
            lifecycle_event = "status_changed"
            success_flag = None

        record_job_lifecycle_event(
            job_id=updated.get("id"),
            event=lifecycle_event,
            persona=_primary_persona(updated),
            tenant_id=updated.get("tenant_id"),
            from_status=current_status.value,
            to_status=desired_status.value,
            success=success_flag,
            latency_ms=latency_ms,
            timestamp=current_timestamp,
            metadata={"requested_status": desired_status.value},
        )
        return updated

    def dependencies_complete(self, job_id: Any, *, tenant_id: Any) -> bool:
        linked_tasks = self._repository.list_linked_tasks(job_id, tenant_id=tenant_id)
        statuses = [
            _normalize_task_status((link.get("task") or {}).get("status"))
            for link in linked_tasks
            if isinstance(link, Mapping)
        ]
        return all(status in _TASK_COMPLETE_STATUSES for status in statuses if status is not None)


__all__ = [
    "JobService",
    "JobServiceError",
    "JobTransitionError",
    "JobDependencyError",
    "JobNotFoundError",
    "JobConcurrencyError",
]

