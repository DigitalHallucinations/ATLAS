"""Job scheduling integration between manifests and the task queue."""

from __future__ import annotations

import asyncio
import datetime as dt
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from modules.Jobs.manifest_loader import JobMetadata
from modules.job_store import JobRunStatus, JobStatus
from modules.job_store.repository import JobNotFoundError, JobStoreRepository
from modules.orchestration.job_manager import JobManager
from modules.Tools.Base_Tools.task_queue import (
    RetryPolicy,
    TaskEvent,
    TaskQueueService,
)


@dataclass
class _Registration:
    job_id: str
    manifest_name: str
    persona: Optional[str]
    schedule_type: str
    expression: str
    timezone: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobScheduler:
    """Bridge job manifests with the durable task queue scheduler."""

    def __init__(
        self,
        job_manager: JobManager,
        task_queue: TaskQueueService,
        repository: JobStoreRepository,
        *,
        tenant_id: str = "default",
    ) -> None:
        self._job_manager = job_manager
        self._task_queue = task_queue
        self._repository = repository
        self._tenant_id = tenant_id
        self._lock = threading.RLock()
        self._registrations: Dict[str, _Registration] = {}
        self._manifest_index: Dict[Tuple[Optional[str], str], str] = {}

        self._task_queue.add_monitor(self._handle_task_event)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_manifest(
        self,
        metadata: JobMetadata,
        *,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> Mapping[str, Any]:
        """Register a manifest on the task queue and persist its schedule."""

        schedule_type, expression, cron_fields, timezone = _resolve_schedule(metadata.recurrence)
        persona_key = metadata.persona or None
        job_name = self._resolve_job_name(metadata)
        payload = {
            "job_name": metadata.name,
            "persona": metadata.persona,
            "source": metadata.source,
        }

        try:
            job_record = self._repository.get_job_by_name(job_name, tenant_id=self._tenant_id)
        except JobNotFoundError:
            job_record = self._repository.create_job(
                job_name,
                tenant_id=self._tenant_id,
                description=metadata.summary or metadata.description,
                status=JobStatus.SCHEDULED,
                metadata={
                    "manifest": {
                        "name": metadata.name,
                        "persona": metadata.persona,
                        "source": metadata.source,
                    },
                    "recurrence": dict(metadata.recurrence),
                },
            )
        else:
            if job_record.get("status") != JobStatus.SCHEDULED.value:
                job_record = self._repository.update_job(
                    job_record["id"],
                    tenant_id=self._tenant_id,
                    changes={"status": JobStatus.SCHEDULED},
                )

        idempotency_key = f"manifest::{persona_key or 'default'}::{metadata.name}"

        if cron_fields is not None:
            queue_status = self._task_queue.schedule_cron(
                name=job_name,
                cron_fields=cron_fields,
                payload=payload,
                idempotency_key=idempotency_key,
                retry_policy=retry_policy,
            )
        else:
            queue_status = self._task_queue.schedule_cron(
                name=job_name,
                cron_schedule=expression,
                payload=payload,
                idempotency_key=idempotency_key,
                retry_policy=retry_policy,
            )

        metadata_payload = {
            "task_queue_job_id": queue_status.get("job_id"),
            "state": queue_status.get("state"),
            "retry_policy": queue_status.get("retry_policy"),
            "recurrence": dict(metadata.recurrence),
        }

        schedule_record = self._repository.upsert_schedule(
            job_record["id"],
            tenant_id=self._tenant_id,
            schedule_type=schedule_type,
            expression=expression,
            timezone_name=timezone,
            next_run_at=queue_status.get("next_run_time"),
            metadata=metadata_payload,
        )

        registration = _Registration(
            job_id=job_record["id"],
            manifest_name=metadata.name,
            persona=metadata.persona,
            schedule_type=schedule_type,
            expression=expression,
            timezone=timezone,
            metadata=metadata_payload,
        )

        queue_job_id = queue_status.get("job_id")
        with self._lock:
            if queue_job_id:
                self._registrations[queue_job_id] = registration
            self._manifest_index[(persona_key, metadata.name)] = job_record["id"]

        return schedule_record

    def apply_override(
        self,
        manifest_name: str,
        *,
        persona: Optional[str] = None,
        next_run_at: Optional[dt.datetime] = None,
        retry_policy: Optional[Mapping[str, Any]] = None,
        state: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Persist manual overrides to the schedule metadata."""

        registration = self._lookup_registration(manifest_name, persona)
        if registration is None:
            raise JobNotFoundError("Manifest has not been scheduled")

        metadata_payload = dict(registration.metadata)
        if retry_policy is not None:
            metadata_payload["retry_policy"] = dict(retry_policy)
        if state is not None:
            metadata_payload["state"] = state

        record = self._repository.upsert_schedule(
            registration.job_id,
            tenant_id=self._tenant_id,
            schedule_type=registration.schedule_type,
            expression=registration.expression,
            timezone_name=registration.timezone,
            next_run_at=next_run_at,
            metadata=metadata_payload,
        )

        with self._lock:
            registration.metadata = metadata_payload

        return record

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------
    def build_executor(self):
        """Return a callable compatible with :class:`TaskQueueService`."""

        def _executor(context: Mapping[str, Any]) -> None:
            payload = context.get("payload", {}) if isinstance(context, Mapping) else {}
            job_name = payload.get("job_name") or context.get("name")
            persona = payload.get("persona")
            run_id = context.get("job_id")

            if not job_name:
                return

            result = self._job_manager.run_job(job_name, persona=persona, run_id=run_id)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)

        return _executor

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_task_event(self, event: TaskEvent) -> None:
        with self._lock:
            registration = self._registrations.get(event.job_id)

        if registration is None:
            return

        metadata_payload = dict(registration.metadata)
        metadata_payload.update(
            {
                "task_queue_job_id": event.job_id,
                "state": event.state,
                "attempt": event.attempt,
                "retry_policy": _serialize_retry_policy(
                    metadata_payload.get("retry_policy")
                    or event.metadata.get("policy")
                    or self._task_queue.get_retry_policy()
                ),
            }
        )
        if event.error:
            metadata_payload["error"] = str(event.error)
        else:
            metadata_payload.pop("error", None)
        if event.metadata.get("last_run_at"):
            metadata_payload["last_run_at"] = _ensure_iso_datetime(event.metadata["last_run_at"])

        self._repository.upsert_schedule(
            registration.job_id,
            tenant_id=self._tenant_id,
            schedule_type=registration.schedule_type,
            expression=registration.expression,
            timezone_name=registration.timezone,
            next_run_at=event.next_run_time,
            metadata=metadata_payload,
        )

        registration.metadata = metadata_payload
        self._record_run(registration, event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_run(self, registration: _Registration, event: TaskEvent) -> None:
        status = _map_event_state(event.state)
        if status is None:
            return

        started_at = _coerce_datetime(event.metadata.get("last_run_at")) or event.timestamp
        run_payload = {
            "attempt": event.attempt,
            "scheduler_state": event.state,
            "payload": dict(event.payload),
        }
        if event.error:
            run_payload["error"] = str(event.error)

        run_record = self._repository.create_run(
            registration.job_id,
            tenant_id=self._tenant_id,
            status=JobRunStatus.RUNNING,
            started_at=started_at,
            finished_at=None,
            metadata=run_payload,
        )

        self._repository.update_run(
            run_record["id"],
            tenant_id=self._tenant_id,
            changes={
                "status": status,
                "finished_at": event.timestamp,
                "metadata": dict(run_payload, final_status=status.value),
            },
        )

    def _lookup_registration(
        self,
        manifest_name: str,
        persona: Optional[str],
    ) -> Optional[_Registration]:
        key = (persona or None, manifest_name)
        with self._lock:
            job_id = self._manifest_index.get(key)
            if job_id is None:
                return None
            for registration in self._registrations.values():
                if registration.job_id == job_id:
                    return registration
        return None

    @staticmethod
    def _resolve_job_name(metadata: JobMetadata) -> str:
        persona_key = metadata.persona or "default"
        return f"{metadata.name}::{persona_key}"


def _resolve_schedule(recurrence: Mapping[str, Any]) -> Tuple[str, str, Optional[Mapping[str, Any]], str]:
    if not isinstance(recurrence, Mapping):
        raise ValueError("Job recurrence must be a mapping")

    timezone = str(recurrence.get("timezone") or "UTC")
    cron_expression = recurrence.get("cron")
    if isinstance(cron_expression, str) and cron_expression.strip():
        return "cron", cron_expression.strip(), None, timezone

    frequency = str(recurrence.get("frequency") or "").lower().strip()
    interval = int(recurrence.get("interval") or 1)
    start_dt = _coerce_datetime(recurrence.get("start_date"))
    hour = start_dt.hour if start_dt else 0
    minute = start_dt.minute if start_dt else 0

    if frequency == "hourly":
        fields = {"minute": minute, "hour": f"*/{max(1, interval)}"}
    elif frequency == "daily":
        day_field = "*" if interval <= 1 else f"*/{interval}"
        fields = {"minute": minute, "hour": hour, "day": day_field}
    elif frequency == "weekly":
        weekday = start_dt.weekday() if start_dt else 0
        fields = {"minute": minute, "hour": hour, "day_of_week": str(weekday)}
        if interval > 1:
            fields["week"] = f"*/{interval}"
    elif frequency == "monthly":
        day = start_dt.day if start_dt else 1
        fields = {"minute": minute, "hour": hour, "day": day, "month": f"*/{max(1, interval)}"}
    else:
        raise ValueError("Unsupported recurrence frequency; specify a cron expression")

    return "cron", _summarize_fields(fields), fields, timezone


def _summarize_fields(fields: Mapping[str, Any]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(fields.items()))


def _serialize_retry_policy(policy: Any) -> Mapping[str, Any]:
    if isinstance(policy, RetryPolicy):
        return policy.to_dict()
    if isinstance(policy, Mapping):
        return {str(key): value for key, value in policy.items()}
    return {}


def _ensure_iso_datetime(value: Any) -> str:
    dt_value = _coerce_datetime(value)
    if dt_value is None:
        return ""
    return dt_value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_datetime(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, dt.datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=dt.timezone.utc)
    return result


def _map_event_state(state: str) -> Optional[JobRunStatus]:
    mapping = {
        "succeeded": JobRunStatus.SUCCEEDED,
        "failed": JobRunStatus.FAILED,
        "retrying": JobRunStatus.FAILED,
        "cancelled": JobRunStatus.CANCELLED,
        "missed": JobRunStatus.FAILED,
    }
    return mapping.get(str(state or "").lower())

