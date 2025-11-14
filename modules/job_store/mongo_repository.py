"""MongoDB-backed repository implementing the job store interfaces."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency for typing/runtime support
    from pymongo.collection import Collection  # type: ignore
    from pymongo.errors import DuplicateKeyError  # type: ignore
    from pymongo.mongo_client import MongoClient  # type: ignore
except Exception:  # pragma: no cover - dependency optional in some environments
    Collection = Any  # type: ignore
    MongoClient = Any  # type: ignore
    class _DuplicateKeyError(Exception):
        pass

    DuplicateKeyError = _DuplicateKeyError  # type: ignore

from modules.store_common.repository_utils import (
    _coerce_dt,
    _coerce_optional_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_meta,
    _normalize_tenant_id,
)

from .models import JobEventType, JobRunStatus, JobStatus
from .repository import (
    JobConcurrencyError,
    JobNotFoundError,
    JobStoreError,
    JobTransitionError,
)

CollectionLike = Any

_ASCENDING = 1
_DESCENDING = -1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Job name must not be empty")
    return text


def _normalize_relationship(value: Any | None) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "relates_to"


def _normalize_status(value: Any | None) -> JobStatus:
    if value is None:
        return JobStatus.DRAFT
    if isinstance(value, JobStatus):
        return value
    text = str(value).strip()
    if not text:
        return JobStatus.DRAFT
    try:
        return JobStatus(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown job status '{value}'") from exc


def _normalize_run_status(value: Any | None) -> JobRunStatus:
    if value is None:
        return JobRunStatus.SCHEDULED
    if isinstance(value, JobRunStatus):
        return value
    text = str(value).strip()
    if not text:
        return JobRunStatus.SCHEDULED
    try:
        return JobRunStatus(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown run status '{value}'") from exc


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


def _encode_cursor_doc(document: Mapping[str, Any]) -> str:
    created_at = document.get("created_at")
    created_iso = _dt_to_iso(created_at) if isinstance(created_at, datetime) else None
    return f"{created_iso or ''}|{document.get('_id')}"


def _decode_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    if not cursor:
        return None
    if "|" not in cursor:
        raise ValueError("Cursor is malformed")
    created_at_text, job_id_text = cursor.split("|", 1)
    created_at = _coerce_dt(created_at_text) if created_at_text else _utc_now()
    job_uuid = _coerce_uuid(job_id_text) or uuid.uuid4()
    return created_at, str(job_uuid)


def _serialize_job(document: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(document.get("_id")),
        "name": document.get("name"),
        "description": document.get("description"),
        "status": document.get("status"),
        "owner_id": str(document.get("owner_id")) if document.get("owner_id") else None,
        "conversation_id": str(document.get("conversation_id")) if document.get("conversation_id") else None,
        "tenant_id": document.get("tenant_id"),
        "metadata": dict(document.get("metadata", {})),
        "created_at": _dt_to_iso(document.get("created_at")),
        "updated_at": _dt_to_iso(document.get("updated_at")),
    }
    return payload


def _serialize_schedule(document: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(document.get("_id")),
        "job_id": str(document.get("job_id")),
        "schedule_type": document.get("schedule_type"),
        "expression": document.get("expression"),
        "timezone": document.get("timezone"),
        "next_run_at": _dt_to_iso(document.get("next_run_at")),
        "metadata": dict(document.get("metadata", {})),
        "created_at": _dt_to_iso(document.get("created_at")),
        "updated_at": _dt_to_iso(document.get("updated_at")),
    }


def _serialize_run(document: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(document.get("_id")),
        "job_id": str(document.get("job_id")),
        "run_number": int(document.get("run_number", 0)),
        "status": document.get("status"),
        "started_at": _dt_to_iso(document.get("started_at")),
        "finished_at": _dt_to_iso(document.get("finished_at")),
        "metadata": dict(document.get("metadata", {})),
        "created_at": _dt_to_iso(document.get("created_at")),
        "updated_at": _dt_to_iso(document.get("updated_at")),
    }


def _serialize_event(document: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(document.get("_id")),
        "job_id": str(document.get("job_id")),
        "event_type": document.get("event_type"),
        "triggered_by_id": str(document.get("triggered_by_id")) if document.get("triggered_by_id") else None,
        "session_id": str(document.get("session_id")) if document.get("session_id") else None,
        "payload": dict(document.get("payload", {})),
        "created_at": _dt_to_iso(document.get("created_at")),
    }


def _serialize_task_link(
    document: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(document.get("_id")),
        "job_id": str(document.get("job_id")),
        "task_id": str(document.get("task_id")),
        "relationship_type": document.get("relationship_type"),
        "metadata": dict(document.get("metadata", {})),
        "created_at": _dt_to_iso(document.get("created_at")),
    }
    if task is not None:
        payload["task"] = dict(task)
    return payload


TaskLoader = Callable[[uuid.UUID, str], Mapping[str, Any] | None]


@dataclass
class MongoJobStoreRepository:
    """MongoDB implementation of the job store repository interface."""

    jobs: CollectionLike
    schedules: CollectionLike
    runs: CollectionLike
    events: CollectionLike
    task_links: CollectionLike
    client: MongoClient | None = None
    task_loader: TaskLoader | None = None

    def __post_init__(self) -> None:
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Construction helpers

    @classmethod
    def from_database(
        cls,
        database: Any,
        *,
        client: MongoClient | None = None,
        task_loader: TaskLoader | None = None,
        jobs_collection: str = "jobs",
        schedules_collection: str = "job_schedules",
        runs_collection: str = "job_runs",
        events_collection: str = "job_events",
        task_links_collection: str = "job_task_links",
    ) -> "MongoJobStoreRepository":
        if database is None:
            raise ValueError("database handle is required to build a Mongo repository")

        jobs = database.get_collection(jobs_collection)
        schedules = database.get_collection(schedules_collection)
        runs = database.get_collection(runs_collection)
        events = database.get_collection(events_collection)
        task_links = database.get_collection(task_links_collection)

        return cls(
            jobs=jobs,
            schedules=schedules,
            runs=runs,
            events=events,
            task_links=task_links,
            client=client,
            task_loader=task_loader,
        )

    # ------------------------------------------------------------------
    # Index helpers

    _JOB_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("tenant_id", _ASCENDING), ("name", _ASCENDING)), {"unique": True}),
        ((("tenant_id", _ASCENDING), ("created_at", _DESCENDING)), {}),
        ((("status", _ASCENDING), ("created_at", _DESCENDING)), {}),
    )

    _SCHEDULE_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("job_id", _ASCENDING),), {"unique": True}),
        ((("next_run_at", _ASCENDING),), {}),
    )

    _RUN_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("job_id", _ASCENDING), ("run_number", _DESCENDING)), {}),
        ((("job_id", _ASCENDING), ("status", _ASCENDING)), {}),
    )

    _EVENT_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("job_id", _ASCENDING), ("created_at", _ASCENDING)), {}),
    )

    _TASK_LINK_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("job_id", _ASCENDING), ("task_id", _ASCENDING)), {"unique": True}),
        ((("task_id", _ASCENDING),), {}),
    )

    def _ensure_indexes(self) -> None:
        self._ensure_collection_indexes(self.jobs, self._JOB_INDEXES)
        self._ensure_collection_indexes(self.schedules, self._SCHEDULE_INDEXES)
        self._ensure_collection_indexes(self.runs, self._RUN_INDEXES)
        self._ensure_collection_indexes(self.events, self._EVENT_INDEXES)
        self._ensure_collection_indexes(self.task_links, self._TASK_LINK_INDEXES)

    def _ensure_collection_indexes(
        self,
        collection: CollectionLike,
        definitions: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]],
    ) -> None:
        if collection is None:
            return
        create_index = getattr(collection, "create_index", None)
        if not callable(create_index):
            return
        for keys, options in definitions:
            key_spec = list(keys)
            create_index(key_spec, **dict(options))

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _new_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _normalize_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
        return _normalize_meta(metadata)

    # ------------------------------------------------------------------
    # Job helpers

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
        limit = max(1, min(int(limit or 50), 100))

        query: Dict[str, Any] = {"tenant_id": tenant_key}

        if status is not None:
            if isinstance(status, (list, tuple, set)) and not isinstance(status, (str, bytes)):
                normalized = [_normalize_status(item).value for item in status]
            else:
                normalized = [_normalize_status(status).value]
            query["status"] = {"$in": normalized}

        owner_uuid = _coerce_uuid(owner_id)
        if owner_uuid is not None:
            query["owner_id"] = str(owner_uuid)

        cursor_state = _decode_cursor(cursor)
        if cursor_state is not None:
            created_at, job_id = cursor_state
            query["$or"] = [
                {"created_at": {"$lt": created_at}},
                {"$and": [{"created_at": created_at}, {"_id": {"$lt": job_id}}]},
            ]

        records = list(
            self.jobs.find(query)
            .sort([("created_at", _DESCENDING), ("_id", _DESCENDING)])
            .limit(limit + 1)
        )
        items = records[:limit]
        next_cursor = None
        if len(records) > limit and items:
            next_cursor = _encode_cursor_doc(items[-1])

        return {"items": [_serialize_job(doc) for doc in items], "next_cursor": next_cursor}

    def get_job_by_name(
        self,
        name: Any,
        *,
        tenant_id: Any,
    ) -> Dict[str, Any]:
        tenant_key = _normalize_tenant_id(tenant_id)
        name_text = _normalize_name(name)
        document = self.jobs.find_one({"tenant_id": tenant_key, "name": name_text})
        if document is None:
            raise JobNotFoundError("Job not found for tenant")
        return _serialize_job(document)

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

        document = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if document is None:
            raise JobNotFoundError("Job not found for tenant")

        payload = _serialize_job(document)

        if with_schedule:
            schedule = self.schedules.find_one({"job_id": payload["id"]})
            if schedule:
                payload["schedule"] = _serialize_schedule(schedule)

        if with_runs:
            runs = list(
                self.runs.find({"job_id": payload["id"]})
                .sort([("run_number", _DESCENDING), ("_id", _DESCENDING)])
            )
            payload["runs"] = [_serialize_run(run) for run in runs]

        if with_events:
            events = list(
                self.events.find({"job_id": payload["id"]}).sort([("created_at", _ASCENDING)])
            )
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
        metadata_dict = self._normalize_metadata(metadata)

        now = _utc_now()
        job_id = self._new_uuid()
        document = {
            "_id": job_id,
            "name": name_text,
            "description": str(description).strip() if description is not None else None,
            "status": status_value.value,
            "owner_id": str(owner_uuid) if owner_uuid else None,
            "conversation_id": str(conversation_uuid) if conversation_uuid else None,
            "tenant_id": tenant_key,
            "metadata": metadata_dict,
            "created_at": now,
            "updated_at": now,
        }

        try:
            self.jobs.insert_one(document)
        except DuplicateKeyError as exc:  # pragma: no cover - unique constraint enforcement
            raise ValueError("Job name already exists for tenant") from exc

        event_doc = {
            "_id": self._new_uuid(),
            "job_id": job_id,
            "event_type": JobEventType.CREATED.value,
            "payload": {"status": status_value.value},
            "created_at": now,
        }
        self.events.insert_one(event_doc)

        payload = _serialize_job(document)
        payload["events"] = [_serialize_event(event_doc)]
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

        document = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if document is None:
            raise JobNotFoundError("Job not found for tenant")

        if expected_updated_at is not None:
            expected = _coerce_dt(expected_updated_at)
            stored = document.get("updated_at")
            stored_dt = stored if isinstance(stored, datetime) else _coerce_dt(stored)
            if stored_dt != expected:
                raise JobConcurrencyError("Job was modified by another transaction")

        updates: Dict[str, Any] = {}
        events: list[Dict[str, Any]] = []
        now = _utc_now()

        status_before = JobStatus(document.get("status", JobStatus.DRAFT.value))
        status_after = status_before
        status_changed = False

        for field, value in changes.items():
            if field == "name":
                updates["name"] = _normalize_name(value)
            elif field == "description":
                updates["description"] = str(value).strip() if value is not None else None
            elif field == "status":
                new_status = _normalize_status(value)
                if new_status != status_before:
                    if not _valid_job_transition(status_before, new_status):
                        raise JobTransitionError("Unsupported job status transition")
                    status_after = new_status
                    status_changed = True
                    updates["status"] = new_status.value
            elif field == "owner_id":
                owner_uuid = _coerce_uuid(value)
                updates["owner_id"] = str(owner_uuid) if owner_uuid else None
            elif field == "conversation_id":
                conversation_uuid = _coerce_uuid(value)
                updates["conversation_id"] = str(conversation_uuid) if conversation_uuid else None
            elif field == "metadata":
                updates["metadata"] = self._normalize_metadata(value)
            else:
                raise ValueError(f"Unsupported job attribute: {field}")

        if not updates and not status_changed:
            return _serialize_job(document)

        updates["updated_at"] = now
        self.jobs.update_one({"_id": str(job_uuid)}, {"$set": updates})

        document.update(updates)

        change_payload = dict(changes)
        change_event = {
            "_id": self._new_uuid(),
            "job_id": str(job_uuid),
            "event_type": JobEventType.UPDATED.value,
            "payload": {"changes": change_payload},
            "created_at": now,
        }
        events.append(change_event)

        if status_changed:
            events.append(
                {
                    "_id": self._new_uuid(),
                    "job_id": str(job_uuid),
                    "event_type": JobEventType.STATUS_CHANGED.value,
                    "payload": {
                        "from": status_before.value,
                        "to": status_after.value,
                    },
                    "created_at": now,
                }
            )

        if events:
            self.events.insert_many(events)

        payload = _serialize_job(document)
        payload["events"] = [_serialize_event(event) for event in events]
        return payload

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        schedule_type_text = str(schedule_type).strip() or "cron"
        expression_text = str(expression).strip()
        if not expression_text:
            raise ValueError("Schedule expression must not be empty")
        timezone_text = str(timezone_name).strip() if timezone_name else "UTC"
        metadata_dict = self._normalize_metadata(metadata)
        next_run_dt = _coerce_optional_dt(next_run_at)
        now = _utc_now()

        existing = self.schedules.find_one({"job_id": str(job_uuid)})
        payload = {
            "job_id": str(job_uuid),
            "schedule_type": schedule_type_text,
            "expression": expression_text,
            "timezone": timezone_text,
            "next_run_at": next_run_dt,
            "metadata": metadata_dict,
            "updated_at": now,
        }
        if existing is None:
            payload.update({"_id": self._new_uuid(), "created_at": now})
            self.schedules.insert_one(payload)
            record = payload
        else:
            payload["created_at"] = existing.get("created_at", now)
            self.schedules.update_one({"_id": existing.get("_id")}, {"$set": payload})
            record = {**existing, **payload}
        return _serialize_schedule(record)

    # ------------------------------------------------------------------
    # Run helpers

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        status_value = _normalize_run_status(status)
        started_dt = _coerce_optional_dt(started_at)
        finished_dt = _coerce_optional_dt(finished_at)
        metadata_dict = self._normalize_metadata(metadata)

        last_run = (
            self.runs.find({"job_id": str(job_uuid)})
            .sort([("run_number", _DESCENDING)])
            .limit(1)
        )
        next_number = 1
        try:
            last_document = next(iter(last_run))
        except StopIteration:
            last_document = None
        if last_document is not None:
            next_number = int(last_document.get("run_number", 0)) + 1

        now = _utc_now()
        run_id = self._new_uuid()
        document = {
            "_id": run_id,
            "job_id": str(job_uuid),
            "run_number": next_number,
            "status": status_value.value,
            "started_at": started_dt,
            "finished_at": finished_dt,
            "metadata": metadata_dict,
            "created_at": now,
            "updated_at": now,
        }
        self.runs.insert_one(document)

        self.events.insert_one(
            {
                "_id": self._new_uuid(),
                "job_id": str(job_uuid),
                "event_type": JobEventType.RUN.value,
                "payload": {
                    "run_number": next_number,
                    "status": status_value.value,
                },
                "created_at": now,
            }
        )

        return _serialize_run(document)

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

        run_document = self.runs.find_one({"_id": str(run_uuid)})
        if run_document is None:
            raise JobNotFoundError("Job run not found for tenant")

        job_id = run_document.get("job_id")
        job = self.jobs.find_one({"_id": job_id, "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        if expected_updated_at is not None:
            expected = _coerce_dt(expected_updated_at)
            stored = run_document.get("updated_at")
            stored_dt = stored if isinstance(stored, datetime) else _coerce_dt(stored)
            if stored_dt != expected:
                raise JobConcurrencyError("Run was modified by another transaction")

        updates: Dict[str, Any] = {}
        status_before = JobRunStatus(run_document.get("status", JobRunStatus.SCHEDULED.value))
        for field, value in changes.items():
            if field == "status":
                new_status = _normalize_run_status(value)
                if new_status != status_before and not _valid_run_transition(status_before, new_status):
                    raise JobTransitionError("Unsupported run status transition")
                updates["status"] = new_status.value
            elif field == "started_at":
                updates["started_at"] = _coerce_optional_dt(value)
            elif field == "finished_at":
                updates["finished_at"] = _coerce_optional_dt(value)
            elif field == "metadata":
                updates["metadata"] = self._normalize_metadata(value)
            else:
                raise ValueError(f"Unsupported run attribute: {field}")

        updates["updated_at"] = _utc_now()
        self.runs.update_one({"_id": str(run_uuid)}, {"$set": updates})
        run_document.update(updates)
        return _serialize_run(run_document)

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        limit = max(1, min(int(limit or 50), 100))
        query: Dict[str, Any] = {"job_id": str(job_uuid)}

        run_cursor: Optional[tuple[int, str]] = None
        if cursor:
            if "|" not in cursor:
                raise ValueError("Cursor is malformed")
            number_text, run_id_text = cursor.split("|", 1)
            run_cursor = (int(number_text), str(_coerce_uuid(run_id_text) or uuid.uuid4()))

        if run_cursor is not None:
            run_number, run_identifier = run_cursor
            query["$or"] = [
                {"run_number": {"$lt": run_number}},
                {"$and": [{"run_number": run_number}, {"_id": {"$lt": run_identifier}}]},
            ]

        records = list(
            self.runs.find(query).sort([("run_number", _DESCENDING), ("_id", _DESCENDING)]).limit(limit + 1)
        )
        items = records[:limit]
        next_cursor = None
        if len(records) > limit and items:
            last = items[-1]
            next_cursor = f"{int(last.get('run_number', 0))}|{last.get('_id')}"

        return {"items": [_serialize_run(doc) for doc in items], "next_cursor": next_cursor}

    # ------------------------------------------------------------------
    # Event helpers

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        now = _utc_now()
        event_document = {
            "_id": self._new_uuid(),
            "job_id": str(job_uuid),
            "event_type": event_type.value if isinstance(event_type, JobEventType) else str(event_type),
            "triggered_by_id": str(_coerce_uuid(triggered_by_id)) if triggered_by_id else None,
            "session_id": str(_coerce_uuid(session_id)) if session_id else None,
            "payload": dict(payload or {}),
            "created_at": now,
        }
        self.events.insert_one(event_document)
        return _serialize_event(event_document)

    # ------------------------------------------------------------------
    # Task linking helpers

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        link = self.task_links.find_one({"job_id": str(job_uuid), "task_id": str(task_uuid)})
        if link is not None:
            return _serialize_task_link(
                link,
                task=self._load_task_snapshot(task_uuid, tenant_key),
            )

        now = _utc_now()
        document = {
            "_id": self._new_uuid(),
            "job_id": str(job_uuid),
            "task_id": str(task_uuid),
            "relationship_type": _normalize_relationship(relationship_type),
            "metadata": self._normalize_metadata(metadata),
            "created_at": now,
        }
        try:
            self.task_links.insert_one(document)
        except DuplicateKeyError as exc:  # pragma: no cover - unique constraint enforcement
            raise ValueError("Task is already linked to this job") from exc
        return _serialize_task_link(
            document,
            task=self._load_task_snapshot(task_uuid, tenant_key),
        )

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        links = list(
            self.task_links.find({"job_id": str(job_uuid)}).sort([("created_at", _ASCENDING)])
        )
        results: list[Dict[str, Any]] = []
        for link in links:
            task_snapshot = self._load_task_snapshot(_coerce_uuid(link.get("task_id")), tenant_key)
            results.append(_serialize_task_link(link, task=task_snapshot))
        return results

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

        job = self.jobs.find_one({"_id": str(job_uuid), "tenant_id": tenant_key})
        if job is None:
            raise JobNotFoundError("Job not found for tenant")

        if link_id is None and task_id is None:
            raise ValueError("A link_id or task_id must be supplied")

        query: Dict[str, Any] = {"job_id": str(job_uuid)}
        link_uuid = _coerce_uuid(link_id) if link_id is not None else None
        task_uuid = _coerce_uuid(task_id) if task_id is not None else None
        if link_uuid is not None:
            query["_id"] = str(link_uuid)
        if task_uuid is not None:
            query["task_id"] = str(task_uuid)

        link = self.task_links.find_one(query)
        if link is None:
            raise ValueError("Linked task was not found for this job")

        self.task_links.delete_one({"_id": link.get("_id")})
        task_snapshot = self._load_task_snapshot(_coerce_uuid(link.get("task_id")), tenant_key)
        return _serialize_task_link(link, task=task_snapshot)

    # ------------------------------------------------------------------
    # Internal helpers

    def _load_task_snapshot(
        self,
        task_id: uuid.UUID | None,
        tenant_id: str,
    ) -> Mapping[str, Any] | None:
        if task_id is None or self.task_loader is None:
            return None
        try:
            snapshot = self.task_loader(task_id, tenant_id)
        except Exception as exc:  # pragma: no cover - defensive logging support
            raise JobStoreError("Task loader failed to resolve linked task") from exc
        return snapshot


__all__ = ["MongoJobStoreRepository"]

