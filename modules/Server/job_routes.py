"""Job oriented REST helpers used by :class:`AtlasServer`."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, Mapping, Optional, Tuple

from jsonschema import Draft202012Validator

from modules.logging.logger import setup_logger
from modules.orchestration.job_manager import JobManager
from modules.orchestration.job_scheduler import JobScheduler
from modules.orchestration.message_bus import MessageBus
from modules.job_store.repository import JobConcurrencyError, JobNotFoundError
from modules.job_store.service import (
    JobDependencyError,
    JobService,
    JobServiceError,
    JobTransitionError,
)

from .conversation_routes import RequestContext

logger = setup_logger(__name__)


class JobRouteError(RuntimeError):
    """Base class for job route errors with HTTP style codes."""

    status_code: int = 400

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class JobValidationError(JobRouteError):
    status_code = 422


class JobAuthorizationError(JobRouteError):
    status_code = 403


class JobNotFoundRouteError(JobRouteError):
    status_code = 404


class JobConflictError(JobRouteError):
    status_code = 409


class _JsonSchemaValidator:
    """Wrapper around :mod:`jsonschema` validators for friendly errors."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self._validator = Draft202012Validator(schema)

    def validate(self, payload: Mapping[str, Any]) -> None:
        errors = list(self._validator.iter_errors(payload))
        if not errors:
            return
        fragments: list[str] = []
        for error in errors:
            path = "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            )
            formatted = path.lstrip(".") or "$"
            fragments.append(f"{formatted}: {error.message}")
        fragments.sort()
        raise JobValidationError("; ".join(fragments))


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        candidate = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Datetime value cannot be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        candidate = datetime.fromisoformat(text)
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


class JobRoutes:
    """Expose CRUD and streaming helpers for job management."""

    def __init__(
        self,
        service: JobService,
        *,
        manager: JobManager | None = None,
        scheduler: JobScheduler | None = None,
        message_bus: MessageBus | None = None,
        page_size_limit: int = 100,
        poll_interval: float = 0.5,
    ) -> None:
        self._service = service
        self._manager = manager
        self._scheduler = scheduler
        self._bus = message_bus
        self._page_size_limit = max(int(page_size_limit), 1)
        self._poll_interval = max(float(poll_interval), 0.05)
        self._create_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "status": {"type": "string", "minLength": 1},
                    "owner_id": {"type": "string", "minLength": 1},
                    "conversation_id": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            }
        )
        self._update_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "owner_id": {"type": "string", "minLength": 1},
                    "conversation_id": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                    "expected_updated_at": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
                "minProperties": 1,
            }
        )
        self._list_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "status": {
                        "oneOf": [
                            {"type": "string", "minLength": 1},
                            {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "uniqueItems": True,
                            },
                        ]
                    },
                    "owner_id": {"type": "string", "minLength": 1},
                    "page_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": page_size_limit,
                    },
                    "cursor": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            }
        )
        self._link_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["task_id"],
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                    "relationship_type": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            }
        )

    def create_job(
        self,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._create_validator.validate(payload)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None

        try:
            return self._service.create_job(
                payload["name"],
                tenant_id=context.tenant_id,
                description=payload.get("description"),
                status=payload.get("status"),
                owner_id=payload.get("owner_id"),
                conversation_id=payload.get("conversation_id"),
                metadata=metadata,
            )
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc
        except JobServiceError as exc:
            raise JobRouteError(str(exc)) from exc

    def update_job(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._update_validator.validate(payload)

        changes = {key: payload[key] for key in payload if key != "expected_updated_at"}
        try:
            return self._service.update_job(
                job_id,
                tenant_id=context.tenant_id,
                changes=changes,
                expected_updated_at=payload.get("expected_updated_at"),
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except JobConcurrencyError as exc:
            raise JobConflictError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc
        except JobServiceError as exc:
            raise JobRouteError(str(exc)) from exc

    def transition_job(
        self,
        job_id: str,
        target_status: Any,
        *,
        context: RequestContext,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        try:
            return self._service.transition_job(
                job_id,
                tenant_id=context.tenant_id,
                target_status=target_status,
                expected_updated_at=expected_updated_at,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except JobDependencyError as exc:
            raise JobConflictError(str(exc)) from exc
        except JobTransitionError as exc:
            raise JobConflictError(str(exc)) from exc
        except JobConcurrencyError as exc:
            raise JobConflictError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc

    def pause_schedule(
        self,
        job_id: str,
        *,
        context: RequestContext,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        scheduler = self._require_scheduler()

        try:
            job_record = self._service.get_job(
                job_id,
                tenant_id=context.tenant_id,
                with_schedule=True,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc

        if expected_updated_at is not None:
            current = job_record.get("updated_at")
            if current is not None and str(current) != str(expected_updated_at):
                raise JobConflictError("Job has been modified since it was last refreshed")

        schedule = job_record.get("schedule")
        if not isinstance(schedule, Mapping):
            raise JobRouteError("Job does not have a configured schedule", status_code=409)

        manifest_name, persona = self._resolve_manifest_identity(job_record)
        if not manifest_name:
            raise JobRouteError("Job manifest metadata is unavailable", status_code=409)

        try:
            scheduler.pause_manifest(manifest_name, persona=persona)
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc

        updated = self._service.get_job(
            job_id,
            tenant_id=context.tenant_id,
            with_schedule=True,
        )
        return self._merge_schedule_metadata(updated)

    def rerun_job(
        self,
        job_id: str,
        *,
        context: RequestContext,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        scheduler = self._require_scheduler()

        try:
            return self._service.rerun_job(
                job_id,
                tenant_id=context.tenant_id,
                scheduler=scheduler,
                expected_updated_at=expected_updated_at,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except JobDependencyError as exc:
            raise JobConflictError(str(exc)) from exc
        except JobTransitionError as exc:
            raise JobConflictError(str(exc)) from exc
        except JobConcurrencyError as exc:
            raise JobConflictError(str(exc)) from exc
        except JobServiceError as exc:
            raise JobRouteError(str(exc)) from exc

    def resume_schedule(
        self,
        job_id: str,
        *,
        context: RequestContext,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        scheduler = self._require_scheduler()

        try:
            job_record = self._service.get_job(
                job_id,
                tenant_id=context.tenant_id,
                with_schedule=True,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc

        if expected_updated_at is not None:
            current = job_record.get("updated_at")
            if current is not None and str(current) != str(expected_updated_at):
                raise JobConflictError("Job has been modified since it was last refreshed")

        schedule = job_record.get("schedule")
        if not isinstance(schedule, Mapping):
            raise JobRouteError("Job does not have a configured schedule", status_code=409)

        manifest_name, persona = self._resolve_manifest_identity(job_record)
        if not manifest_name:
            raise JobRouteError("Job manifest metadata is unavailable", status_code=409)

        try:
            scheduler.resume_manifest(manifest_name, persona=persona)
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc

        updated = self._service.get_job(
            job_id,
            tenant_id=context.tenant_id,
            with_schedule=True,
        )
        return self._merge_schedule_metadata(updated)

    def get_job(
        self,
        job_id: str,
        *,
        context: RequestContext,
        include_schedule: bool = False,
        include_runs: bool = False,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        self._require_context(context)
        try:
            return self._service.get_job(
                job_id,
                tenant_id=context.tenant_id,
                with_schedule=include_schedule,
                with_runs=include_runs,
                with_events=include_events,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc

    def list_jobs(
        self,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        params = dict(params or {})
        self._list_validator.validate(params)

        page_size = int(params.get("page_size") or min(20, self._page_size_limit))
        page_size = max(1, min(page_size, self._page_size_limit))

        try:
            listing = self._service.list_jobs(
                tenant_id=context.tenant_id,
                status=params.get("status"),
                owner_id=params.get("owner_id"),
                cursor=params.get("cursor"),
                limit=page_size,
            )
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc

        items = listing.get("items", [])
        next_cursor = listing.get("next_cursor")
        return {
            "items": items,
            "page": {
                "next_cursor": next_cursor,
                "page_size": page_size,
                "count": len(items),
            },
        }

    def link_task(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._link_validator.validate(payload)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None

        try:
            return self._service.link_task(
                job_id,
                payload["task_id"],
                tenant_id=context.tenant_id,
                relationship_type=payload.get("relationship_type"),
                metadata=metadata,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc

    def unlink_task(
        self,
        job_id: str,
        *,
        context: RequestContext,
        link_id: Any | None = None,
        task_id: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        if link_id is None and task_id is None:
            raise JobValidationError("Either link_id or task_id must be provided")

        try:
            return self._service.unlink_task(
                job_id,
                tenant_id=context.tenant_id,
                link_id=link_id,
                task_id=task_id,
            )
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise JobValidationError(str(exc)) from exc

    def list_linked_tasks(
        self,
        job_id: str,
        *,
        context: RequestContext,
    ) -> list[Dict[str, Any]]:
        self._require_context(context)
        try:
            return self._service.list_linked_tasks(job_id, tenant_id=context.tenant_id)
        except JobNotFoundError as exc:
            raise JobNotFoundRouteError(str(exc)) from exc

    async def stream_job_events(
        self,
        job_id: str,
        *,
        context: RequestContext,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        self._require_context(context)

        if self._bus is None or not hasattr(self._bus, "subscribe"):
            async for item in self._poll_job_events(job_id, context=context, after=after):
                yield item
            return

        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        subscriptions = []

        async def _handler(topic: str, message: Any) -> None:
            payload = self._normalize_bus_payload(topic, message)
            if not self._should_emit_event(payload, context, job_id):
                return
            await queue.put(self._enrich_event(payload, context))

        for topic in self._event_topics():
            try:
                async def _subscriber(message, _topic=topic):
                    await _handler(_topic, message)

                subscription = self._bus.subscribe(topic, _subscriber)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Failed to subscribe to job topic '%s'", topic)
                continue
            subscriptions.append(subscription)

        try:
            while True:
                payload = await queue.get()
                if job_id:
                    payload.setdefault("job_id", job_id)
                payload.setdefault("tenant_id", context.tenant_id)
                yield payload
        finally:  # pragma: no cover - cleanup
            for subscription in subscriptions:
                try:
                    subscription.cancel()
                except Exception:
                    logger.exception("Error cancelling job topic subscription")

    async def _poll_job_events(
        self,
        job_id: str,
        *,
        context: RequestContext,
        after: Optional[str],
    ) -> AsyncIterator[Dict[str, Any]]:
        start_time = _parse_datetime(after) if after else datetime.now(timezone.utc)
        last_seen = start_time
        seen_ids: set[str] = set()

        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                record = self._service.get_job(
                    job_id,
                    tenant_id=context.tenant_id,
                    with_events=True,
                )
            except JobNotFoundError as exc:
                raise JobNotFoundRouteError(str(exc)) from exc

            events = record.get("events") or []
            for event in events:
                created_at = event.get("created_at")
                if created_at is None:
                    continue
                moment = _parse_datetime(created_at)
                identifier = str(event.get("id"))
                if moment < start_time:
                    continue
                if moment < last_seen or (moment == last_seen and identifier in seen_ids):
                    continue
                seen_ids.add(identifier)
                last_seen = moment
                yield self._enrich_event(event, context, job=record)

    def _require_context(self, context: Optional[RequestContext]) -> None:
        if context is None or not context.tenant_id:
            raise JobAuthorizationError("A tenant scoped context is required")

    def _require_scheduler(self) -> JobScheduler:
        if self._scheduler is None:
            raise JobRouteError("Job scheduler is not configured", status_code=503)
        return self._scheduler

    def _event_topics(self) -> Iterable[str]:
        return (
            "jobs.created",
            "jobs.updated",
            "jobs.completed",
            "job.created",
            "job.updated",
            "job.status_changed",
        )

    def _normalize_bus_payload(self, topic: str, message: Any) -> Dict[str, Any]:
        payload = getattr(message, "payload", message)
        if isinstance(payload, Mapping):
            if "event" in payload and isinstance(payload.get("data"), Mapping):
                data = dict(payload.get("data") or {})
                data.setdefault("event", str(payload.get("event") or topic))
            else:
                data = dict(payload)
                data.setdefault("event", str(payload.get("event") or topic))
        else:
            data = {"value": payload, "event": topic}
        data.setdefault("topic", topic)
        return data

    def _should_emit_event(
        self,
        payload: Mapping[str, Any],
        context: RequestContext,
        job_id: Optional[str],
    ) -> bool:
        tenant = payload.get("tenant_id")
        if tenant is not None and str(tenant) != context.tenant_id:
            return False
        identifier = payload.get("job_id")
        if job_id and identifier is not None and str(identifier) != job_id:
            return False
        if job_id and identifier in {None, ""}:
            return False
        return True

    def _enrich_event(
        self,
        event: Mapping[str, Any],
        context: RequestContext,
        *,
        job: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(event)
        payload.setdefault("tenant_id", context.tenant_id)
        if job is not None:
            job_payload = dict(job)
            job_payload.pop("events", None)
            payload.setdefault("job", job_payload)
            payload.setdefault("job_id", job_payload.get("id"))
        else:
            payload.setdefault("job_id", event.get("job_id"))
        if "event_type" not in payload and "event" in payload:
            payload["event_type"] = str(payload["event"])
        return payload

    def _resolve_manifest_identity(
        self, job_record: Mapping[str, Any]
    ) -> Tuple[str, Optional[str]]:
        metadata = job_record.get("metadata") if isinstance(job_record.get("metadata"), Mapping) else {}
        manifest_info = metadata.get("manifest") if isinstance(metadata.get("manifest"), Mapping) else {}
        manifest_name = str(manifest_info.get("name") or "").strip()
        persona_value = manifest_info.get("persona")
        if persona_value is not None:
            persona_value = str(persona_value).strip() or None

        if not manifest_name:
            raw_name = str(job_record.get("name") or "").strip()
            if raw_name:
                if "::" in raw_name:
                    base, persona_hint = raw_name.split("::", 1)
                    manifest_name = base.strip()
                    if persona_value is None and persona_hint:
                        persona_value = persona_hint.strip() or None
                else:
                    manifest_name = raw_name

        return manifest_name, persona_value

    def _merge_schedule_metadata(self, job_record: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(job_record)
        schedule = payload.get("schedule")
        if isinstance(schedule, Mapping):
            metadata = payload.get("metadata")
            metadata_payload = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata_payload["schedule"] = dict(schedule)
            schedule_meta = schedule.get("metadata")
            if isinstance(schedule_meta, Mapping):
                state = schedule_meta.get("state")
                if state is not None:
                    metadata_payload["schedule_state"] = state
            payload["metadata"] = metadata_payload
        return payload


__all__ = [
    "JobRoutes",
    "JobRouteError",
    "JobValidationError",
    "JobAuthorizationError",
    "JobNotFoundRouteError",
    "JobConflictError",
]
