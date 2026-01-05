"""Task oriented REST helpers used by :class:`AtlasServer`."""

from __future__ import annotations

import asyncio
import base64
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, Mapping, Optional, Sequence

from jsonschema import Draft202012Validator

from modules.logging.logger import setup_logger
from core.messaging import AgentBus
from modules.task_store.repository import (
    TaskConcurrencyError,
    TaskNotFoundError,
)
from modules.task_store.service import (
    TaskDependencyError,
    TaskService,
    TaskServiceError,
    TaskTransitionError,
)

from .conversation_routes import RequestContext

logger = setup_logger(__name__)


class TaskRouteError(RuntimeError):
    """Base class for task route errors with HTTP style codes."""

    status_code: int = 400

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class TaskValidationError(TaskRouteError):
    status_code = 422


class TaskAuthorizationError(TaskRouteError):
    status_code = 403


class TaskNotFoundRouteError(TaskRouteError):
    status_code = 404


class TaskConflictError(TaskRouteError):
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
        raise TaskValidationError("; ".join(fragments))


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


def _encode_cursor(created_at: str, task_id: str) -> str:
    token = f"{created_at}|{task_id}"
    raw = token.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return encoded.rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - validation error reporting
        raise TaskValidationError("Invalid pagination cursor supplied") from exc
    try:
        timestamp_text, task_id_text = decoded.split("|", 1)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise TaskValidationError("Malformed pagination cursor") from exc
    try:
        timestamp = _parse_datetime(timestamp_text)
        task_id = uuid.UUID(task_id_text)
    except (TypeError, ValueError) as exc:  # noqa: BLE001 - validation error reporting
        raise TaskValidationError("Invalid pagination cursor supplied") from exc

    return timestamp, task_id


class TaskRoutes:
    """Expose CRUD and search operations for task management."""

    def __init__(
        self,
        service: TaskService,
        *,
        agent_bus: AgentBus | None = None,
        page_size_limit: int = 100,
        poll_interval: float = 0.5,
    ) -> None:
        self._service = service
        self._bus = agent_bus
        self._page_size_limit = max(int(page_size_limit), 1)
        self._poll_interval = max(float(poll_interval), 0.05)
        self._create_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["title", "conversation_id"],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "status": {"type": "string", "minLength": 1},
                    "priority": {"type": "integer"},
                    "owner_id": {"type": "string", "minLength": 1},
                    "session_id": {"type": "string", "minLength": 1},
                    "conversation_id": {"type": "string", "minLength": 1},
                    "due_at": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                },
                "additionalProperties": False,
            }
        )
        self._update_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "status": {"type": "string", "minLength": 1},
                    "priority": {"type": "integer"},
                    "owner_id": {"type": "string", "minLength": 1},
                    "session_id": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                    "due_at": {"type": "string", "minLength": 1},
                    "expected_updated_at": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
                "anyOf": [
                    {"required": ["title"]},
                    {"required": ["description"]},
                    {"required": ["status"]},
                    {"required": ["priority"]},
                    {"required": ["owner_id"]},
                    {"required": ["session_id"]},
                    {"required": ["metadata"]},
                    {"required": ["due_at"]},
                ],
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
                    "conversation_id": {"type": "string", "minLength": 1},
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
        self._search_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
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
                    "conversation_id": {"type": "string", "minLength": 1},
                    "metadata": {"type": "object"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": page_size_limit,
                    },
                    "offset": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            }
        )

    def create_task(
        self,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._create_validator.validate(payload)

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None

        try:
            record = self._service.create_task(
                payload["title"],
                tenant_id=context.tenant_id,
                description=payload.get("description"),
                status=payload.get("status"),
                priority=payload.get("priority"),
                owner_id=payload.get("owner_id"),
                session_id=payload.get("session_id"),
                conversation_id=payload["conversation_id"],
                due_at=payload.get("due_at"),
                metadata=metadata,
            )
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc
        except TaskServiceError as exc:
            raise TaskRouteError(str(exc)) from exc

        self._emit_events(record.get("events") or [], context)
        return record

    def update_task(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._update_validator.validate(payload)

        changes = {key: payload[key] for key in payload if key != "expected_updated_at"}
        try:
            record = self._service.update_task(
                task_id,
                tenant_id=context.tenant_id,
                changes=changes,
                expected_updated_at=payload.get("expected_updated_at"),
            )
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc
        except TaskConcurrencyError as exc:
            raise TaskConflictError(str(exc)) from exc
        except TaskNotFoundError as exc:
            raise TaskNotFoundRouteError(str(exc)) from exc
        except TaskServiceError as exc:
            raise TaskRouteError(str(exc)) from exc

        self._emit_events(record.get("events") or [], context)
        return record

    def transition_task(
        self,
        task_id: str,
        target_status: Any,
        *,
        context: RequestContext,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        self._require_context(context)
        try:
            record = self._service.transition_task(
                task_id,
                tenant_id=context.tenant_id,
                target_status=target_status,
                expected_updated_at=expected_updated_at,
            )
        except TaskNotFoundError as exc:
            raise TaskNotFoundRouteError(str(exc)) from exc
        except (TaskTransitionError, TaskDependencyError) as exc:
            raise TaskConflictError(str(exc)) from exc
        except TaskConcurrencyError as exc:
            raise TaskConflictError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc
        except TaskServiceError as exc:
            raise TaskRouteError(str(exc)) from exc

        self._emit_events(record.get("events") or [], context)
        return record

    def get_task(
        self,
        task_id: str,
        *,
        context: RequestContext,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        self._require_context(context)
        try:
            return self._service.get_task(
                task_id,
                tenant_id=context.tenant_id,
                with_events=include_events,
            )
        except TaskNotFoundError as exc:
            raise TaskNotFoundRouteError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc

    def list_tasks(
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

        cursor = params.get("cursor")
        cursor_marker: tuple[datetime, uuid.UUID] | None = None
        if cursor:
            cursor_marker = _decode_cursor(str(cursor))

        try:
            tasks = self._service.list_tasks(
                tenant_id=context.tenant_id,
                status=params.get("status"),
                owner_id=params.get("owner_id"),
                conversation_id=params.get("conversation_id"),
                limit=page_size + 1,
                cursor=cursor_marker,
            )
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc

        page_items = tasks[:page_size]
        next_cursor: Optional[str] = None
        if len(tasks) > page_size and page_items:
            last = page_items[-1]
            next_cursor = _encode_cursor(last["created_at"], last["id"])

        return {
            "items": page_items,
            "page": {
                "next_cursor": next_cursor,
                "page_size": page_size,
                "count": len(page_items),
            },
        }

    def search_tasks(
        self,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._search_validator.validate(payload)

        try:
            tasks = self._service.list_tasks(
                tenant_id=context.tenant_id,
                status=payload.get("status"),
                owner_id=payload.get("owner_id"),
                conversation_id=payload.get("conversation_id"),
            )
        except (ValueError, TypeError) as exc:
            raise TaskValidationError(str(exc)) from exc

        text_query = str(payload.get("text") or "").strip().lower()
        metadata_filter = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None

        def _matches(item: Mapping[str, Any]) -> bool:
            if text_query:
                haystack = " ".join(
                    str(piece or "").lower()
                    for piece in [item.get("title"), item.get("description")]
                )
                if text_query not in haystack:
                    return False
            if metadata_filter:
                metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
                for key, value in metadata_filter.items():
                    if metadata.get(key) != value:
                        return False
            return True

        filtered = [item for item in tasks if _matches(item)]

        offset = int(payload.get("offset") or 0)
        limit = int(payload.get("limit") or min(self._page_size_limit, 20))
        limit = max(1, min(limit, self._page_size_limit))

        sliced = filtered[offset : offset + limit]
        return {"count": len(filtered), "items": sliced}

    async def stream_task_events(
        self,
        task_id: str,
        *,
        context: RequestContext,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        self._require_context(context)
        start_time = _parse_datetime(after) if after else datetime.now(timezone.utc)
        last_seen = start_time
        seen_ids: set[str] = set()

        if self._bus is None or not hasattr(self._bus, "subscribe"):
            while True:
                await asyncio.sleep(self._poll_interval)
                try:
                    record = self._service.get_task(
                        task_id,
                        tenant_id=context.tenant_id,
                        with_events=True,
                    )
                except TaskNotFoundError as exc:
                    raise TaskNotFoundRouteError(str(exc)) from exc
                events = record.get("events") or []
                for event in events:
                    created_at = event.get("created_at")
                    if created_at is None:
                        continue
                    moment = _parse_datetime(created_at)
                    identifier = str(event.get("id"))
                    if moment <= last_seen and identifier in seen_ids:
                        continue
                    if moment < start_time:
                        continue
                    seen_ids.add(identifier)
                    last_seen = moment
                    yield self._enrich_event(event, context, record)
        else:
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
            topic = self._topic_name(task_id)

            async def _handler(message) -> None:  # pragma: no cover - requires async backend
                payload = dict(message.payload)
                if payload.get("tenant_id") != context.tenant_id:
                    return
                await queue.put(payload)

            subscription = self._bus.subscribe(topic, _handler)
            try:
                while True:
                    payload = await queue.get()
                    if payload.get("created_at"):
                        last_seen = _parse_datetime(payload["created_at"])
                    yield payload
            finally:  # pragma: no cover - cleanup
                subscription.cancel()

    # -- helpers ---------------------------------------------------------

    def _require_context(self, context: Optional[RequestContext]) -> None:
        if context is None or not context.tenant_id:
            raise TaskAuthorizationError("A tenant scoped context is required")

    def _emit_events(
        self,
        events: Iterable[Mapping[str, Any]],
        context: RequestContext,
        task: Optional[Mapping[str, Any]] = None,
    ) -> None:
        for event in events:
            payload = self._enrich_event(event, context, task)
            self._publish_event(payload)

    def _enrich_event(
        self,
        event: Mapping[str, Any],
        context: RequestContext,
        task: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(event)
        payload.setdefault("tenant_id", context.tenant_id)
        payload.setdefault("task_id", (task or {}).get("id") or event.get("task_id"))
        if task is not None:
            payload.setdefault("task", dict(task))
        return payload

    def _publish_event(self, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        topic = self._topic_name(payload.get("task_id"))
        try:
            self._bus.publish_from_sync(topic, dict(payload))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to publish task event for %s", payload.get("task_id"))

    def _topic_name(self, task_id: Any) -> str:
        identifier = "unknown" if task_id is None else str(task_id)
        return f"task.events.{identifier}"


__all__ = [
    "TaskRoutes",
    "TaskRouteError",
    "TaskValidationError",
    "TaskAuthorizationError",
    "TaskNotFoundRouteError",
    "TaskConflictError",
]
