"""Conversation focused REST helpers used by :class:`AtlasServer`."""

from __future__ import annotations

import asyncio
import base64
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

from jsonschema import Draft202012Validator

from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger
from modules.orchestration.message_bus import MessageBus

if TYPE_CHECKING:
    from modules.task_store.service import TaskService

logger = setup_logger(__name__)


class ConversationRouteError(RuntimeError):
    """Base class for conversation route errors with HTTP style codes."""

    status_code: int = 400

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class ConversationValidationError(ConversationRouteError):
    status_code = 422


class ConversationAuthorizationError(ConversationRouteError):
    status_code = 403


class ConversationNotFoundError(ConversationRouteError):
    status_code = 404


@dataclass(frozen=True)
class RequestContext:
    """Request level context used for multi-tenant enforcement."""

    tenant_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    roles: Sequence[str] = ()
    metadata: Optional[Mapping[str, Any]] = None
    roles_authenticated: bool = False

    @classmethod
    def from_authenticated_claims(
        cls,
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        roles: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RequestContext":
        """Build a context whose roles originate from verified identity claims."""

        normalized_roles: list[str] = []
        for role in roles or ():
            token = str(role).strip()
            if token:
                normalized_roles.append(token)

        return cls(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            roles=tuple(normalized_roles),
            metadata=metadata,
            roles_authenticated=True,
        )


class _JsonSchemaValidator:
    """Thin wrapper around :mod:`jsonschema` validators."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self._validator = Draft202012Validator(schema)

    def validate(self, payload: Mapping[str, Any]) -> None:
        errors = list(self._validator.iter_errors(payload))
        if not errors:
            return
        fragments = []
        for error in errors:
            path = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
            formatted_path = path.lstrip(".") or "$"
            fragments.append(f"{formatted_path}: {error.message}")
        message = "; ".join(sorted(fragments))
        raise ConversationValidationError(message)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _encode_cursor(created_at: str, message_id: str) -> str:
    token = f"{created_at}|{message_id}"
    raw = token.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return encoded.rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - validation error reporting
        raise ConversationValidationError("Invalid pagination cursor supplied") from exc
    try:
        timestamp_text, message_id_text = decoded.split("|", 1)
    except ValueError as exc:  # pragma: no cover - defensive, invalid cursor
        raise ConversationValidationError("Malformed pagination cursor") from exc
    return _parse_datetime(timestamp_text), uuid.UUID(message_id_text)


def _metadata_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in candidate:
            return False
        candidate_value = candidate[key]
        if isinstance(value, Mapping) and isinstance(candidate_value, Mapping):
            if not _metadata_matches(candidate_value, value):
                return False
            continue
        if candidate_value != value:
            return False
    return True


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        fragments = []
        text = content.get("text")
        if isinstance(text, str):
            fragments.append(text)
        summary = content.get("summary")
        if isinstance(summary, str):
            fragments.append(summary)
        return "\n".join(fragments)
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        return "\n".join(_extract_text(item) for item in content)
    return str(content)


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


class ConversationRoutes:
    """Expose CRUD and search operations for the conversation store."""

    def __init__(
        self,
        repository: ConversationStoreRepository,
        *,
        message_bus: Optional[MessageBus] = None,
        task_service: Optional["TaskService"] = None,
        page_size_limit: int = 100,
        poll_interval: float = 0.5,
    ) -> None:
        self._repository = repository
        self._bus = message_bus
        self._task_service = task_service
        self._page_size_limit = max(int(page_size_limit), 1)
        self._poll_interval = max(float(poll_interval), 0.05)
        self._create_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["conversation_id", "role", "content"],
                "properties": {
                    "conversation_id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "content": {},
                    "metadata": {"type": "object"},
                    "extra": {"type": "object"},
                    "message_id": {"type": "string", "minLength": 1},
                    "timestamp": {"type": "string", "minLength": 1},
                    "assets": {"type": "array", "items": {"type": "object"}},
                    "vectors": {"type": "array", "items": {"type": "object"}},
                    "events": {"type": "array", "items": {"type": "object"}},
                    "conversation_metadata": {"type": "object"},
                    "message_type": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            }
        )
        self._update_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["conversation_id"],
                "properties": {
                    "conversation_id": {"type": "string"},
                    "content": {},
                    "metadata": {"type": "object"},
                    "extra": {"type": "object"},
                    "events": {"type": "array", "items": {"type": "object"}},
                    "message_type": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
                "anyOf": [
                    {"required": ["content"]},
                    {"required": ["metadata"]},
                    {"required": ["extra"]},
                    {"required": ["events"]},
                    {"required": ["message_type"]},
                    {"required": ["status"]},
                ],
            }
        )
        self._delete_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "required": ["conversation_id"],
                "properties": {
                    "conversation_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "metadata": {"type": "object"},
                    "message_type": {"type": "string", "minLength": 1},
                    "status": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            }
        )
        self._list_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "page_size": {"type": "integer", "minimum": 1, "maximum": page_size_limit},
                    "cursor": {"type": "string"},
                    "direction": {"type": "string", "enum": ["forward", "backward"]},
                    "metadata": {"type": "object"},
                    "include_deleted": {"type": "boolean"},
                    "message_types": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                    "statuses": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                },
                "additionalProperties": False,
            }
        )
        self._search_validator = _JsonSchemaValidator(
            {
                "type": "object",
                "properties": {
                    "conversation_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "text": {"type": "string"},
                    "metadata": {"type": "object"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": page_size_limit},
                    "offset": {"type": "integer", "minimum": 0},
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "top_k": {"type": "integer", "minimum": 1},
                    "vector": {
                        "type": "object",
                        "required": ["values"],
                        "properties": {
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 1,
                            },
                            "namespace": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            }
        )

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._create_validator.validate(payload)
        conversation_id = payload["conversation_id"]
        conversation_metadata = dict(payload.get("conversation_metadata") or {})
        user_uuid, session_uuid = self._resolve_identity(context)
        self._ensure_conversation_access(
            conversation_id,
            context,
            conversation_metadata,
            session_uuid=session_uuid,
        )

        metadata = dict(payload.get("metadata") or {})
        if context.user_id and "user" not in metadata:
            metadata["user"] = context.user_id
        extra = dict(payload.get("extra") or {})
        start_time = datetime.now(timezone.utc)
        stored = self._repository.add_message(
            conversation_id,
            tenant_id=context.tenant_id,
            role=str(payload["role"]),
            content=payload.get("content"),
            metadata=metadata,
            user_id=user_uuid,
            session_id=session_uuid,
            timestamp=payload.get("timestamp"),
            message_id=payload.get("message_id"),
            extra=extra,
            assets=list(payload.get("assets") or []),
            vectors=list(payload.get("vectors") or []),
            events=list(payload.get("events") or []),
            message_type=payload.get("message_type"),
            status=payload.get("status"),
        )
        events = self._repository.fetch_message_events(
            tenant_id=context.tenant_id,
            message_id=stored["id"],
            after=start_time,
        )
        self._emit_events(events, context, stored)
        return stored

    def update_message(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._update_validator.validate(payload)
        conversation_id = payload["conversation_id"]
        _, session_uuid = self._resolve_identity(context)
        self._ensure_conversation_access(
            conversation_id,
            context,
            session_uuid=session_uuid,
        )

        start_time = datetime.now(timezone.utc)
        stored = self._repository.record_edit(
            conversation_id,
            message_id,
            tenant_id=context.tenant_id,
            content=payload.get("content"),
            metadata=payload.get("metadata"),
            extra=payload.get("extra"),
            events=list(payload.get("events") or []),
            message_type=payload.get("message_type"),
            status=payload.get("status"),
        )
        events = self._repository.fetch_message_events(
            tenant_id=context.tenant_id,
            message_id=message_id,
            after=start_time,
        )
        self._emit_events(events, context, stored)
        return stored

    def delete_message(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._delete_validator.validate(payload)
        conversation_id = payload["conversation_id"]
        _, session_uuid = self._resolve_identity(context)
        self._ensure_conversation_access(
            conversation_id,
            context,
            session_uuid=session_uuid,
        )

        start_time = datetime.now(timezone.utc)
        self._repository.soft_delete_message(
            conversation_id,
            message_id,
            tenant_id=context.tenant_id,
            reason=payload.get("reason"),
            metadata=payload.get("metadata"),
            message_type=payload.get("message_type"),
            status=payload.get("status"),
        )
        stored = self._repository.get_message(
            conversation_id, message_id, tenant_id=context.tenant_id
        )
        events = self._repository.fetch_message_events(
            tenant_id=context.tenant_id,
            message_id=message_id,
            after=start_time,
        )
        self._emit_events(events, context, stored)
        return {"status": "deleted", "message": stored}

    def list_conversations(
        self,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        """Return a paginated listing of conversations for the tenant."""

        self._require_context(context)
        params = params or {}

        limit_value: Optional[int] = None
        if "limit" in params and params.get("limit") is not None:
            try:
                limit_value = int(params["limit"])
            except (TypeError, ValueError) as exc:
                raise ConversationValidationError(
                    "Conversation listing limit must be an integer"
                ) from exc
            if limit_value < 0:
                raise ConversationValidationError(
                    "Conversation listing limit must be non-negative"
                )

        offset_value = 0
        if params.get("offset") is not None:
            try:
                offset_value = int(params["offset"])
            except (TypeError, ValueError) as exc:
                raise ConversationValidationError(
                    "Conversation listing offset must be an integer"
                ) from exc
            if offset_value < 0:
                raise ConversationValidationError(
                    "Conversation listing offset must be non-negative"
                )

        order_value = str(params.get("order") or "desc").lower()
        if order_value not in {"asc", "desc"}:
            raise ConversationValidationError(
                "Conversation listing order must be 'asc' or 'desc'"
            )

        records = self._repository.list_conversations_for_tenant(
            context.tenant_id,
            limit=limit_value,
            offset=offset_value,
            order=order_value,
        )

        return {
            "items": records,
            "count": len(records),
            "limit": limit_value,
            "offset": offset_value,
            "order": order_value,
        }

    def reset_conversation(
        self,
        conversation_id: str,
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        """Remove all messages from a conversation while retaining metadata."""

        self._require_context(context)
        conversation = self._repository.get_conversation(
            conversation_id, tenant_id=context.tenant_id
        )
        if conversation is None:
            raise ConversationNotFoundError(
                "Conversation is not accessible for this tenant"
            )

        self._repository.reset_conversation(
            conversation_id, tenant_id=context.tenant_id
        )
        return {"status": "reset", "conversation": conversation}

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        """Permanently remove a conversation for the current tenant."""

        self._require_context(context)
        conversation = self._repository.get_conversation(
            conversation_id, tenant_id=context.tenant_id
        )
        if conversation is None:
            raise ConversationNotFoundError(
                "Conversation is not accessible for this tenant"
            )

        self._repository.hard_delete_conversation(
            conversation_id, tenant_id=context.tenant_id
        )
        return {"status": "deleted", "conversation": conversation}

    def list_messages(
        self,
        conversation_id: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        params = params or {}
        self._require_context(context)
        self._list_validator.validate(params)
        _, session_uuid = self._resolve_identity(context)
        self._ensure_conversation_access(
            conversation_id,
            context,
            session_uuid=session_uuid,
        )

        limit = int(params.get("page_size") or 20)
        limit = max(1, min(limit, self._page_size_limit))
        direction = str(params.get("direction") or "forward").lower()
        include_deleted = bool(params.get("include_deleted", False))
        metadata_filter = dict(params.get("metadata") or {})
        message_types = list(params.get("message_types") or [])
        statuses = list(params.get("statuses") or [])
        cursor_value = params.get("cursor")
        cursor = _decode_cursor(cursor_value) if cursor_value else None

        records = self._repository.fetch_messages(
            conversation_id,
            tenant_id=context.tenant_id,
            limit=limit + 1,
            cursor=cursor,
            direction=direction,
            metadata_filter=metadata_filter or None,
            include_deleted=include_deleted,
            message_types=message_types or None,
            statuses=statuses or None,
        )

        next_cursor = None
        previous_cursor = None
        if len(records) > limit:
            overflow = records.pop()
            if direction == "forward":
                next_cursor = _encode_cursor(overflow["created_at"], overflow["id"])
            else:
                previous_cursor = _encode_cursor(overflow["created_at"], overflow["id"])

        return {
            "items": records,
            "page": {
                "size": limit,
                "direction": direction,
                "next_cursor": next_cursor,
                "previous_cursor": previous_cursor,
            },
        }

    def search_conversations(
        self,
        payload: Mapping[str, Any],
        *,
        context: RequestContext,
    ) -> Dict[str, Any]:
        self._require_context(context)
        self._search_validator.validate(payload)
        _, session_uuid = self._resolve_identity(context)
        metadata_filter = dict(payload.get("metadata") or {})
        limit = int(payload.get("limit") or 20)
        limit = max(1, min(limit, self._page_size_limit))

        raw_ids = list(payload.get("conversation_ids") or [])
        if raw_ids:
            conversation_ids = []
            for identifier in raw_ids:
                conversation_ids.append(
                    self._ensure_conversation_access(
                        identifier,
                        context,
                        session_uuid=session_uuid,
                    )["id"]
                )
        else:
            conversation_ids = [record["id"] for record in self._repository.list_conversations_for_tenant(context.tenant_id)]

        if not conversation_ids:
            return {"count": 0, "items": []}

        results: Dict[str, Dict[str, Any]] = {}
        text_query = str(payload.get("text") or "").strip()
        vector_block = payload.get("vector")
        vector_values: Sequence[float] = ()
        if vector_block:
            vector_values = [float(value) for value in vector_block.get("values") or []]

        order = str(payload.get("order") or "desc").lower()
        if order not in {"asc", "desc"}:
            order = "desc"
        offset_value = max(int(payload.get("offset") or 0), 0)
        window = offset_value + limit
        query_limit = max(window, 1)
        raw_top_k = payload.get("top_k")
        top_k_value: Optional[int] = None
        if raw_top_k is not None:
            try:
                top_k_value = max(int(raw_top_k), 1)
            except (TypeError, ValueError):
                top_k_value = None
        if top_k_value is None:
            top_k_value = query_limit

        if text_query or not vector_values:
            for message in self._repository.query_messages_by_text(
                conversation_ids=conversation_ids,
                tenant_id=context.tenant_id,
                text=text_query,
                metadata_filter=metadata_filter or None,
                include_deleted=False,
                order=order,
                limit=query_limit,
            ):
                identifier = message["id"]
                record = results.setdefault(
                    identifier,
                    {
                        "conversation_id": message["conversation_id"],
                        "message": message,
                        "score": 0.0,
                    },
                )
                record["message"] = message
                if text_query:
                    record["score"] = max(record["score"], 1.0)

        if vector_values:
            for message, vector in self._repository.query_message_vectors(
                conversation_ids=conversation_ids,
                tenant_id=context.tenant_id,
                metadata_filter=metadata_filter or None,
                include_deleted=False,
                order=order,
                offset=offset_value,
                limit=query_limit,
                top_k=top_k_value,
            ):
                embedding = vector.get("embedding") or []
                if not embedding:
                    continue
                similarity = _cosine_similarity(
                    vector_values, [float(component) for component in embedding]
                )
                if similarity <= 0:
                    continue
                identifier = message["id"]
                record = results.setdefault(
                    identifier,
                    {
                        "conversation_id": message["conversation_id"],
                        "message": message,
                        "score": 0.0,
                    },
                )
                record["message"] = message
                record["score"] = max(record["score"], float(similarity))

        def _sort_key(entry: Mapping[str, Any]) -> tuple[float, float]:
            raw_created = entry["message"].get("created_at") or entry["message"].get("timestamp")
            created_ts = 0.0
            if raw_created:
                created_ts = _parse_datetime(raw_created).timestamp()
            timestamp_key = created_ts if order == "asc" else -created_ts
            return (-float(entry["score"]), timestamp_key)

        filtered_items = [
            item
            for item in results.values()
            if item["score"] > 0 or (not text_query and not vector_values)
        ]
        ordered = sorted(filtered_items, key=_sort_key)
        windowed = ordered[offset_value : offset_value + limit]
        return {"count": len(windowed), "items": windowed}

    async def stream_message_events(
        self,
        conversation_id: str,
        *,
        context: RequestContext,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        self._require_context(context)
        _, session_uuid = self._resolve_identity(context)
        self._ensure_conversation_access(
            conversation_id,
            context,
            session_uuid=session_uuid,
        )
        last_seen = after
        backlog = self._repository.fetch_message_events(
            tenant_id=context.tenant_id,
            conversation_id=conversation_id,
            after=after,
        )
        for event in backlog:
            enriched = self._enrich_event(event, context)
            last_seen = enriched["created_at"]
            yield enriched

        if self._bus is None:
            while True:
                await asyncio.sleep(self._poll_interval)
                fresh = self._repository.fetch_message_events(
                    tenant_id=context.tenant_id,
                    conversation_id=conversation_id,
                    after=last_seen,
                )
                for event in fresh:
                    enriched = self._enrich_event(event, context)
                    last_seen = enriched["created_at"]
                    yield enriched
        else:
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
            topic = self._topic_name(conversation_id)

            async def _handler(message) -> None:
                payload = dict(message.payload)
                if payload.get("tenant_id") != context.tenant_id:
                    return
                await queue.put(payload)

            subscription = self._bus.subscribe(topic, _handler)
            try:
                while True:
                    payload = await queue.get()
                    last_seen = payload.get("created_at") or last_seen
                    yield payload
            finally:
                subscription.cancel()

    # -- internal helpers -------------------------------------------------

    def _require_context(self, context: Optional[RequestContext]) -> None:
        if context is None or not context.tenant_id:
            raise ConversationAuthorizationError("A tenant scoped context is required")

    def _resolve_identity(
        self, context: RequestContext
    ) -> tuple[Optional[uuid.UUID], Optional[uuid.UUID]]:
        """Ensure the context user/session exist in the conversation store."""

        if self._repository is None:
            return None, None

        user_uuid: Optional[uuid.UUID] = None
        session_uuid: Optional[uuid.UUID] = None

        user_metadata: Dict[str, Any] = {"tenant_id": context.tenant_id}
        session_metadata: Dict[str, Any] = {"tenant_id": context.tenant_id}
        user_display_name: Optional[str] = None

        extra_metadata = context.metadata if isinstance(context.metadata, Mapping) else None
        if isinstance(extra_metadata, Mapping):
            user_section = extra_metadata.get("user")
            if isinstance(user_section, Mapping):
                user_metadata.update(dict(user_section))
            session_section = extra_metadata.get("session")
            if isinstance(session_section, Mapping):
                session_metadata.update(dict(session_section))
            display_candidate = extra_metadata.get("user_display_name")
            if isinstance(display_candidate, str):
                stripped = display_candidate.strip()
                user_display_name = stripped or None

        if context.roles:
            user_metadata.setdefault("roles", list(context.roles))

        if context.user_id:
            user_uuid = self._repository.ensure_user(
                context.user_id,
                display_name=user_display_name,
                metadata=user_metadata,
            )

        if context.session_id:
            session_uuid = self._repository.ensure_session(
                user_uuid,
                context.session_id,
                metadata=session_metadata,
            )

        return user_uuid, session_uuid

    def _ensure_conversation_access(
        self,
        conversation_id: str,
        context: RequestContext,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        session_uuid: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        tenant_id = context.tenant_id
        conversation = self._repository.get_conversation(
            conversation_id, tenant_id=tenant_id
        )
        if conversation is None:
            try:
                self._repository.ensure_conversation(
                    conversation_id,
                    tenant_id=tenant_id,
                    session_id=session_uuid,
                    metadata=dict(metadata or {}),
                )
            except ValueError as exc:
                raise ConversationAuthorizationError(
                    "Conversation is not accessible for this tenant"
                ) from exc
            conversation = self._repository.get_conversation(
                conversation_id, tenant_id=tenant_id
            )
        if conversation is None:
            raise ConversationNotFoundError("Conversation could not be created")
        if metadata:
            merged_metadata = dict(conversation.get("metadata") or {})
            before = merged_metadata.copy()
            merged_metadata.update(metadata)
            if merged_metadata != before:
                self._repository.ensure_conversation(
                    conversation_id,
                    tenant_id=tenant_id,
                    session_id=session_uuid,
                    metadata=merged_metadata,
                )
                conversation = self._repository.get_conversation(
                    conversation_id, tenant_id=tenant_id
                )
        return conversation

    def _emit_events(
        self,
        events: Iterable[Mapping[str, Any]],
        context: RequestContext,
        message: Optional[Mapping[str, Any]] = None,
    ) -> None:
        for event in events:
            enriched = self._enrich_event(event, context, message)
            self._publish_event(enriched)

    def _enrich_event(
        self,
        event: Mapping[str, Any],
        context: RequestContext,
        message: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(event)
        payload.setdefault("tenant_id", context.tenant_id)
        if message is not None:
            payload.setdefault("message", dict(message))
        return payload

    def _publish_event(self, payload: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish_from_sync(self._topic_name(payload.get("conversation_id")), dict(payload))
        except Exception:  # pragma: no cover - best effort logging only
            logger.exception("Failed to publish conversation event for %s", payload.get("message_id"))

    def _topic_name(self, conversation_id: Any) -> str:
        identifier = "unknown" if conversation_id is None else str(conversation_id)
        return f"conversation.events.{identifier}"


__all__ = [
    "ConversationRoutes",
    "ConversationRouteError",
    "ConversationValidationError",
    "ConversationAuthorizationError",
    "ConversationNotFoundError",
    "RequestContext",
]
