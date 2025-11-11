"""Utilities shared by store repositories."""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from modules.conversation_store.models import Conversation

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy import and_, or_, select
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - fallback when SQLAlchemy is absent
    def select(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy select is unavailable in this environment")

    def and_(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy and_ is unavailable in this environment")

    def or_(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy or_ is unavailable in this environment")

    class IntegrityError(Exception):
        """Fallback IntegrityError when SQLAlchemy is not installed."""

        pass

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


@contextlib.contextmanager
def _session_scope(session_factory: sessionmaker) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""

    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class BaseStoreRepository:
    """Base helper shared by store repositories built on SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    @contextlib.contextmanager
    def _session_scope(self) -> Iterator[Session]:
        with _session_scope(self._session_factory) as session:
            yield session

    # -- schema helpers -------------------------------------------------

    def _get_schema_metadata(self) -> Any:  # pragma: no cover - abstract hook
        raise NotImplementedError("Repositories must provide schema metadata")

    def _ensure_additional_schema(self, engine: Any) -> None:  # pragma: no cover - extension hook
        """Allow subclasses to run additional schema setup after metadata creation."""

        pass

    def _resolve_engine(self) -> Any:
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
        return engine

    def create_schema(self) -> None:
        metadata = self._get_schema_metadata()
        if metadata is None or not hasattr(metadata, "create_all"):
            raise RuntimeError("Repository metadata must define a create_all method")
        engine = self._resolve_engine()
        metadata.create_all(engine)
        self._ensure_additional_schema(engine)

    def _create_with_created_event(
        self,
        *,
        model_cls: Type[ModelT],
        model_kwargs: Mapping[str, Any],
        serializer: Callable[[ModelT], Dict[str, Any]],
        event_model_cls: Type[EventModelT],
        event_enum: Type[EnumT],
        event_serializer: Callable[[EventModelT], Dict[str, Any]],
        event_foreign_key: str,
        tenant_id: str,
        conversation_id: Optional[uuid.UUID],
        require_conversation: bool = False,
        event_payload_factory: Optional[Callable[[ModelT], Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        with self._session_scope() as session:
            if conversation_id is None:
                if require_conversation:
                    raise ValueError("Conversation identifier is required")
            else:
                conversation = session.get(Conversation, conversation_id)
                if conversation is None:
                    raise ValueError("Conversation does not exist")
                if conversation.tenant_id != tenant_id:
                    raise ValueError("Conversation belongs to a different tenant")

            record = model_cls(**dict(model_kwargs))
            session.add(record)
            session.flush()
            session.refresh(record)
            payload = serializer(record)

            if event_payload_factory is None:
                event_payload: Dict[str, Any] = {}
            else:
                event_payload = dict(event_payload_factory(record))

            event = event_model_cls(
                **{
                    event_foreign_key: getattr(record, "id"),
                    "event_type": event_enum.CREATED,
                    "payload": event_payload,
                }
            )
            session.add(event)
            session.flush()

            payload["events"] = [event_serializer(event)]
            return payload

    def _update_with_optimistic_lock(
        self,
        *,
        load_record: Callable[[Session], ModelT],
        expected_updated_at: Any | None,
        concurrency_error_factory: Callable[[], Exception],
        changes: Mapping[str, Any],
        field_mutators: Mapping[
            str, Callable[[ModelT, Any, Session, Dict[str, Any]], None]
        ],
        serializer: Callable[[ModelT], Dict[str, Any]],
        event_factories: Sequence[
            Callable[[ModelT, Mapping[str, Any], Dict[str, Any]], Sequence[EventModelT]]
        ],
        event_serializer: Callable[[EventModelT], Dict[str, Any]],
        unknown_field_error_factory: Callable[[str], Exception],
        context_factory: Optional[Callable[[ModelT], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        with self._session_scope() as session:
            record = load_record(session)

            if expected_updated_at is not None:
                expected = _coerce_dt(expected_updated_at)
                current = getattr(record, "updated_at", None)
                if current is None:
                    raise concurrency_error_factory()
                if current.tzinfo is None:
                    current = current.replace(tzinfo=timezone.utc)
                else:
                    current = current.astimezone(timezone.utc)
                if current != expected:
                    raise concurrency_error_factory()

            context: Dict[str, Any]
            if context_factory is None:
                context = {}
            else:
                context = dict(context_factory(record))

            for field, value in changes.items():
                mutator = field_mutators.get(field)
                if mutator is None:
                    raise unknown_field_error_factory(field)
                mutator(record, value, session, context)

            session.flush()
            payload = serializer(record)

            events: list[EventModelT] = []
            for factory in event_factories:
                produced = factory(record, changes, context)
                if not produced:
                    continue
                events.extend(list(produced))

            for event in events:
                session.add(event)

            if events:
                session.flush()
                payload["events"] = [event_serializer(event) for event in events]
            else:
                payload["events"] = []

            return payload


EnumT = TypeVar("EnumT", bound=Enum)
ModelT = TypeVar("ModelT")
EventModelT = TypeVar("EventModelT")


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


def _coerce_optional_dt(value: Any | None) -> Optional[datetime]:
    if value in (None, ""):
        return None
    return _coerce_dt(value)


def _dt_to_iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    normalized = moment.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _normalize_meta(
    metadata: Mapping[str, Any] | None,
    *,
    error_message: str = "Metadata must be a mapping",
) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(error_message)
    return dict(metadata)


def _normalize_enum_value(
    value: Any | None,
    enum_cls: Type[EnumT],
    default_member: EnumT,
) -> EnumT:
    """Normalize arbitrary input into an enum member.

    Parameters
    ----------
    value:
        Raw value that may be ``None``, an enum member, a string, or other primitive.
    enum_cls:
        Enumeration class to coerce into.
    default_member:
        Fallback enum member used when ``value`` is ``None`` or empty.

    Returns
    -------
    EnumT
        A member of ``enum_cls`` corresponding to the provided ``value``.
    """

    if value is None:
        return default_member
    if isinstance(value, enum_cls):
        return value

    text = str(value).strip()
    if not text:
        return default_member

    candidates: list[str] = []
    # For string-based enums we prefer a case-insensitive lookup.
    if issubclass(enum_cls, str):
        lowered = text.lower()
        if lowered not in candidates:
            candidates.append(lowered)
    if text not in candidates:
        candidates.append(text)
    uppered = text.upper()
    if uppered not in candidates:
        candidates.append(uppered)

    for candidate in candidates:
        try:
            return enum_cls(candidate)
        except ValueError:
            continue

    normalized_name = text.lower()
    for member in enum_cls:
        if member.name.lower() == normalized_name:
            return member

    # Allow the enum class to raise the canonical ValueError for unknown values.
    return enum_cls(text)


__all__ = [
    "BaseStoreRepository",
    "IntegrityError",
    "Session",
    "and_",
    "joinedload",
    "or_",
    "select",
    "sessionmaker",
    "_coerce_dt",
    "_coerce_optional_dt",
    "_coerce_uuid",
    "_dt_to_iso",
    "_normalize_enum_value",
    "_normalize_meta",
    "_normalize_tenant_id",
    "_session_scope",
]
