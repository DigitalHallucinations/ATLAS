"""Repository helpers for working with the conversation store."""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import and_, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    Conversation,
    Message,
    MessageAsset,
    MessageEvent,
    MessageVector,
)


def _coerce_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if value is None:
        raise ValueError("UUID value cannot be None")
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    text = str(value)
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.UUID(hex=text.replace("-", ""))


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise TypeError(f"Unsupported datetime value: {value!r}")


def create_conversation_engine(url: str, **engine_kwargs: Any) -> Engine:
    """Create a SQLAlchemy engine for the conversation store."""

    return create_engine(url, future=True, **engine_kwargs)


class ConversationStoreRepository:
    """Persistence helper that wraps CRUD operations for conversation data."""

    def __init__(
        self,
        session_factory: sessionmaker,
        *,
        retention: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session_factory = session_factory
        self._retention = retention or {}

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

    # -- bootstrap helpers -------------------------------------------------

    def create_schema(self) -> None:
        """Create conversation store tables if they do not already exist."""

        engine: Engine | None = getattr(self._session_factory, "bind", None)
        if engine is None:
            raise RuntimeError("Session factory is not bound to an engine")
        Base.metadata.create_all(engine)

    # -- CRUD operations ----------------------------------------------------

    def ensure_conversation(
        self,
        conversation_id: Any,
        *,
        session_id: Any | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """Ensure that a conversation row exists and return its UUID."""

        conversation_uuid = _coerce_uuid(conversation_id)
        session_uuid = _coerce_uuid(session_id) if session_id is not None else None
        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                conversation = Conversation(
                    id=conversation_uuid,
                    session_id=session_uuid,
                    metadata=metadata or {},
                )
                session.add(conversation)
            else:
                if metadata:
                    merged = dict(conversation.metadata or {})
                    merged.update(metadata)
                    conversation.metadata = merged
                if session_uuid and conversation.session_id != session_uuid:
                    conversation.session_id = session_uuid
        return conversation_uuid

    def load_recent_messages(
        self,
        conversation_id: Any,
        *,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return serialized messages ordered by creation time ascending."""

        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            stmt = select(Message).where(Message.conversation_id == conversation_uuid)
            if before is not None:
                stmt = stmt.where(Message.created_at < before)
            stmt = stmt.order_by(Message.created_at.desc())
            if limit:
                stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()
        return [self._serialize_message(row) for row in reversed(rows)]

    def add_message(
        self,
        conversation_id: Any,
        *,
        role: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Any | None = None,
        timestamp: Any | None = None,
        message_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
        vectors: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Insert a message and related resources, returning the stored payload."""

        conversation_uuid = _coerce_uuid(conversation_id)
        user_uuid = _coerce_uuid(user_id) if user_id else None
        extra_payload = dict(extra or {})
        message_metadata = dict(metadata or {})

        with self._session_scope() as session:
            if message_id is not None:
                existing = session.execute(
                    select(Message).where(
                        and_(
                            Message.conversation_id == conversation_uuid,
                            Message.client_message_id == message_id,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    return self._serialize_message(existing)

            message = Message(
                conversation_id=conversation_uuid,
                user_id=user_uuid,
                role=role,
                content=content,
                metadata=message_metadata,
                extra=extra_payload,
                client_message_id=message_id,
            )

            if timestamp is not None:
                message.created_at = _coerce_dt(timestamp)

            session.add(message)
            session.flush()

            self._store_assets(session, message, assets)
            self._store_vectors(session, message, vectors)
            self._store_events(session, message, events)
            self._store_events(
                session,
                message,
                [
                    {
                        "event_type": "created",
                        "metadata": {"role": role},
                    }
                ],
            )

            session.flush()
            return self._serialize_message(message)

    def record_edit(
        self,
        conversation_id: Any,
        message_id: Any,
        *,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Update an existing message and record corresponding events."""

        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if message is None or message.conversation_id != conversation_uuid:
                raise ValueError("Unknown message or conversation")

            if content is not None:
                message.content = content
            if metadata is not None:
                existing = dict(message.metadata or {})
                existing.update(metadata)
                message.metadata = existing
            if extra is not None:
                existing_extra = dict(message.extra or {})
                existing_extra.update(extra)
                message.extra = existing_extra

            message.updated_at = datetime.now(timezone.utc)
            self._store_events(session, message, events)
            self._store_events(
                session,
                message,
                [
                    {
                        "event_type": "edited",
                        "metadata": {},
                    }
                ],
            )
            session.flush()
            return self._serialize_message(message)

    def soft_delete_message(
        self,
        conversation_id: Any,
        message_id: Any,
        *,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if message is None or message.conversation_id != conversation_uuid:
                raise ValueError("Unknown message or conversation")

            message.deleted_at = datetime.now(timezone.utc)
            audit_metadata = {"reason": reason} if reason else {}
            if metadata:
                audit_metadata.update(metadata)
            self._store_events(
                session,
                message,
                [
                    {
                        "event_type": "soft_deleted",
                        "metadata": audit_metadata,
                    }
                ],
            )

    def hard_delete_conversation(self, conversation_id: Any) -> None:
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                return
            session.delete(conversation)

    def reset_conversation(self, conversation_id: Any) -> None:
        """Remove all messages but leave the conversation row intact."""

        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            session.execute(
                delete(Message).where(Message.conversation_id == conversation_uuid)
            )

    # -- internal helpers ---------------------------------------------------

    def _store_assets(
        self,
        session: Session,
        message: Message,
        assets: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not assets:
            maybe_audio = message.extra.get("audio") if message.extra else None
            if maybe_audio is not None:
                assets = [
                    {
                        "asset_type": "audio",
                        "metadata": {"format": message.extra.get("audio_format")},
                        "uri": message.extra.get("audio_id"),
                    }
                ]
            else:
                return
        for asset in assets:
            asset_type = str(asset.get("asset_type") or asset.get("type") or "attachment")
            record = MessageAsset(
                conversation_id=message.conversation_id,
                message_id=message.id,
                asset_type=asset_type,
                uri=asset.get("uri"),
                metadata=dict(asset.get("metadata") or {}),
            )
            session.add(record)

    def _store_vectors(
        self,
        session: Session,
        message: Message,
        vectors: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not vectors:
            return
        for vector in vectors:
            record = MessageVector(
                conversation_id=message.conversation_id,
                message_id=message.id,
                provider=vector.get("provider"),
                embedding=vector.get("embedding"),
                metadata=dict(vector.get("metadata") or {}),
            )
            session.add(record)

    def _store_events(
        self,
        session: Session,
        message: Message,
        events: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not events:
            return
        for event in events:
            record = MessageEvent(
                conversation_id=message.conversation_id,
                message_id=message.id,
                event_type=str(event.get("event_type") or event.get("type") or "event"),
                metadata=dict(event.get("metadata") or {}),
            )
            session.add(record)

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(message.id),
            "conversation_id": str(message.conversation_id),
            "role": message.role,
            "content": message.content,
            "metadata": dict(message.metadata or {}),
            "timestamp": message.created_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        if message.extra:
            payload.update(message.extra)
        if message.client_message_id:
            payload["message_id"] = message.client_message_id
        if message.deleted_at is not None:
            payload["deleted_at"] = message.deleted_at.astimezone(timezone.utc).isoformat()
        return payload
