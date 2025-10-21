"""Repository helpers for working with the conversation store."""

from __future__ import annotations

import contextlib
import hashlib
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from sqlalchemy import and_, create_engine, delete, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    Conversation,
    Message,
    MessageAsset,
    MessageEvent,
    MessageVector,
    Session as StoreSession,
    User,
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


def _hash_vector(values: Sequence[float]) -> str:
    hasher = hashlib.sha1()
    for component in values:
        hasher.update(struct.pack("!d", float(component)))
    return hasher.hexdigest()


def _default_vector_key(
    message_id: uuid.UUID,
    provider: Optional[str],
    model: Optional[str],
    version: Optional[str],
) -> str:
    provider_part = (provider or "conversation").strip() or "conversation"
    model_part = (model or "default").strip() or "default"
    version_part = (version or "v0").strip() or "v0"
    return f"{message_id}:{provider_part}:{model_part}:{version_part}"


@dataclass
class _VectorPayload:
    provider: Optional[str]
    vector_key: str
    embedding: List[float]
    metadata: Dict[str, Any]
    model: Optional[str]
    version: Optional[str]
    checksum: str
    dimensions: int


def _normalize_vector_payload(message: Message, vector: Mapping[str, Any]) -> _VectorPayload:
    if not isinstance(vector, Mapping):
        raise TypeError("Vector payloads must be mappings containing embedding data.")

    provider = vector.get("provider")
    if provider is not None:
        provider = str(provider).strip() or None

    model = vector.get("model") or vector.get("embedding_model")
    if model is not None:
        model = str(model).strip() or None

    version = vector.get("model_version") or vector.get("embedding_model_version")
    if version is not None:
        version = str(version).strip() or None

    raw_values = vector.get("embedding")
    if raw_values is None:
        raw_values = vector.get("values")
    if raw_values is None:
        raise ValueError("Vector payloads must include 'embedding' or 'values'.")
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (bytes, bytearray, str)):
        raise TypeError("Vector embeddings must be a sequence of numeric values.")

    embedding: List[float] = []
    for component in raw_values:
        embedding.append(float(component))
    if not embedding:
        raise ValueError("Vector embeddings must contain at least one component.")

    checksum = vector.get("checksum") or vector.get("embedding_checksum")
    if checksum is None or not str(checksum).strip():
        checksum = _hash_vector(embedding)
    else:
        checksum = str(checksum)

    raw_vector_key = vector.get("vector_key") or vector.get("id")
    vector_key = str(raw_vector_key).strip() if raw_vector_key is not None else ""
    if not vector_key:
        vector_key = _default_vector_key(message.id, provider, model, version)

    metadata = dict(vector.get("metadata") or {})
    metadata.setdefault("conversation_id", str(message.conversation_id))
    metadata.setdefault("message_id", str(message.id))
    metadata.setdefault("namespace", metadata.get("namespace") or str(message.conversation_id))
    if provider:
        metadata.setdefault("provider", provider)
    if model:
        metadata.setdefault("model", model)
        metadata.setdefault("embedding_model", model)
    if version:
        metadata.setdefault("model_version", version)
        metadata.setdefault("embedding_model_version", version)
    metadata.setdefault("vector_key", vector_key)
    metadata.setdefault("checksum", checksum)
    metadata.setdefault("embedding_checksum", checksum)
    metadata.setdefault("dimensions", len(embedding))

    return _VectorPayload(
        provider=provider,
        vector_key=vector_key,
        embedding=embedding,
        metadata=metadata,
        model=model,
        version=version,
        checksum=str(checksum),
        dimensions=len(embedding),
    )


def _serialize_vector(vector: MessageVector) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(vector.id),
        "vector_key": vector.vector_key,
        "conversation_id": str(vector.conversation_id),
        "message_id": str(vector.message_id),
        "provider": vector.provider,
        "embedding_model": vector.embedding_model,
        "embedding_model_version": vector.embedding_model_version,
        "embedding_checksum": vector.embedding_checksum,
        "dimensions": vector.dimensions,
        "metadata": dict(vector.meta or {}),
        "created_at": vector.created_at.astimezone(timezone.utc).isoformat(),
    }
    if vector.updated_at is not None:
        payload["updated_at"] = vector.updated_at.astimezone(timezone.utc).isoformat()
    embedding = vector.embedding
    if embedding is not None:
        payload["embedding"] = [float(component) for component in embedding]
    return payload


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

    def ensure_user(
        self,
        external_id: Any,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> uuid.UUID:
        """Ensure that a user record exists for ``external_id`` and return its UUID."""

        if external_id is None:
            raise ValueError("External user identifier must not be None")

        identifier = str(external_id).strip()
        if not identifier:
            raise ValueError("External user identifier must not be empty")

        display = display_name.strip() if isinstance(display_name, str) else None
        metadata_payload = dict(metadata or {})

        with self._session_scope() as session:
            record = session.execute(
                select(User).where(User.external_id == identifier)
            ).scalar_one_or_none()

            if record is None:
                record = User(
                    external_id=identifier,
                    display_name=display or identifier,
                    meta=metadata_payload,
                )
                session.add(record)
                session.flush()
            else:
                updated = False
                if display and record.display_name != display:
                    record.display_name = display
                    updated = True

                if metadata_payload:
                    merged = dict(record.meta or {})
                    before = merged.copy()
                    merged.update(metadata_payload)
                    if merged != before:
                        record.meta = merged
                        updated = True

                if updated:
                    session.flush()

            return record.id

    def ensure_session(
        self,
        user_id: Any | None,
        external_session_id: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> uuid.UUID:
        """Ensure that a session record exists and return its UUID."""

        if external_session_id is None:
            raise ValueError("External session identifier must not be None")

        identifier = str(external_session_id).strip()
        if not identifier:
            raise ValueError("External session identifier must not be empty")

        user_uuid = _coerce_uuid(user_id) if user_id is not None else None
        metadata_payload = dict(metadata or {})

        with self._session_scope() as session:
            record = session.execute(
                select(StoreSession).where(StoreSession.external_id == identifier)
            ).scalar_one_or_none()

            if record is None:
                record = StoreSession(
                    external_id=identifier,
                    user_id=user_uuid,
                    meta=metadata_payload,
                )
                session.add(record)
                session.flush()
            else:
                updated = False
                if user_uuid is not None and record.user_id != user_uuid:
                    record.user_id = user_uuid
                    updated = True

                if metadata_payload:
                    merged = dict(record.meta or {})
                    before = merged.copy()
                    merged.update(metadata_payload)
                    if merged != before:
                        record.meta = merged
                        updated = True

                if updated:
                    session.flush()

            return record.id

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
                    meta=metadata or {},
                )
                session.add(conversation)
            else:
                if metadata:
                    merged = dict(conversation.meta or {})
                    merged.update(metadata)
                    conversation.meta = merged
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
        user: Any | None = None,
        user_display_name: Optional[str] = None,
        user_metadata: Optional[Mapping[str, Any]] = None,
        session_id: Any | None = None,
        session: Any | None = None,
        session_metadata: Optional[Mapping[str, Any]] = None,
        timestamp: Any | None = None,
        message_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
        vectors: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Insert a message and related resources, returning the stored payload."""

        conversation_uuid = _coerce_uuid(conversation_id)

        user_uuid: uuid.UUID | None
        user_metadata_payload: Optional[Dict[str, Any]] = None
        if user_metadata:
            if isinstance(user_metadata, Mapping):
                user_metadata_payload = dict(user_metadata)
            else:
                user_metadata_payload = dict(user_metadata)

        session_metadata_payload: Optional[Dict[str, Any]] = None
        if session_metadata:
            if isinstance(session_metadata, Mapping):
                session_metadata_payload = dict(session_metadata)
            else:
                session_metadata_payload = dict(session_metadata)

        if user_id is not None:
            user_uuid = _coerce_uuid(user_id)
        elif user is not None:
            user_uuid = self.ensure_user(
                user,
                display_name=user_display_name,
                metadata=user_metadata_payload,
            )
        else:
            user_uuid = None

        session_uuid: uuid.UUID | None
        if session_id is not None:
            session_uuid = _coerce_uuid(session_id)
        elif session is not None:
            session_uuid = self.ensure_session(
                user_uuid,
                session,
                metadata=session_metadata_payload,
            )
        else:
            session_uuid = None

        if session_uuid is not None:
            self.ensure_conversation(conversation_uuid, session_id=session_uuid)

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
                meta=message_metadata,
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
                existing = dict(message.meta or {})
                existing.update(metadata)
                message.meta = existing
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

    # -- inspection helpers --------------------------------------------------

    def get_conversation(self, conversation_id: Any) -> Optional[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            record = session.get(Conversation, conversation_uuid)
            if record is None:
                return None
            payload: Dict[str, Any] = {
                "id": str(record.id),
                "session_id": str(record.session_id) if record.session_id else None,
                "title": record.title,
                "metadata": dict(record.meta or {}),
                "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
            }
            if record.archived_at is not None:
                payload["archived_at"] = record.archived_at.astimezone(timezone.utc).isoformat()
            return payload

    def get_message(self, conversation_id: Any, message_id: Any) -> Dict[str, Any]:
        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if message is None or message.conversation_id != conversation_uuid:
                raise ValueError("Unknown message or conversation")
            return self._serialize_message(message)

    def list_conversations_for_tenant(self, tenant_id: str) -> List[Dict[str, Any]]:
        tenant_key = str(tenant_id)
        with self._session_scope() as session:
            stmt = select(Conversation).where(
                Conversation.meta.contains({"tenant_id": tenant_key})
            )
            rows = session.execute(stmt).scalars().all()

        conversations: List[Dict[str, Any]] = []
        for record in rows:
            payload: Dict[str, Any] = {
                "id": str(record.id),
                "session_id": str(record.session_id) if record.session_id else None,
                "title": record.title,
                "metadata": dict(record.meta or {}),
                "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
            }
            if record.archived_at is not None:
                payload["archived_at"] = record.archived_at.astimezone(timezone.utc).isoformat()
            conversations.append(payload)
        return conversations

    def fetch_messages(
        self,
        conversation_id: Any,
        *,
        limit: int = 20,
        cursor: Optional[tuple[datetime, uuid.UUID]] = None,
        direction: str = "forward",
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = True,
    ) -> List[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            stmt = select(Message).where(Message.conversation_id == conversation_uuid)
            if metadata_filter:
                stmt = stmt.where(Message.meta.contains(dict(metadata_filter)))
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))

            if direction == "backward":
                if cursor is not None:
                    created_at, message_uuid = cursor
                    stmt = stmt.where(
                        or_(
                            Message.created_at < created_at,
                            and_(
                                Message.created_at == created_at,
                                Message.id < message_uuid,
                            ),
                        )
                    )
                stmt = stmt.order_by(Message.created_at.desc(), Message.id.desc())
            else:
                if cursor is not None:
                    created_at, message_uuid = cursor
                    stmt = stmt.where(
                        or_(
                            Message.created_at > created_at,
                            and_(
                                Message.created_at == created_at,
                                Message.id > message_uuid,
                            ),
                        )
                    )
                stmt = stmt.order_by(Message.created_at.asc(), Message.id.asc())

            stmt = stmt.limit(max(limit, 1))
            rows = session.execute(stmt).scalars().all()

        messages = [self._serialize_message(row) for row in rows]
        if direction == "backward":
            messages.reverse()
        return messages

    def fetch_message_events(
        self,
        *,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        after: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self._session_scope() as session:
            stmt = select(MessageEvent)
            if conversation_id is not None:
                stmt = stmt.where(MessageEvent.conversation_id == _coerce_uuid(conversation_id))
            if message_id is not None:
                stmt = stmt.where(MessageEvent.message_id == _coerce_uuid(message_id))
            if after is not None:
                stmt = stmt.where(MessageEvent.created_at > _coerce_dt(after))
            stmt = stmt.order_by(MessageEvent.created_at.asc(), MessageEvent.id.asc())
            if limit:
                stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()

        events: List[Dict[str, Any]] = []
        for event in rows:
            payload: Dict[str, Any] = {
                "id": str(event.id),
                "conversation_id": str(event.conversation_id),
                "message_id": str(event.message_id),
                "event_type": event.event_type,
                "metadata": dict(event.meta or {}),
                "created_at": event.created_at.astimezone(timezone.utc).isoformat(),
            }
            events.append(payload)
        return events

    # -- vector catalog operations -------------------------------------------

    def upsert_message_vectors(
        self,
        message_id: Any,
        vectors: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not vectors:
            return []

        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if message is None:
                raise ValueError("Unknown message supplied for vector upsert.")

            stored: List[Dict[str, Any]] = []
            for vector in vectors:
                payload = _normalize_vector_payload(message, vector)
                stored.append(self._upsert_vector_record(session, message, payload))
            return stored

    def fetch_message_vectors(
        self,
        *,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        message_ids: Optional[Sequence[Any]] = None,
        vector_keys: Optional[Sequence[str]] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._session_scope() as session:
            stmt = select(MessageVector)
            if conversation_id is not None:
                stmt = stmt.where(
                    MessageVector.conversation_id == _coerce_uuid(conversation_id)
                )
            if message_id is not None:
                stmt = stmt.where(MessageVector.message_id == _coerce_uuid(message_id))
            if message_ids:
                stmt = stmt.where(
                    MessageVector.message_id.in_([_coerce_uuid(item) for item in message_ids])
                )
            if vector_keys:
                stmt = stmt.where(MessageVector.vector_key.in_(list(vector_keys)))
            if provider is not None:
                stmt = stmt.where(MessageVector.provider == provider)

            stmt = stmt.order_by(MessageVector.created_at.asc())
            rows = session.execute(stmt).scalars().all()

        return [_serialize_vector(row) for row in rows]

    def delete_message_vectors(
        self,
        *,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        vector_keys: Optional[Sequence[str]] = None,
    ) -> int:
        with self._session_scope() as session:
            stmt = delete(MessageVector)
            if conversation_id is not None:
                stmt = stmt.where(
                    MessageVector.conversation_id == _coerce_uuid(conversation_id)
                )
            if message_id is not None:
                stmt = stmt.where(MessageVector.message_id == _coerce_uuid(message_id))
            if vector_keys:
                stmt = stmt.where(MessageVector.vector_key.in_(list(vector_keys)))

            result = session.execute(stmt)
            return int(result.rowcount or 0)

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
                meta=dict(asset.get("metadata") or {}),
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
            payload = _normalize_vector_payload(message, vector)
            self._upsert_vector_record(session, message, payload)

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
                meta=dict(event.get("metadata") or {}),
            )
            session.add(record)

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(message.id),
            "conversation_id": str(message.conversation_id),
            "role": message.role,
            "content": message.content,
            "metadata": dict(message.meta or {}),
            "timestamp": message.created_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "created_at": message.created_at.astimezone(timezone.utc).isoformat(),
            "updated_at": message.updated_at.astimezone(timezone.utc).isoformat()
            if message.updated_at is not None
            else None,
        }
        payload["user_id"] = str(message.user_id) if message.user_id else None
        session_ref = None
        if message.conversation is not None and message.conversation.session_id is not None:
            session_ref = str(message.conversation.session_id)
        payload["session_id"] = session_ref
        if message.extra:
            payload.update(message.extra)
        if message.client_message_id:
            payload["message_id"] = message.client_message_id
        if message.deleted_at is not None:
            payload["deleted_at"] = message.deleted_at.astimezone(timezone.utc).isoformat()
        return payload

    def _upsert_vector_record(
        self,
        session: Session,
        message: Message,
        payload: _VectorPayload,
    ) -> Dict[str, Any]:
        existing = session.execute(
            select(MessageVector).where(MessageVector.vector_key == payload.vector_key)
        ).scalar_one_or_none()

        if existing is None:
            existing = MessageVector(
                conversation_id=message.conversation_id,
                message_id=message.id,
                provider=payload.provider,
                vector_key=payload.vector_key,
            )
            session.add(existing)
        else:
            if existing.message_id != message.id:
                raise ValueError(
                    "Vector key already associated with a different message in the conversation store."
                )
            existing.provider = payload.provider

        existing.conversation_id = message.conversation_id
        existing.provider = payload.provider
        existing.vector_key = payload.vector_key
        existing.embedding = payload.embedding
        existing.embedding_model = payload.model
        existing.embedding_model_version = payload.version
        existing.embedding_checksum = payload.checksum
        existing.meta = dict(payload.metadata)
        existing.dimensions = payload.dimensions

        session.flush()
        return _serialize_vector(existing)
