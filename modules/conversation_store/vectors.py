"""Vector catalog helpers for the conversation store."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Mapping, Optional, Sequence

from ._compat import Session, delete, joinedload, select
from ._shared import (
    _coerce_uuid,
    _coerce_vector_payload,
    _normalize_tenant_id,
)
from .models import Message, MessageVector


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


class VectorStore:
    """Manage vector embeddings associated with conversation messages."""

    def __init__(self, session_scope: Callable[[], ContextManager[Session]]) -> None:
        self._session_scope = session_scope
        self._serialize_message: Callable[[Message], Dict[str, Any]] | None = None

    def set_message_serializer(
        self, serializer: Callable[[Message], Dict[str, Any]]
    ) -> None:
        self._serialize_message = serializer

    # ------------------------------------------------------------------
    # Normalisation helpers

    @staticmethod
    def _normalize_vector_payload(message: Message, vector: Mapping[str, Any]) -> _VectorPayload:
        coerced = _coerce_vector_payload(message.id, message.conversation_id, vector)

        metadata = dict(coerced.metadata)

        return _VectorPayload(
            provider=coerced.provider,
            vector_key=coerced.vector_key,
            embedding=list(coerced.embedding),
            metadata=metadata,
            model=coerced.model,
            version=coerced.version,
            checksum=coerced.checksum,
            dimensions=len(coerced.embedding),
        )

    @staticmethod
    def _serialize_vector(vector: MessageVector) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(vector.id),
            "vector_key": vector.vector_key,
            "conversation_id": str(vector.conversation_id),
            "message_id": str(vector.message_id),
            "tenant_id": vector.tenant_id,
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

    # ------------------------------------------------------------------
    # Internal helpers

    def store_message_vectors(
        self,
        session: Session,
        message: Message,
        vectors: Optional[Sequence[Mapping[str, Any]]],
    ) -> None:
        if not vectors:
            return
        for vector in vectors:
            payload = self._normalize_vector_payload(message, vector)
            self._upsert_vector_record(session, message, payload)

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
                tenant_id=message.tenant_id,
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
        existing.tenant_id = message.tenant_id
        existing.provider = payload.provider
        existing.vector_key = payload.vector_key
        existing.embedding = payload.embedding
        existing.embedding_model = payload.model
        existing.embedding_model_version = payload.version
        existing.embedding_checksum = payload.checksum
        existing.meta = dict(payload.metadata)
        existing.dimensions = payload.dimensions

        session.flush()
        return self._serialize_vector(existing)

    # ------------------------------------------------------------------
    # Catalog operations

    def query_message_vectors(
        self,
        *,
        conversation_ids: Sequence[Any],
        tenant_id: Any,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = False,
        order: str = "desc",
        offset: int = 0,
        limit: Optional[int] = None,
        batch_size: int = 200,
        top_k: Optional[int] = None,
    ) -> Iterator[tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._serialize_message is None:
            raise RuntimeError("Message serializer has not been configured for VectorStore")

        conversation_uuids = [
            _coerce_uuid(identifier) for identifier in conversation_ids if identifier is not None
        ]
        if not conversation_uuids:
            return

        tenant_key = _normalize_tenant_id(tenant_id)
        order = "asc" if str(order or "").lower() == "asc" else "desc"
        batch_size = max(int(batch_size), 1)
        remaining = None if limit is None else max(int(limit), 0)
        offset_value = max(int(offset), 0)
        top_remaining = None if top_k is None else max(int(top_k), 0)

        with self._session_scope() as session:
            stmt = (
                select(MessageVector)
                .join(Message, Message.id == MessageVector.message_id)
                .where(MessageVector.conversation_id.in_(conversation_uuids))
                .where(MessageVector.tenant_id == tenant_key)
                .where(Message.tenant_id == tenant_key)
                .options(joinedload(MessageVector.message).joinedload(Message.conversation))
            )

            if metadata_filter:
                stmt = stmt.where(MessageVector.meta.contains(dict(metadata_filter)))
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))

            if order == "asc":
                stmt = stmt.order_by(Message.created_at.asc(), Message.id.asc())
            else:
                stmt = stmt.order_by(Message.created_at.desc(), Message.id.desc())

            while True:
                fetch_size = batch_size
                if remaining is not None:
                    if remaining <= 0:
                        break
                    fetch_size = min(fetch_size, remaining)
                if top_remaining is not None:
                    if top_remaining <= 0:
                        break
                    fetch_size = min(fetch_size, top_remaining)

                windowed = stmt.offset(offset_value).limit(fetch_size)
                rows = session.execute(windowed).scalars().all()
                if not rows:
                    break

                for vector in rows:
                    message = vector.message
                    if message is None:
                        continue
                    serialized_message = self._serialize_message(message)
                    payload = self._serialize_vector(vector)
                    yield serialized_message, payload

                offset_value += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                    if remaining <= 0:
                        break
                if top_remaining is not None:
                    top_remaining -= len(rows)
                    if top_remaining <= 0:
                        break
                if len(rows) < fetch_size:
                    break

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
                payload = self._normalize_vector_payload(message, vector)
                stored.append(self._upsert_vector_record(session, message, payload))
            return stored

    def fetch_message_vectors(
        self,
        *,
        tenant_id: Any | None = None,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        message_ids: Optional[Sequence[Any]] = None,
        vector_keys: Optional[Sequence[str]] = None,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        tenant_key = _normalize_tenant_id(tenant_id) if tenant_id is not None else None
        with self._session_scope() as session:
            stmt = select(MessageVector)
            if tenant_key is not None:
                stmt = stmt.where(MessageVector.tenant_id == tenant_key)
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

        return [self._serialize_vector(row) for row in rows]

    def delete_message_vectors(
        self,
        *,
        tenant_id: Any,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        vector_keys: Optional[Sequence[str]] = None,
    ) -> int:
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = delete(MessageVector)
            stmt = stmt.where(MessageVector.tenant_id == tenant_key)
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


__all__ = ["VectorStore"]
