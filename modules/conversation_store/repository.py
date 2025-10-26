"""Repository helpers for working with the conversation store."""

from __future__ import annotations

import contextlib
import hashlib
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy import and_, create_engine, delete, func, inspect, or_, select, text
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    def _raise_sqlalchemy(name: str):
        return RuntimeError(f"SQLAlchemy function '{name}' is unavailable in this environment")

    def create_engine(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("create_engine")

    def delete(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("delete")

    def func(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("func")

    def inspect(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("inspect")

    def select(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("select")

    def text(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("text")

    def and_(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("and_")

    def or_(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("or_")
try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover - lightweight fallback when SQLAlchemy is absent
    class Engine:  # type: ignore[assignment]
        pass

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - lightweight fallback when SQLAlchemy is absent
    class IntegrityError(Exception):  # type: ignore[assignment]
        pass

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.orm import Session, joinedload, sessionmaker
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    class Session:  # type: ignore[assignment]
        pass

    class _Sessionmaker:
        def __init__(self, *_args, **_kwargs):
            raise _raise_sqlalchemy("sessionmaker")

    def joinedload(*_args, **_kwargs):  # type: ignore[override]
        raise _raise_sqlalchemy("joinedload")

    sessionmaker = _Sessionmaker  # type: ignore[assignment]
else:  # pragma: no cover - sanitize stubbed implementations
    if not isinstance(sessionmaker, type):
        class _Sessionmaker:
            def __init__(self, *_args, **_kwargs):
                raise _raise_sqlalchemy("sessionmaker")

        sessionmaker = _Sessionmaker  # type: ignore[assignment]

from .models import (
    Base,
    Conversation,
    EpisodicMemory,
    GraphEdge,
    GraphNode,
    Message,
    MessageAsset,
    MessageEvent,
    MessageVector,
    PasswordResetToken,
    Session as StoreSession,
    User,
    UserCredential,
    UserLoginAttempt,
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


def _normalize_tenant_id(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("Tenant identifier must be a non-empty string")
    return text


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        normalised = text
        if normalised.endswith("Z"):
            normalised = normalised[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalised)
        except ValueError:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _dt_to_iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc).replace(microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


_DEFAULT_MESSAGE_TYPE = "text"
_DEFAULT_STATUS = "sent"
_MISSING = object()


def _normalize_message_type(value: Any) -> str:
    if value is None:
        return _DEFAULT_MESSAGE_TYPE
    text = str(value).strip()
    return text or _DEFAULT_MESSAGE_TYPE


def _normalize_status(value: Any) -> str:
    if value is None:
        return _DEFAULT_STATUS
    text = str(value).strip()
    return text or _DEFAULT_STATUS


def _normalize_episode_tags(tags: Optional[Sequence[Any]]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    if not tags:
        return normalized
    for tag in tags:
        if tag is None:
            continue
        text = str(tag).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_like(item) for item in value]
    return value


def _normalize_node_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Graph node key must be a non-empty string")
    return text


def _normalize_edge_key(value: Any | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_edge_type(value: Any | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_text_content(payload: Any) -> str:
    """Extract a plain-text representation from a message payload."""

    if isinstance(payload, Mapping):
        value = payload.get("text")
        if isinstance(value, str):
            return value
    if isinstance(payload, str):
        return payload
    return ""


def _normalize_attempts(attempts: Sequence[Any]) -> List[str]:
    normalised: List[str] = []
    for attempt in attempts:
        if isinstance(attempt, str):
            text = attempt.strip()
            if text:
                normalised.append(text)
                continue
        if isinstance(attempt, datetime):
            normalised.append(_dt_to_iso(_coerce_dt(attempt)) or "")
            continue
        candidate = str(attempt).strip()
        if candidate:
            normalised.append(candidate)
    return normalised


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


@dataclass
class _GraphNodePayload:
    key: str
    label: Optional[str]
    node_type: Optional[str]
    metadata: Optional[Dict[str, Any]]
    metadata_supplied: bool
    label_supplied: bool
    node_type_supplied: bool


@dataclass
class _GraphEdgePayload:
    edge_key: Optional[str]
    source_key: str
    target_key: str
    edge_type: Optional[str]
    weight: Optional[float]
    metadata: Optional[Dict[str, Any]]
    metadata_supplied: bool
    edge_type_supplied: bool
    weight_supplied: bool
    edge_key_supplied: bool


def _normalize_graph_node_payload(node: Mapping[str, Any]) -> _GraphNodePayload:
    if not isinstance(node, Mapping):
        raise TypeError("Graph node definitions must be mappings")

    raw_key = (
        node.get("key")
        or node.get("node_key")
        or node.get("node_id")
        or node.get("id")
    )
    key = _normalize_node_key(raw_key)

    label_supplied = "label" in node
    label = node.get("label")
    if label is not None:
        label = str(label).strip() or None

    node_type_supplied = "node_type" in node or "type" in node
    node_type = node.get("type") or node.get("node_type")
    if node_type is not None:
        node_type = str(node_type).strip() or None

    metadata_supplied = "metadata" in node
    metadata_payload: Optional[Dict[str, Any]]
    if metadata_supplied:
        metadata_payload = dict(_normalize_json_like(node.get("metadata") or {}))
    else:
        metadata_payload = None

    return _GraphNodePayload(
        key=key,
        label=label,
        node_type=node_type,
        metadata=metadata_payload,
        metadata_supplied=metadata_supplied,
        label_supplied=label_supplied,
        node_type_supplied=node_type_supplied,
    )


def _normalize_graph_edge_payload(edge: Mapping[str, Any]) -> _GraphEdgePayload:
    if not isinstance(edge, Mapping):
        raise TypeError("Graph edge definitions must be mappings")

    source = (
        edge.get("source")
        or edge.get("from")
        or edge.get("source_key")
        or edge.get("source_id")
    )
    target = (
        edge.get("target")
        or edge.get("to")
        or edge.get("target_key")
        or edge.get("target_id")
    )
    source_key = _normalize_node_key(source)
    target_key = _normalize_node_key(target)

    edge_key_supplied = "edge_key" in edge or "key" in edge
    raw_edge_key = edge.get("edge_key") or edge.get("key")
    edge_key = _normalize_edge_key(raw_edge_key)

    edge_type_supplied = "edge_type" in edge or "type" in edge
    raw_type = edge.get("edge_type") or edge.get("type")
    edge_type = _normalize_edge_type(raw_type)

    weight_supplied = "weight" in edge
    weight_value = edge.get("weight")
    weight = float(weight_value) if weight_value is not None else None

    metadata_supplied = "metadata" in edge
    metadata_payload: Optional[Dict[str, Any]]
    if metadata_supplied:
        metadata_payload = dict(_normalize_json_like(edge.get("metadata") or {}))
    else:
        metadata_payload = None

    return _GraphEdgePayload(
        edge_key=edge_key,
        source_key=source_key,
        target_key=target_key,
        edge_type=edge_type,
        weight=weight,
        metadata=metadata_payload,
        metadata_supplied=metadata_supplied,
        edge_type_supplied=edge_type_supplied,
        weight_supplied=weight_supplied,
        edge_key_supplied=edge_key_supplied,
    )


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


def _serialize_node(node: GraphNode) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(node.id),
        "tenant_id": node.tenant_id,
        "key": node.node_key,
        "label": node.label,
        "type": node.node_type,
        "metadata": dict(node.meta or {}),
        "created_at": _dt_to_iso(node.created_at),
        "updated_at": _dt_to_iso(node.updated_at),
    }
    return payload


def _serialize_edge(
    edge: GraphEdge,
    node_index: Optional[Mapping[uuid.UUID, GraphNode]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": str(edge.id),
        "tenant_id": edge.tenant_id,
        "edge_key": edge.edge_key,
        "source_id": str(edge.source_id),
        "target_id": str(edge.target_id),
        "edge_type": edge.edge_type,
        "weight": float(edge.weight) if edge.weight is not None else None,
        "metadata": dict(edge.meta or {}),
        "created_at": _dt_to_iso(edge.created_at),
        "updated_at": _dt_to_iso(edge.updated_at),
    }
    if node_index is not None:
        source_node = node_index.get(edge.source_id)
        target_node = node_index.get(edge.target_id)
        if source_node is not None:
            payload.setdefault("source_key", source_node.node_key)
        if target_node is not None:
            payload.setdefault("target_key", target_node.node_key)
    return payload


def _serialize_credential(record: UserCredential) -> Dict[str, Any]:
    attempts = list(record.failed_attempts or [])
    normalized_attempts = _normalize_attempts(attempts)
    data: Dict[str, Any] = {
        "id": int(record.id),
        "user_id": str(record.user_id) if record.user_id else None,
        "username": record.username,
        "password_hash": record.password_hash,
        "email": record.email,
        "name": record.name,
        "dob": record.dob,
        "last_login": _dt_to_iso(record.last_login),
        "failed_attempts": normalized_attempts,
        "lockout_until": _dt_to_iso(record.lockout_until),
        "created_at": _dt_to_iso(record.created_at),
        "updated_at": _dt_to_iso(record.updated_at),
    }
    return data


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

        from modules.task_store import ensure_task_schema  # local import to avoid circular deps
        from modules.job_store import ensure_job_schema

        engine: Engine | None = getattr(self._session_factory, "bind", None)
        if engine is None:
            with contextlib.ExitStack() as stack:
                try:
                    session = stack.enter_context(self._session_factory())
                except Exception as exc:  # pragma: no cover - defensive fallback
                    raise RuntimeError("Session factory is not bound to an engine") from exc
                engine = session.get_bind()
        if engine is None:
            raise RuntimeError("Session factory is not bound to an engine")
        Base.metadata.create_all(engine)
        ensure_task_schema(engine)
        ensure_job_schema(engine)
        if getattr(engine.dialect, "name", "") == "postgresql":
            inspector = inspect(engine)
            columns = {column["name"] for column in inspector.get_columns("messages")}
            with engine.begin() as connection:
                if "message_text_tsv" not in columns:
                    connection.execute(
                        text("ALTER TABLE messages ADD COLUMN message_text_tsv tsvector")
                    )
                connection.execute(
                    text(
                        """
                        UPDATE messages
                           SET message_text_tsv = to_tsvector(
                               'simple',
                               COALESCE(content->>'text', '')
                           )
                         WHERE message_text_tsv IS NULL
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS ix_messages_message_text_tsv
                            ON messages USING gin (message_text_tsv)
                        """
                    )
                )

    # -- credential helpers -------------------------------------------------

    def create_user_account(
        self,
        username: str,
        password_hash: str,
        email: str,
        *,
        name: Optional[str] = None,
        dob: Optional[str] = None,
        user_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        cleaned_username = str(username).strip()
        if not cleaned_username:
            raise ValueError("Username must not be empty")
        cleaned_email = str(email).strip().lower()
        if not cleaned_email:
            raise ValueError("Email must not be empty")

        normalised_name = name.strip() if isinstance(name, str) and name.strip() else None
        normalised_dob = dob.strip() if isinstance(dob, str) and dob.strip() else None
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None

        with self._session_scope() as session:
            record = UserCredential(
                user_id=user_uuid,
                username=cleaned_username,
                password_hash=password_hash,
                email=cleaned_email,
                name=normalised_name,
                dob=normalised_dob,
                failed_attempts=[],
            )
            session.add(record)
            try:
                session.flush()
            except IntegrityError:
                raise
            return _serialize_credential(record)

    def attach_credential(self, username: str, user_id: Any) -> Optional[str]:
        cleaned_username = str(username).strip()
        if not cleaned_username:
            return None
        user_uuid = _coerce_uuid(user_id)

        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned_username)
            ).scalar_one_or_none()
            if credential is None:
                return None
            if credential.user_id == user_uuid:
                existing = credential.user_id
                return str(existing) if existing is not None else None
            credential.user_id = user_uuid
            session.flush()
            return str(credential.user_id) if credential.user_id is not None else None

    def get_user_account(self, username: str) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if record is None:
                return None
            return _serialize_credential(record)

    def get_user_account_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        cleaned = str(email).strip().lower()
        if not cleaned:
            return None
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential).where(UserCredential.email == cleaned)
            ).scalar_one_or_none()
            if record is None:
                return None
            return _serialize_credential(record)

    def get_username_for_email(self, email: str) -> Optional[str]:
        record = self.get_user_account_by_email(email)
        if not record:
            return None
        return record["username"]

    def list_user_accounts(self) -> List[Dict[str, Any]]:
        with self._session_scope() as session:
            rows = session.execute(select(UserCredential)).scalars().all()
            return [_serialize_credential(row) for row in rows]

    def search_user_accounts(self, query_text: Optional[str]) -> List[Dict[str, Any]]:
        search_term = (str(query_text or "").strip().lower())
        with self._session_scope() as session:
            stmt = select(UserCredential)
            if search_term:
                like_term = f"%{search_term}%"
                stmt = stmt.where(
                    or_(
                        UserCredential.username.ilike(like_term),
                        UserCredential.email.ilike(like_term),
                        UserCredential.name.ilike(like_term),
                    )
                )
            rows = session.execute(stmt).scalars().all()
            return [_serialize_credential(row) for row in rows]

    def update_user_account(
        self,
        username: str,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
        password_hash: Optional[str] = None,
        user_id: Optional[Any] = None,
    ) -> bool:
        cleaned = str(username).strip()
        if not cleaned:
            return False
        with self._session_scope() as session:
            record = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if record is None:
                return False

            if email is not None:
                cleaned_email = str(email).strip().lower() or None
                record.email = cleaned_email
            if name is not None:
                record.name = name.strip() or None
            if dob is not None:
                record.dob = dob.strip() or None
            if password_hash is not None:
                record.password_hash = password_hash
            if user_id is not None:
                record.user_id = _coerce_uuid(user_id)
            try:
                session.flush()
            except IntegrityError:
                raise
            return True

    def delete_user_account(self, username: str) -> bool:
        cleaned = str(username).strip()
        if not cleaned:
            return False
        with self._session_scope() as session:
            result = session.execute(
                delete(UserCredential).where(UserCredential.username == cleaned)
            )
            return result.rowcount > 0

    def set_user_password(self, username: str, password_hash: str) -> bool:
        return self.update_user_account(username, password_hash=password_hash)

    def update_last_login(
        self,
        username: str,
        timestamp: Any,
        *,
        user_id: Optional[Any] = None,
    ) -> bool:
        cleaned = str(username).strip()
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None
        if not cleaned and user_uuid is None:
            return False
        moment = _coerce_dt(timestamp)
        with self._session_scope() as session:
            if user_uuid is not None:
                record = session.execute(
                    select(UserCredential).where(UserCredential.user_id == user_uuid)
                ).scalar_one_or_none()
            else:
                record = session.execute(
                    select(UserCredential).where(UserCredential.username == cleaned)
                ).scalar_one_or_none()
            if record is None:
                return False
            record.last_login = moment
            session.flush()
            return True

    def set_lockout_state(
        self,
        username: str,
        attempts: Sequence[Any],
        lockout_until: Optional[Any],
        *,
        user_id: Optional[Any] = None,
    ) -> bool:
        cleaned = str(username).strip()
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None
        if not cleaned and user_uuid is None:
            return False
        normalized_attempts = _normalize_attempts(attempts)
        lockout_dt = _coerce_dt(lockout_until) if lockout_until is not None else None
        with self._session_scope() as session:
            if user_uuid is not None:
                record = session.execute(
                    select(UserCredential).where(UserCredential.user_id == user_uuid)
                ).scalar_one_or_none()
            else:
                record = session.execute(
                    select(UserCredential).where(UserCredential.username == cleaned)
                ).scalar_one_or_none()
            if record is None:
                return False
            record.failed_attempts = normalized_attempts
            record.lockout_until = lockout_dt
            session.flush()
            return True

    def clear_lockout_state(self, username: str) -> bool:
        return self.set_lockout_state(username, [], None)

    def get_lockout_state(self, username: str) -> Optional[Dict[str, Any]]:
        record = self.get_user_account(username)
        if record is None:
            return None
        return {
            "username": record["username"],
            "failed_attempts": list(record.get("failed_attempts") or []),
            "lockout_until": record.get("lockout_until"),
        }

    def get_all_lockout_states(self) -> List[Dict[str, Any]]:
        accounts = self.list_user_accounts()
        states: List[Dict[str, Any]] = []
        for account in accounts:
            if account.get("failed_attempts") or account.get("lockout_until"):
                states.append(
                    {
                        "username": account["username"],
                        "failed_attempts": list(account.get("failed_attempts") or []),
                        "lockout_until": account.get("lockout_until"),
                    }
                )
        return states

    def record_login_attempt(
        self,
        username: Optional[str],
        timestamp: Any,
        successful: bool,
        reason: Optional[str],
    ) -> None:
        moment = _coerce_dt(timestamp)
        cleaned_username = None if username in (None, "") else str(username)
        trimmed_reason = None if reason in (None, "") else str(reason)
        with self._session_scope() as session:
            credential = None
            if cleaned_username:
                credential = session.execute(
                    select(UserCredential).where(UserCredential.username == cleaned_username)
                ).scalar_one_or_none()
            attempt = UserLoginAttempt(
                credential=credential,
                username=cleaned_username,
                attempted_at=moment,
                successful=bool(successful),
                reason=trimmed_reason,
            )
            session.add(attempt)

    def get_login_attempts(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned or limit <= 0:
            return []
        with self._session_scope() as session:
            stmt = (
                select(UserLoginAttempt)
                .where(UserLoginAttempt.username == cleaned)
                .order_by(UserLoginAttempt.attempted_at.desc(), UserLoginAttempt.id.desc())
                .limit(int(limit))
            )
            rows = session.execute(stmt).scalars().all()
        attempts: List[Dict[str, Any]] = []
        for row in rows:
            attempts.append(
                {
                    "timestamp": _dt_to_iso(row.attempted_at),
                    "successful": bool(row.successful),
                    "reason": row.reason if row.reason not in (None, "") else None,
                }
            )
        return attempts

    def prune_login_attempts(self, username: str, limit: int) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            return
        with self._session_scope() as session:
            if limit <= 0:
                session.execute(
                    delete(UserLoginAttempt).where(UserLoginAttempt.username == cleaned)
                )
                return

            subquery = (
                select(UserLoginAttempt.id)
                .where(UserLoginAttempt.username == cleaned)
                .order_by(UserLoginAttempt.attempted_at.desc(), UserLoginAttempt.id.desc())
                .limit(int(limit))
            )
            session.execute(
                delete(UserLoginAttempt).where(
                    and_(
                        UserLoginAttempt.username == cleaned,
                        ~UserLoginAttempt.id.in_(subquery),
                    )
                )
            )

    def upsert_password_reset_token(
        self,
        username: str,
        token_hash: str,
        expires_at: Any,
        created_at: Optional[Any] = None,
    ) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            raise ValueError("Username must not be empty when storing reset token")
        expires_dt = _coerce_dt(expires_at) if expires_at is not None else None
        created_dt = (
            _coerce_dt(created_at)
            if created_at is not None
            else _coerce_dt(datetime.now(timezone.utc))
        )
        with self._session_scope() as session:
            credential = session.execute(
                select(UserCredential).where(UserCredential.username == cleaned)
            ).scalar_one_or_none()
            if credential is None:
                raise ValueError(f"Unknown username '{cleaned}' for reset token")
            token = session.execute(
                select(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            ).scalar_one_or_none()
            if token is None:
                token = PasswordResetToken(
                    credential=credential,
                    username=cleaned,
                    token_hash=token_hash,
                    expires_at=expires_dt,
                    created_at=created_dt,
                )
                session.add(token)
            else:
                token.token_hash = token_hash
                token.expires_at = expires_dt
                token.created_at = created_dt

    def get_password_reset_token(self, username: str) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        with self._session_scope() as session:
            token = session.execute(
                select(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            ).scalar_one_or_none()
            if token is None:
                return None
            return {
                "username": token.username,
                "token_hash": token.token_hash,
                "expires_at": _dt_to_iso(token.expires_at),
            }

    def delete_password_reset_token(self, username: str) -> None:
        cleaned = str(username).strip()
        if not cleaned:
            return
        with self._session_scope() as session:
            session.execute(
                delete(PasswordResetToken).where(PasswordResetToken.username == cleaned)
            )

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
        tenant_id: Any,
        session_id: Any | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """Ensure that a conversation row exists and return its UUID."""

        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        session_uuid = _coerce_uuid(session_id) if session_id is not None else None
        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                conversation = Conversation(
                    id=conversation_uuid,
                    session_id=session_uuid,
                    tenant_id=tenant_key,
                    meta=metadata or {},
                )
                session.add(conversation)
            else:
                if conversation.tenant_id and conversation.tenant_id != tenant_key:
                    raise ValueError("Conversation belongs to a different tenant")
                if conversation.tenant_id != tenant_key:
                    conversation.tenant_id = tenant_key
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
        tenant_id: Any,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return serialized messages ordered by creation time ascending."""

        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = select(Message).where(Message.conversation_id == conversation_uuid)
            stmt = stmt.where(Message.tenant_id == tenant_key)
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
        tenant_id: Any,
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
        message_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a message and related resources, returning the stored payload."""

        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)

        normalized_type = _normalize_message_type(message_type)
        normalized_status = _normalize_status(status)

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

        self.ensure_conversation(
            conversation_uuid, tenant_id=tenant_key, session_id=session_uuid
        )

        extra_payload = dict(extra or {})
        message_metadata = dict(metadata or {})

        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                raise ValueError("Conversation could not be loaded after creation")
            if conversation.tenant_id != tenant_key:
                raise ValueError("Conversation belongs to a different tenant")
            if session_uuid is not None and conversation.session_id != session_uuid:
                conversation.session_id = session_uuid

            if message_id is not None:
                existing = session.execute(
                    select(Message).where(
                        and_(
                            Message.conversation_id == conversation_uuid,
                            Message.client_message_id == message_id,
                            Message.tenant_id == tenant_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    return self._serialize_message(existing)

            message = Message(
                conversation_id=conversation_uuid,
                tenant_id=tenant_key,
                user_id=user_uuid,
                role=role,
                message_type=normalized_type,
                status=normalized_status,
                content=content,
                meta=message_metadata,
                extra=extra_payload,
                client_message_id=message_id,
            )

            bind = session.get_bind()
            text_content = _extract_text_content(content)
            if (
                bind is not None
                and getattr(bind.dialect, "name", "") == "postgresql"
            ):
                message.message_text_tsv = func.to_tsvector("simple", text_content)
            else:
                message.message_text_tsv = None

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
                        "metadata": {
                            "role": role,
                            "message_type": normalized_type,
                            "status": normalized_status,
                        },
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
        tenant_id: Any,
        content: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        message_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing message and record corresponding events."""

        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if (
                message is None
                or message.conversation_id != conversation_uuid
                or message.tenant_id != tenant_key
            ):
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

            if message_type is not None:
                message.message_type = _normalize_message_type(message_type)
            if status is not None:
                message.status = _normalize_status(status)

            message.updated_at = datetime.now(timezone.utc)
            self._store_events(session, message, events)
            self._store_events(
                session,
                message,
                [
                    {
                        "event_type": "edited",
                        "metadata": {
                            "message_type": message.message_type,
                            "status": message.status,
                        },
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
        tenant_id: Any,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        message_type: Optional[str] = None,
    ) -> None:
        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if (
                message is None
                or message.conversation_id != conversation_uuid
                or message.tenant_id != tenant_key
            ):
                raise ValueError("Unknown message or conversation")

            if message_type is not None:
                message.message_type = _normalize_message_type(message_type)
            if status is None:
                message.status = _normalize_status("deleted")
            else:
                message.status = _normalize_status(status)
            message.deleted_at = datetime.now(timezone.utc)
            message.updated_at = datetime.now(timezone.utc)
            audit_metadata = {"reason": reason} if reason else {}
            if metadata:
                audit_metadata.update(metadata)
            self._store_events(
                session,
                message,
                [
                    {
                        "event_type": "soft_deleted",
                        "metadata": {
                            **audit_metadata,
                            "message_type": message.message_type,
                            "status": message.status,
                        },
                    }
                ],
            )

    def hard_delete_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None or conversation.tenant_id != tenant_key:
                return
            session.delete(conversation)

    def archive_conversation(
        self,
        conversation_id: Any,
        *,
        tenant_id: Any,
        metadata: Optional[Mapping[str, Any]] = None,
        archived_at: Optional[datetime] = None,
    ) -> bool:
        """Mark a conversation as archived without deleting it."""

        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        metadata_payload = dict(metadata or {})
        archived_moment = (
            _coerce_dt(archived_at)
            if archived_at is not None
            else datetime.now(timezone.utc)
        )

        with self._session_scope() as session:
            conversation = session.get(Conversation, conversation_uuid)
            if conversation is None:
                return False
            if conversation.tenant_id and conversation.tenant_id != tenant_key:
                raise ValueError("Conversation belongs to a different tenant")
            if conversation.tenant_id != tenant_key:
                conversation.tenant_id = tenant_key

            conversation.archived_at = archived_moment
            if metadata_payload:
                merged = dict(conversation.meta or {})
                merged.update(metadata_payload)
                conversation.meta = merged

            session.flush()
            return True

    def reset_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        """Remove all messages but leave the conversation row intact."""

        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            session.execute(
                delete(Message).where(
                    Message.conversation_id == conversation_uuid,
                    Message.tenant_id == tenant_key,
                )
            )

    # -- episodic memory helpers --------------------------------------------

    def append_episodic_memory(
        self,
        *,
        tenant_id: Any,
        content: Any,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Any | None = None,
        expires_at: Any | None = None,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        user_id: Any | None = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist an episodic memory entry and return the stored payload."""

        if content is None:
            raise ValueError("Episode content must not be None")

        tenant_key = _normalize_tenant_id(tenant_id)
        conversation_uuid = _coerce_uuid(conversation_id) if conversation_id is not None else None
        message_uuid = _coerce_uuid(message_id) if message_id is not None else None
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None

        tags_payload = _normalize_episode_tags(tags)
        metadata_payload = dict(metadata or {})
        content_payload = _normalize_json_like(content)

        occurred_dt = _coerce_dt(occurred_at) if occurred_at is not None else datetime.now(timezone.utc)
        expires_dt = _coerce_dt(expires_at) if expires_at is not None else None

        title_value = title.strip() if isinstance(title, str) else None
        if title_value == "":
            title_value = None

        if conversation_uuid is not None:
            self.ensure_conversation(conversation_uuid, tenant_id=tenant_key)

        with self._session_scope() as session:
            record = EpisodicMemory(
                tenant_id=tenant_key,
                conversation_id=conversation_uuid,
                message_id=message_uuid,
                user_id=user_uuid,
                title=title_value,
                content=content_payload,
                tags=tags_payload,
                meta=metadata_payload,
                occurred_at=occurred_dt,
                expires_at=expires_dt,
            )
            session.add(record)
            session.flush()
            return self._serialize_episode(record)

    def query_episodic_memories(
        self,
        *,
        tenant_id: Any,
        tags_all: Optional[Sequence[Any]] = None,
        tags_any: Optional[Sequence[Any]] = None,
        from_time: Any | None = None,
        to_time: Any | None = None,
        limit: Optional[int] = None,
        offset: int = 0,
        conversation_id: Any | None = None,
        include_expired: bool = False,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Query episodic memories for a tenant with optional filters."""

        tenant_key = _normalize_tenant_id(tenant_id)
        from_dt = _coerce_dt(from_time) if from_time is not None else None
        to_dt = _coerce_dt(to_time) if to_time is not None else None
        order_value = str(order or "desc").lower()
        order_desc = order_value not in {"asc", "ascending"}
        offset_value = max(int(offset), 0)
        limit_value = None if limit is None else max(int(limit), 0)

        required_tags = {tag.lower() for tag in _normalize_episode_tags(tags_all)}
        any_tags = {tag.lower() for tag in _normalize_episode_tags(tags_any)}

        now = datetime.now(timezone.utc)

        with self._session_scope() as session:
            stmt = select(EpisodicMemory).where(EpisodicMemory.tenant_id == tenant_key)
            if conversation_id is not None:
                stmt = stmt.where(EpisodicMemory.conversation_id == _coerce_uuid(conversation_id))
            if from_dt is not None:
                stmt = stmt.where(EpisodicMemory.occurred_at >= from_dt)
            if to_dt is not None:
                stmt = stmt.where(EpisodicMemory.occurred_at <= to_dt)
            if not include_expired:
                stmt = stmt.where(
                    or_(
                        EpisodicMemory.expires_at.is_(None),
                        EpisodicMemory.expires_at > now,
                    )
                )
            if order_desc:
                stmt = stmt.order_by(EpisodicMemory.occurred_at.desc(), EpisodicMemory.id.desc())
            else:
                stmt = stmt.order_by(EpisodicMemory.occurred_at.asc(), EpisodicMemory.id.asc())

            serialized_rows = [
                self._serialize_episode(record) for record in session.execute(stmt).scalars().all()
            ]

        episodes: List[Dict[str, Any]] = []
        for payload in serialized_rows:
            tags_lower = {
                str(tag).strip().lower()
                for tag in payload.get("tags", [])
                if isinstance(tag, str) and tag.strip()
            }
            if required_tags and not required_tags.issubset(tags_lower):
                continue
            if any_tags and not tags_lower.intersection(any_tags):
                continue
            episodes.append(payload)

        if offset_value:
            if offset_value >= len(episodes):
                return []
            episodes = episodes[offset_value:]
        if limit_value is not None:
            episodes = episodes[:limit_value]
        return episodes

    def prune_episodic_memories(
        self,
        *,
        tenant_id: Any,
        before: Any | None = None,
        expired_only: bool = False,
        limit: Optional[int] = None,
        conversation_id: Any | None = None,
    ) -> int:
        """Delete episodic memories matching filters and return the number removed."""

        tenant_key = _normalize_tenant_id(tenant_id)
        cutoff = _coerce_dt(before) if before is not None else None
        now = datetime.now(timezone.utc)
        limit_value = None if limit is None else max(int(limit), 0)

        with self._session_scope() as session:
            stmt = select(EpisodicMemory.id).where(EpisodicMemory.tenant_id == tenant_key)
            if conversation_id is not None:
                stmt = stmt.where(EpisodicMemory.conversation_id == _coerce_uuid(conversation_id))
            if cutoff is not None:
                stmt = stmt.where(EpisodicMemory.occurred_at < cutoff)
            if expired_only:
                stmt = stmt.where(EpisodicMemory.expires_at.is_not(None))
                stmt = stmt.where(EpisodicMemory.expires_at <= now)

            stmt = stmt.order_by(EpisodicMemory.occurred_at.asc(), EpisodicMemory.id.asc())
            if limit_value is not None:
                if limit_value == 0:
                    return 0
                stmt = stmt.limit(limit_value)

            ids = session.execute(stmt).scalars().all()
            if not ids:
                return 0
            session.execute(delete(EpisodicMemory).where(EpisodicMemory.id.in_(ids)))
            return len(ids)

    # -- memory graph helpers --------------------------------------------

    def upsert_graph_nodes(
        self,
        *,
        tenant_id: Any,
        nodes: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not nodes:
            return []

        tenant_key = _normalize_tenant_id(tenant_id)
        payloads = [_normalize_graph_node_payload(node) for node in nodes]

        with self._session_scope() as session:
            stored: List[Dict[str, Any]] = []
            for payload in payloads:
                record = session.execute(
                    select(GraphNode)
                    .where(GraphNode.tenant_id == tenant_key)
                    .where(GraphNode.node_key == payload.key)
                ).scalar_one_or_none()

                if record is None:
                    record = GraphNode(
                        tenant_id=tenant_key,
                        node_key=payload.key,
                        label=payload.label,
                        node_type=payload.node_type,
                        meta=payload.metadata or {},
                    )
                    session.add(record)
                    session.flush()
                else:
                    updated = False
                    if payload.label_supplied and record.label != payload.label:
                        record.label = payload.label
                        updated = True
                    if payload.node_type_supplied and record.node_type != payload.node_type:
                        record.node_type = payload.node_type
                        updated = True
                    if payload.metadata_supplied:
                        record.meta = payload.metadata or {}
                        updated = True
                    if updated:
                        session.flush()

                stored.append(_serialize_node(record))
            return stored

    def upsert_graph_edges(
        self,
        *,
        tenant_id: Any,
        edges: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not edges:
            return []

        tenant_key = _normalize_tenant_id(tenant_id)
        payloads = [_normalize_graph_edge_payload(edge) for edge in edges]

        required_keys: Set[str] = set()
        for payload in payloads:
            required_keys.add(payload.source_key)
            required_keys.add(payload.target_key)

        with self._session_scope() as session:
            rows = session.execute(
                select(GraphNode)
                .where(GraphNode.tenant_id == tenant_key)
                .where(GraphNode.node_key.in_(sorted(required_keys)))
            ).scalars().all()
            nodes_by_key = {record.node_key: record for record in rows}
            missing = sorted(required_keys - set(nodes_by_key))
            if missing:
                raise ValueError(
                    "Unknown graph node keys for edge upsert: " + ", ".join(missing)
                )

            stored_edges: List[Dict[str, Any]] = []
            for payload in payloads:
                source_node = nodes_by_key[payload.source_key]
                target_node = nodes_by_key[payload.target_key]

                record: Optional[GraphEdge] = None
                if payload.edge_key_supplied and payload.edge_key is not None:
                    record = session.execute(
                        select(GraphEdge)
                        .where(GraphEdge.tenant_id == tenant_key)
                        .where(GraphEdge.edge_key == payload.edge_key)
                    ).scalar_one_or_none()

                if record is None:
                    stmt = (
                        select(GraphEdge)
                        .where(GraphEdge.tenant_id == tenant_key)
                        .where(GraphEdge.source_id == source_node.id)
                        .where(GraphEdge.target_id == target_node.id)
                    )
                    if payload.edge_type_supplied:
                        if payload.edge_type is None:
                            stmt = stmt.where(GraphEdge.edge_type.is_(None))
                        else:
                            stmt = stmt.where(GraphEdge.edge_type == payload.edge_type)
                    stmt = stmt.limit(1)
                    record = session.execute(stmt).scalar_one_or_none()

                if record is None:
                    record = GraphEdge(
                        tenant_id=tenant_key,
                        edge_key=payload.edge_key if payload.edge_key_supplied else None,
                        source=source_node,
                        target=target_node,
                        edge_type=payload.edge_type if payload.edge_type_supplied else None,
                        weight=payload.weight if payload.weight_supplied else None,
                        meta=payload.metadata or {},
                    )
                    session.add(record)
                    session.flush()
                else:
                    if payload.edge_key_supplied and record.edge_key != payload.edge_key:
                        record.edge_key = payload.edge_key
                    if record.source_id != source_node.id:
                        record.source = source_node
                    if record.target_id != target_node.id:
                        record.target = target_node
                    if payload.edge_type_supplied and record.edge_type != payload.edge_type:
                        record.edge_type = payload.edge_type
                    if payload.weight_supplied and record.weight != payload.weight:
                        record.weight = payload.weight
                    if payload.metadata_supplied:
                        record.meta = payload.metadata or {}
                    session.flush()

                stored_edges.append(
                    _serialize_edge(
                        record,
                        node_index={
                            source_node.id: source_node,
                            target_node.id: target_node,
                        },
                    )
                )

            return stored_edges

    def query_graph(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tenant_key = _normalize_tenant_id(tenant_id)
        requested_keys = {
            _normalize_node_key(item) for item in (node_keys or []) if item is not None
        }
        edge_type_filters = {
            value
            for value in (
                _normalize_edge_type(item)
                for item in (edge_types or [])
            )
            if value is not None
        }

        with self._session_scope() as session:
            node_stmt = select(GraphNode).where(GraphNode.tenant_id == tenant_key)
            if requested_keys:
                node_stmt = node_stmt.where(GraphNode.node_key.in_(sorted(requested_keys)))
            node_stmt = node_stmt.order_by(GraphNode.node_key.asc())
            seed_nodes = session.execute(node_stmt).scalars().all()

            nodes_by_id: Dict[uuid.UUID, GraphNode] = {record.id: record for record in seed_nodes}

            edges: List[GraphEdge]
            if requested_keys and not nodes_by_id:
                edges = []
            else:
                edge_stmt = select(GraphEdge).where(GraphEdge.tenant_id == tenant_key)
                if requested_keys:
                    node_ids = list(nodes_by_id.keys())
                    if node_ids:
                        edge_stmt = edge_stmt.where(
                            or_(
                                GraphEdge.source_id.in_(node_ids),
                                GraphEdge.target_id.in_(node_ids),
                            )
                        )
                if edge_type_filters:
                    edge_stmt = edge_stmt.where(
                        GraphEdge.edge_type.in_(sorted(edge_type_filters))
                    )
                edge_stmt = edge_stmt.order_by(
                    GraphEdge.created_at.asc(), GraphEdge.id.asc()
                )
                edges = session.execute(edge_stmt).scalars().all()

            missing_ids: Set[uuid.UUID] = set()
            for edge in edges:
                if edge.source_id not in nodes_by_id:
                    missing_ids.add(edge.source_id)
                if edge.target_id not in nodes_by_id:
                    missing_ids.add(edge.target_id)

            if missing_ids:
                extra_nodes = session.execute(
                    select(GraphNode)
                    .where(GraphNode.tenant_id == tenant_key)
                    .where(GraphNode.id.in_(list(missing_ids)))
                ).scalars().all()
                for record in extra_nodes:
                    nodes_by_id.setdefault(record.id, record)

            serialized_nodes = [
                _serialize_node(record)
                for record in sorted(nodes_by_id.values(), key=lambda item: item.node_key)
            ]
            serialized_edges = [
                _serialize_edge(edge, node_index=nodes_by_id) for edge in edges
            ]

            return {"nodes": serialized_nodes, "edges": serialized_edges}

    def delete_graph_entries(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_keys: Optional[Sequence[Any]] = None,
        edge_ids: Optional[Sequence[Any]] = None,
    ) -> Dict[str, int]:
        tenant_key = _normalize_tenant_id(tenant_id)

        normalized_node_keys = {
            _normalize_node_key(item)
            for item in (node_keys or [])
            if item not in (None, "")
        }
        normalized_edge_keys = {
            _normalize_edge_key(item)
            for item in (edge_keys or [])
            if item not in (None, "")
        }
        normalized_edge_keys.discard(None)
        edge_uuid_values = {
            _coerce_uuid(item)
            for item in (edge_ids or [])
            if item not in (None, "")
        }

        removed_nodes = 0
        removed_edges = 0

        with self._session_scope() as session:
            if normalized_edge_keys or edge_uuid_values:
                edge_stmt = delete(GraphEdge).where(GraphEdge.tenant_id == tenant_key)
                if normalized_edge_keys:
                    edge_stmt = edge_stmt.where(
                        GraphEdge.edge_key.in_(sorted(normalized_edge_keys))
                    )
                if edge_uuid_values:
                    edge_stmt = edge_stmt.where(GraphEdge.id.in_(list(edge_uuid_values)))
                result = session.execute(edge_stmt)
                removed_edges = int(result.rowcount or 0)

            if normalized_node_keys:
                node_stmt = delete(GraphNode).where(GraphNode.tenant_id == tenant_key)
                node_stmt = node_stmt.where(GraphNode.node_key.in_(sorted(normalized_node_keys)))
                result = session.execute(node_stmt)
                removed_nodes = int(result.rowcount or 0)

        return {"nodes": removed_nodes, "edges": removed_edges}

    # -- inspection helpers --------------------------------------------------

    def get_conversation(
        self, conversation_id: Any, *, tenant_id: Any | None = None
    ) -> Optional[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id) if tenant_id is not None else None
        with self._session_scope() as session:
            record = session.get(Conversation, conversation_uuid)
            if record is None:
                return None
            if tenant_key is not None and record.tenant_id != tenant_key:
                return None
            payload: Dict[str, Any] = {
                "id": str(record.id),
                "session_id": str(record.session_id) if record.session_id else None,
                "title": record.title,
                "metadata": dict(record.meta or {}),
                "tenant_id": record.tenant_id,
                "created_at": record.created_at.astimezone(timezone.utc).isoformat(),
            }
            if record.archived_at is not None:
                payload["archived_at"] = record.archived_at.astimezone(timezone.utc).isoformat()
            return payload

    def get_message(
        self, conversation_id: Any, message_id: Any, *, tenant_id: Any
    ) -> Dict[str, Any]:
        conversation_uuid = _coerce_uuid(conversation_id)
        message_uuid = _coerce_uuid(message_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            message = session.get(Message, message_uuid)
            if (
                message is None
                or message.conversation_id != conversation_uuid
                or message.tenant_id != tenant_key
            ):
                raise ValueError("Unknown message or conversation")
            return self._serialize_message(message)

    def list_conversations_for_tenant(
        self,
        tenant_id: str,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
    ) -> List[Dict[str, Any]]:
        tenant_key = _normalize_tenant_id(tenant_id)
        order_value = str(order or "desc").lower()
        order_desc = order_value not in {"asc", "ascending"}
        offset_value = max(int(offset), 0)
        window = None if limit is None else max(int(limit), 0)

        with self._session_scope() as session:
            stmt = select(Conversation).where(Conversation.tenant_id == tenant_key)
            if order_desc:
                stmt = stmt.order_by(
                    Conversation.created_at.desc(), Conversation.id.desc()
                )
            else:
                stmt = stmt.order_by(
                    Conversation.created_at.asc(), Conversation.id.asc()
                )

            if offset_value:
                stmt = stmt.offset(offset_value)
            if window is not None and window > 0:
                stmt = stmt.limit(window)

            rows = session.execute(stmt).scalars().all()

        conversations: List[Dict[str, Any]] = []
        for record in rows:
            payload: Dict[str, Any] = {
                "id": str(record.id),
                "session_id": str(record.session_id) if record.session_id else None,
                "title": record.title,
                "metadata": dict(record.meta or {}),
                "tenant_id": record.tenant_id,
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
        tenant_id: Any,
        limit: int = 20,
        cursor: Optional[tuple[datetime, uuid.UUID]] = None,
        direction: str = "forward",
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = True,
        message_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = select(Message).where(Message.conversation_id == conversation_uuid)
            stmt = stmt.where(Message.tenant_id == tenant_key)
            if metadata_filter:
                stmt = stmt.where(Message.meta.contains(dict(metadata_filter)))
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))
            if message_types:
                normalized_types = {
                    _normalize_message_type(item) for item in message_types if item is not None
                }
                if normalized_types:
                    stmt = stmt.where(Message.message_type.in_(sorted(normalized_types)))
            if statuses:
                normalized_statuses = {
                    _normalize_status(item) for item in statuses if item is not None
                }
                if normalized_statuses:
                    stmt = stmt.where(Message.status.in_(sorted(normalized_statuses)))

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

    def stream_conversation_messages(
        self,
        conversation_id: Any,
        *,
        tenant_id: Any,
        batch_size: int = 200,
        cursor: Optional[tuple[datetime, uuid.UUID]] = None,
        direction: str = "forward",
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = True,
        message_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield messages for the conversation by repeatedly paging ``fetch_messages``."""

        window = max(int(batch_size), 1)
        next_cursor = cursor

        while True:
            messages = self.fetch_messages(
                conversation_id,
                tenant_id=tenant_id,
                limit=window,
                cursor=next_cursor,
                direction=direction,
                metadata_filter=metadata_filter,
                include_deleted=include_deleted,
                message_types=message_types,
                statuses=statuses,
            )

            if not messages:
                break

            for payload in messages:
                yield payload

            if len(messages) < window:
                break

            anchor = messages[0] if direction == "backward" else messages[-1]
            created_value = anchor.get("created_at") or anchor.get("timestamp")
            if created_value is None:
                break

            try:
                created_at = _coerce_dt(created_value)
                message_uuid = _coerce_uuid(anchor.get("id"))
            except Exception:
                break

            next_cursor = (created_at, message_uuid)

    def query_messages_by_text(
        self,
        *,
        conversation_ids: Sequence[Any],
        tenant_id: Any,
        text: str = "",
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_deleted: bool = False,
        order: str = "desc",
        offset: int = 0,
        limit: Optional[int] = None,
        batch_size: int = 200,
    ) -> Iterator[Dict[str, Any]]:
        conversation_uuids = [
            _coerce_uuid(identifier) for identifier in conversation_ids if identifier is not None
        ]
        if not conversation_uuids:
            return

        tenant_key = _normalize_tenant_id(tenant_id)
        normalized_text = str(text or "").strip()
        order = "asc" if str(order or "").lower() == "asc" else "desc"
        batch_size = max(int(batch_size), 1)
        remaining = None if limit is None else max(int(limit), 0)
        offset_value = max(int(offset), 0)

        with self._session_scope() as session:
            bind = session.get_bind()
            supports_full_text = bool(
                bind is not None and getattr(bind.dialect, "name", "") == "postgresql"
            )
            stmt = (
                select(Message)
                .where(Message.conversation_id.in_(conversation_uuids))
                .where(Message.tenant_id == tenant_key)
                .options(joinedload(Message.conversation))
            )
            if metadata_filter:
                stmt = stmt.where(Message.meta.contains(dict(metadata_filter)))
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))

            content_text = Message.content["text"].astext
            if normalized_text:
                if supports_full_text and hasattr(Message, "message_text_tsv"):
                    ts_query = func.plainto_tsquery("simple", normalized_text)
                    stmt = stmt.where(
                        Message.message_text_tsv.match(
                            ts_query, postgresql_regconfig="simple"
                        )
                    )
                else:
                    pattern = f"%{normalized_text}%"
                    stmt = stmt.where(content_text.ilike(pattern))

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

                windowed = stmt.offset(offset_value).limit(fetch_size)
                rows = session.execute(windowed).scalars().all()
                if not rows:
                    break

                serialized = [self._serialize_message(row) for row in rows]
                for payload in serialized:
                    yield payload

                offset_value += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                    if remaining <= 0:
                        break
                if len(rows) < fetch_size:
                    break

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
                    yield self._serialize_message(message), _serialize_vector(vector)
                    if top_remaining is not None:
                        top_remaining -= 1
                        if top_remaining <= 0:
                            break

                offset_value += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                    if remaining <= 0:
                        break
                if top_remaining is not None and top_remaining <= 0:
                    break
                if len(rows) < fetch_size:
                    break

    def fetch_message_events(
        self,
        *,
        tenant_id: Any,
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        after: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        tenant_key = _normalize_tenant_id(tenant_id)
        with self._session_scope() as session:
            stmt = select(MessageEvent)
            stmt = stmt.where(MessageEvent.tenant_id == tenant_key)
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
                "tenant_id": event.tenant_id,
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

        return [_serialize_vector(row) for row in rows]

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

    # -- retention helpers --------------------------------------------------

    def _retention_days(self, key: str) -> Optional[int]:
        value = self._retention.get(key)
        if value is None:
            return None
        try:
            days = int(value)
        except (TypeError, ValueError):
            return None
        return max(days, 0)

    def prune_expired_messages(self, *, now: Optional[datetime] = None) -> Dict[str, int]:
        """Apply message-level retention policies and return summary counts."""

        moment = now or datetime.now(timezone.utc)
        stats = {"soft_deleted": 0, "hard_deleted": 0}

        soft_delete_after = self._retention_days("soft_delete_after_days")
        soft_delete_grace = self._retention_days("soft_delete_grace_days")
        retention_days = self._retention_days("message_retention_days")
        if retention_days is None:
            retention_days = self._retention_days("days")

        with self._session_scope() as session:
            if soft_delete_after and soft_delete_after > 0:
                cutoff = moment - timedelta(days=soft_delete_after)
                rows = (
                    session.execute(
                        select(Message).where(
                            Message.deleted_at.is_(None),
                            Message.created_at < cutoff,
                        )
                    )
                    .scalars()
                    .all()
                )
                for message in rows:
                    message.deleted_at = moment
                    message.status = "deleted"
                    stats["soft_deleted"] += 1
                if rows:
                    session.flush()

            if soft_delete_grace and soft_delete_grace > 0:
                cutoff = moment - timedelta(days=soft_delete_grace)
                result = session.execute(
                    delete(Message).where(
                        Message.deleted_at.is_not(None),
                        Message.deleted_at < cutoff,
                    )
                )
                stats["hard_deleted"] += int(result.rowcount or 0)

            if retention_days and retention_days > 0:
                cutoff = moment - timedelta(days=retention_days)
                result = session.execute(
                    delete(Message).where(Message.created_at < cutoff)
                )
                stats["hard_deleted"] += int(result.rowcount or 0)

        return stats

    def prune_archived_conversations(
        self, *, now: Optional[datetime] = None
    ) -> Dict[str, int]:
        """Apply conversation-level retention policies."""

        moment = now or datetime.now(timezone.utc)
        stats = {"archived": 0, "deleted": 0}

        archive_after = self._retention_days("conversation_archive_days")
        archived_retention = self._retention_days("archived_conversation_retention_days")
        tenant_limits = self._retention.get("tenant_limits")

        with self._session_scope() as session:
            if archive_after and archive_after > 0:
                cutoff = moment - timedelta(days=archive_after)
                rows = (
                    session.execute(
                        select(Conversation).where(
                            Conversation.archived_at.is_(None),
                            Conversation.created_at < cutoff,
                        )
                    )
                    .scalars()
                    .all()
                )
                for conversation in rows:
                    conversation.archived_at = moment
                    stats["archived"] += 1
                if rows:
                    session.flush()

            if isinstance(tenant_limits, Mapping):
                for tenant_key, policy in tenant_limits.items():
                    if not isinstance(policy, Mapping):
                        continue
                    max_conversations = policy.get("max_conversations")
                    if max_conversations is None:
                        continue
                    try:
                        limit = int(max_conversations)
                    except (TypeError, ValueError):
                        continue
                    if limit < 1:
                        continue
                    try:
                        normalized_tenant = _normalize_tenant_id(tenant_key)
                    except ValueError:
                        continue
                    rows = (
                        session.execute(
                            select(Conversation)
                            .where(
                                Conversation.tenant_id == normalized_tenant,
                                Conversation.archived_at.is_(None),
                            )
                            .order_by(
                                Conversation.created_at.desc(), Conversation.id.desc()
                            )
                        )
                        .scalars()
                        .all()
                    )
                    excess = rows[limit:]
                    for conversation in excess:
                        conversation.archived_at = moment
                        stats["archived"] += 1
                    if excess:
                        session.flush()

            if archived_retention and archived_retention > 0:
                cutoff = moment - timedelta(days=archived_retention)
                result = session.execute(
                    delete(Conversation).where(
                        Conversation.archived_at.is_not(None),
                        Conversation.archived_at < cutoff,
                    )
                )
                stats["deleted"] += int(result.rowcount or 0)

        return stats

    def run_retention(self, *, now: Optional[datetime] = None) -> Dict[str, Dict[str, int]]:
        """Execute all configured retention policies and return aggregated stats."""

        moment = now or datetime.now(timezone.utc)
        messages = self.prune_expired_messages(now=moment)
        conversations = self.prune_archived_conversations(now=moment)
        return {"messages": messages, "conversations": conversations}

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
                tenant_id=message.tenant_id,
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
                tenant_id=message.tenant_id,
                event_type=str(event.get("event_type") or event.get("type") or "event"),
                meta=dict(event.get("metadata") or {}),
            )
            session.add(record)

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(message.id),
            "conversation_id": str(message.conversation_id),
            "tenant_id": message.tenant_id,
            "role": message.role,
            "message_type": message.message_type,
            "status": message.status,
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

    def _serialize_episode(self, episode: EpisodicMemory) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": str(episode.id),
            "tenant_id": episode.tenant_id,
            "content": episode.content,
            "tags": list(episode.tags or []),
            "metadata": dict(episode.meta or {}),
            "occurred_at": episode.occurred_at.astimezone(timezone.utc).isoformat(),
            "created_at": episode.created_at.astimezone(timezone.utc).isoformat(),
            "updated_at": episode.updated_at.astimezone(timezone.utc).isoformat(),
        }
        if episode.title:
            payload["title"] = episode.title
        if episode.conversation_id:
            payload["conversation_id"] = str(episode.conversation_id)
        if episode.message_id:
            payload["message_id"] = str(episode.message_id)
        if episode.user_id:
            payload["user_id"] = str(episode.user_id)
        if episode.expires_at is not None:
            payload["expires_at"] = episode.expires_at.astimezone(timezone.utc).isoformat()
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
        return _serialize_vector(existing)
