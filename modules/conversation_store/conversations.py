"""Conversation CRUD helpers for the conversation store."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from ._compat import Session, and_, delete, func, joinedload, or_, select
from ._shared import (
    _coerce_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_episode_tags,
    _normalize_json_like,
    _normalize_message_type,
    _normalize_status,
    _normalize_tenant_id,
    _tenant_filter,
)
from .models import (
    Conversation,
    EpisodicMemory,
    Message,
    MessageAsset,
    MessageEvent,
    MessageVector,
    Session as StoreSession,
    User,
    UserCredential,
)
from .vectors import VectorStore


class ConversationStore:
    """High-level CRUD helpers for conversations and messages."""

    def __init__(
        self,
        session_scope: Callable[[], ContextManager[Session]],
        vector_store: VectorStore,
        *,
        retention: Optional[Dict[str, Any]] = None,
        require_tenant_context: bool = False,
    ) -> None:
        self._session_scope = session_scope
        self._vectors = vector_store
        self._retention = retention or {}
        self._require_tenant_context = bool(require_tenant_context)
        self._vectors.set_message_serializer(self._serialize_message)

    # ------------------------------------------------------------------
    # Internal helpers

    def _normalize_tenant(self, tenant_id: Optional[Any]) -> Optional[str]:
        tenant = _normalize_tenant_id(tenant_id)
        if self._require_tenant_context and tenant is None:
            raise ValueError("Tenant context is required for conversation operations")
        return tenant

    def _retention_days(self, key: str) -> Optional[int]:
        value = self._retention.get(key)
        if value is None:
            return None
        try:
            days = int(value)
        except (TypeError, ValueError):
            return None
        return max(days, 0)

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

    # ------------------------------------------------------------------
    # User helpers

    def _lookup_user_by_username(
        self, session: Session, username: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[User]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        tenant = self._normalize_tenant(tenant_id)

        credential = session.execute(
            select(UserCredential)
            .options(joinedload(UserCredential.user))
            .where(UserCredential.username == cleaned)
            .where(_tenant_filter(UserCredential.tenant_id, tenant))
        ).scalar_one_or_none()

        if credential and credential.user is not None:
            return credential.user

        user_stmt = select(User).where(User.external_id == cleaned)
        user_stmt = user_stmt.where(_tenant_filter(User.tenant_id, tenant))
        user = session.execute(user_stmt).scalar_one_or_none()

        if user is not None:
            if credential and credential.user_id != user.id:
                credential.user_id = user.id
                session.flush()
            return user

        if credential and credential.user_id is not None:
            user = session.execute(
                select(User).where(User.id == credential.user_id)
            ).scalar_one_or_none()
            if user is not None:
                return user

        return None

    def ensure_user(
        self,
        external_id: Any,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[Any] = None,
    ) -> uuid.UUID:
        """Ensure that a user record exists for ``external_id`` and return its UUID."""

        if external_id is None:
            raise ValueError("External user identifier must not be None")

        identifier = str(external_id).strip()
        if not identifier:
            raise ValueError("External user identifier must not be empty")

        display = display_name.strip() if isinstance(display_name, str) else None
        metadata_payload = dict(metadata or {})
        tenant = self._normalize_tenant(tenant_id)

        with self._session_scope() as session:
            stmt = select(User).where(User.external_id == identifier)
            stmt = stmt.where(_tenant_filter(User.tenant_id, tenant))
            record = session.execute(stmt).scalar_one_or_none()

            if record is None:
                record = User(
                    external_id=identifier,
                    display_name=display,
                    tenant_id=tenant,
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
                if tenant is not None and record.tenant_id != tenant:
                    record.tenant_id = tenant
                    updated = True
                if updated:
                    session.flush()

            return record.id

    def get_user_profile(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None
        with self._session_scope() as session:
            user_record = self._lookup_user_by_username(session, cleaned, tenant_id=tenant_id)
            if user_record is None:
                return None

            profile_payload = (
                dict(user_record.meta or {}).get("profile")
                if isinstance(user_record.meta, Mapping)
                else None
            )
            if not isinstance(profile_payload, Mapping):
                profile_payload = {}
            else:
                profile_payload = {
                    str(key): _normalize_json_like(value)
                    for key, value in profile_payload.items()
                }

            documents_payload = (
                dict(user_record.meta or {}).get("documents")
                if isinstance(user_record.meta, Mapping)
                else None
            )
            if not isinstance(documents_payload, Mapping):
                documents_payload = {}
            else:
                documents_payload = {
                    str(key): _normalize_json_like(value)
                    for key, value in documents_payload.items()
                }

            return {
                "username": user_record.external_id or cleaned,
                "display_name": user_record.display_name,
                "profile": profile_payload,
                "documents": documents_payload,
                "user_id": str(user_record.id),
            }

    def list_user_profiles(
        self, *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        tenant = self._normalize_tenant(tenant_id)
        with self._session_scope() as session:
            stmt = select(User)
            if tenant is None:
                stmt = stmt.where(User.tenant_id.is_(None))
            else:
                stmt = stmt.where(User.tenant_id == tenant)
            stmt = stmt.order_by(User.created_at.asc())
            rows = session.execute(stmt).scalars().all()

            profiles: List[Dict[str, Any]] = []
            for row in rows:
                profile = (
                    dict(row.meta or {}).get("profile")
                    if isinstance(row.meta, Mapping)
                    else None
                )
                if not isinstance(profile, Mapping):
                    profile = {}
                documents = (
                    dict(row.meta or {}).get("documents")
                    if isinstance(row.meta, Mapping)
                    else None
                )
                if not isinstance(documents, Mapping):
                    documents = {}
                profiles.append(
                    {
                        "username": row.external_id,
                        "display_name": row.display_name,
                        "profile": {
                            str(key): _normalize_json_like(value)
                            for key, value in profile.items()
                        },
                        "documents": {
                            str(key): _normalize_json_like(value)
                            for key, value in documents.items()
                        },
                        "user_id": str(row.id),
                    }
                )
            return profiles

    def upsert_user_profile(
        self,
        username: str,
        *,
        tenant_id: Optional[Any] = None,
        display_name: Optional[str] = None,
        profile: Optional[Mapping[str, Any]] = None,
        documents: Optional[Mapping[str, Any]] = None,
        merge_documents: bool = False,
    ) -> Optional[Dict[str, Any]]:
        cleaned = str(username).strip()
        if not cleaned:
            return None

        metadata_payload = dict(profile or {})
        documents_payload = dict(documents or {})

        normalized_profile = _normalize_json_like(profile) if isinstance(profile, Mapping) else None
        normalized_documents = (
            _normalize_json_like(documents) if isinstance(documents, Mapping) else None
        )

        with self._session_scope() as session:
            user_record = session.execute(
                select(User).where(User.external_id == cleaned)
            ).scalar_one_or_none()

            if user_record is None:
                user_id = self.ensure_user(
                    cleaned,
                    tenant_id=tenant_id,
                    display_name=display_name,
                    metadata={"profile": metadata_payload or None},
                )
            else:
                user_id = user_record.id

        user_uuid = uuid.UUID(str(user_id))

        with self._session_scope() as session:
            user_record = session.execute(
                select(User).where(User.id == user_uuid)
            ).scalar_one()

            current_meta = dict(user_record.meta or {})

            if normalized_profile is not None:
                current_meta["profile"] = normalized_profile

            if normalized_documents is not None:
                existing_documents = {}
                if merge_documents:
                    stored_documents = current_meta.get("documents")
                    if isinstance(stored_documents, Mapping):
                        existing_documents = dict(stored_documents)
                existing_documents.update(
                    dict(normalized_documents) if isinstance(normalized_documents, Mapping) else {}
                )
                current_meta["documents"] = existing_documents

            user_record.meta = current_meta

            if display_name:
                cleaned_display = display_name.strip()
                if cleaned_display and user_record.display_name != cleaned_display:
                    user_record.display_name = cleaned_display

            session.flush()

            profile_payload = (
                dict(current_meta.get("profile"))
                if isinstance(current_meta.get("profile"), Mapping)
                else {}
            )
            documents_payload = (
                dict(current_meta.get("documents"))
                if isinstance(current_meta.get("documents"), Mapping)
                else {}
            )

            return {
                "username": user_record.external_id or cleaned,
                "display_name": user_record.display_name,
                "profile": profile_payload,
                "documents": documents_payload,
                "user_id": str(user_record.id),
            }

    # ------------------------------------------------------------------
    # Session and conversation helpers

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
        tenant_key = self._normalize_tenant(tenant_id)
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
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = self._normalize_tenant(tenant_id)
        before_dt = _coerce_dt(before) if before is not None else None
        limit_value = None if limit is None else max(int(limit), 0)

        with self._session_scope() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_uuid)
                .where(Message.tenant_id == tenant_key)
                .order_by(Message.created_at.desc(), Message.id.desc())
            )
            if before_dt is not None:
                stmt = stmt.where(Message.created_at < before_dt)
            if limit_value is not None and limit_value > 0:
                stmt = stmt.limit(limit_value)
            rows = session.execute(stmt).scalars().all()

        return [self._serialize_message(row) for row in rows]

    def add_message(
        self,
        conversation_id: Any,
        *,
        tenant_id: Any,
        role: str,
        content: Mapping[str, Any],
        message_type: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Any | None = None,
        session_id: Any | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        client_message_id: Optional[str] = None,
        created_at: Optional[Any] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        vectors: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        tenant_key = self._normalize_tenant(tenant_id)
        conversation_uuid = self.ensure_conversation(
            conversation_id,
            tenant_id=tenant_key,
            session_id=session_id,
        )
        user_uuid = _coerce_uuid(user_id) if user_id is not None else None
        message_role = str(role).strip() or "assistant"
        message_type_value = _normalize_message_type(message_type)
        status_value = _normalize_status(status)
        content_payload = dict(content)
        metadata_payload = dict(metadata or {})
        extra_payload = dict(extra or {})

        created_dt = _coerce_dt(created_at) if created_at is not None else datetime.now(timezone.utc)

        with self._session_scope() as session:
            record = Message(
                conversation_id=conversation_uuid,
                tenant_id=tenant_key,
                user_id=user_uuid,
                role=message_role,
                message_type=message_type_value,
                status=status_value,
                content=content_payload,
                meta=metadata_payload,
                extra=extra_payload,
                client_message_id=client_message_id,
                created_at=created_dt,
                updated_at=created_dt,
            )
            session.add(record)
            session.flush()

            self._store_assets(session, record, assets)
            self._store_events(session, record, events)
            self._vectors.store_message_vectors(session, record, vectors)

            session.flush()
            return self._serialize_message(record)

    def record_edit(
        self,
        message_id: Any,
        *,
        tenant_id: Any,
        content: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        vectors: Optional[List[Dict[str, Any]]] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        tenant_key = self._normalize_tenant(tenant_id)
        message_uuid = _coerce_uuid(message_id)

        with self._session_scope() as session:
            record = session.get(Message, message_uuid)
            if record is None or record.tenant_id != tenant_key:
                return None

            record.content = dict(content)
            if metadata is not None:
                record.meta = dict(metadata)
            if extra is not None:
                record.extra = dict(extra)
            record.updated_at = datetime.now(timezone.utc)

            self._store_assets(session, record, assets)
            self._store_events(session, record, events)
            self._vectors.store_message_vectors(session, record, vectors)

            session.flush()
            return self._serialize_message(record)

    def soft_delete_message(self, message_id: Any, *, tenant_id: Any) -> bool:
        tenant_key = self._normalize_tenant(tenant_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            record = session.get(Message, message_uuid)
            if record is None or record.tenant_id != tenant_key:
                return False
            if record.deleted_at is not None:
                return False
            record.deleted_at = datetime.now(timezone.utc)
            record.status = "deleted"
            session.flush()
            return True

    def hard_delete_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        tenant_key = self._normalize_tenant(tenant_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            session.execute(
                delete(MessageVector).where(MessageVector.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(MessageEvent).where(MessageEvent.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(MessageAsset).where(MessageAsset.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(Message).where(Message.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(EpisodicMemory).where(EpisodicMemory.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(Conversation)
                .where(Conversation.id == conversation_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )

    def archive_conversation(self, conversation_id: Any, *, tenant_id: Any) -> bool:
        tenant_key = self._normalize_tenant(tenant_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            record = session.get(Conversation, conversation_uuid)
            if record is None or record.tenant_id != tenant_key:
                return False
            if record.archived_at is not None:
                return False
            record.archived_at = datetime.now(timezone.utc)
            session.flush()
            return True

    def reset_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        tenant_key = self._normalize_tenant(tenant_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            session.execute(
                delete(MessageVector).where(MessageVector.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(MessageEvent).where(MessageEvent.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(MessageAsset).where(MessageAsset.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(Message).where(Message.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(EpisodicMemory).where(EpisodicMemory.conversation_id == conversation_uuid)
            )
            session.execute(
                delete(Conversation)
                .where(Conversation.id == conversation_uuid)
                .where(Conversation.tenant_id == tenant_key)
            )

    # ------------------------------------------------------------------
    # Episodic memory helpers

    def append_episodic_memory(
        self,
        *,
        tenant_id: Any,
        content: Mapping[str, Any],
        conversation_id: Any | None = None,
        message_id: Any | None = None,
        user_id: Any | None = None,
        title: Optional[str] = None,
        tags: Optional[Sequence[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        occurred_at: Any | None = None,
        expires_at: Any | None = None,
    ) -> Dict[str, Any]:
        if content is None:
            raise ValueError("Episode content must not be None")

        tenant_key = self._normalize_tenant(tenant_id)
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

        tenant_key = self._normalize_tenant(tenant_id)
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

        tenant_key = self._normalize_tenant(tenant_id)
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

    # ------------------------------------------------------------------
    # Conversation/message retrieval helpers

    def get_conversation(
        self, conversation_id: Any, *, tenant_id: Any
    ) -> Optional[Dict[str, Any]]:
        tenant_key = self._normalize_tenant(tenant_id)
        conversation_uuid = _coerce_uuid(conversation_id)
        with self._session_scope() as session:
            record = session.get(Conversation, conversation_uuid)
            if record is None or record.tenant_id != tenant_key:
                return None
            payload = {
                "id": str(record.id),
                "tenant_id": record.tenant_id,
                "session_id": str(record.session_id) if record.session_id else None,
                "metadata": dict(record.meta or {}),
                "created_at": _dt_to_iso(record.created_at),
                "updated_at": _dt_to_iso(record.updated_at),
                "archived_at": _dt_to_iso(record.archived_at),
            }
            return payload

    def get_message(
        self, message_id: Any, *, tenant_id: Any
    ) -> Optional[Dict[str, Any]]:
        tenant_key = self._normalize_tenant(tenant_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            record = session.get(Message, message_uuid)
            if record is None or record.tenant_id != tenant_key:
                return None
            return self._serialize_message(record)

    def list_conversations_for_tenant(
        self,
        *,
        tenant_id: Any,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
        include_archived: bool = True,
    ) -> List[Dict[str, Any]]:
        tenant_key = self._normalize_tenant(tenant_id)
        limit_value = None if limit is None else max(int(limit), 0)
        offset_value = max(int(offset), 0)
        order_value = str(order or "desc").lower()
        order_desc = order_value not in {"asc", "ascending"}

        with self._session_scope() as session:
            stmt = select(Conversation).where(Conversation.tenant_id == tenant_key)
            if not include_archived:
                stmt = stmt.where(Conversation.archived_at.is_(None))
            if order_desc:
                stmt = stmt.order_by(Conversation.created_at.desc(), Conversation.id.desc())
            else:
                stmt = stmt.order_by(Conversation.created_at.asc(), Conversation.id.asc())
            if offset_value:
                stmt = stmt.offset(offset_value)
            if limit_value is not None:
                stmt = stmt.limit(limit_value)
            rows = session.execute(stmt).scalars().all()

        conversations: List[Dict[str, Any]] = []
        for row in rows:
            conversations.append(
                {
                    "id": str(row.id),
                    "tenant_id": row.tenant_id,
                    "session_id": str(row.session_id) if row.session_id else None,
                    "metadata": dict(row.meta or {}),
                    "created_at": _dt_to_iso(row.created_at),
                    "updated_at": _dt_to_iso(row.updated_at),
                    "archived_at": _dt_to_iso(row.archived_at),
                }
        )
        return conversations

    def list_known_tenants(self) -> List[Any]:
        """Return tenant identifiers that have recorded conversations."""

        with self._session_scope() as session:
            rows = session.execute(
                select(Conversation.tenant_id).distinct().order_by(Conversation.tenant_id)
            ).all()

        tenants: List[Any] = []
        for tenant_id, in rows:
            if tenant_id is not None:
                tenants.append(tenant_id)
        return tenants

    def fetch_messages(
        self,
        *,
        conversation_id: Any,
        tenant_id: Any,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = self._normalize_tenant(tenant_id)
        limit_value = None if limit is None else max(int(limit), 0)
        offset_value = max(int(offset), 0)
        order_value = str(order or "desc").lower()
        order_desc = order_value not in {"asc", "ascending"}

        with self._session_scope() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_uuid)
                .where(Message.tenant_id == tenant_key)
            )
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))
            if order_desc:
                stmt = stmt.order_by(Message.created_at.desc(), Message.id.desc())
            else:
                stmt = stmt.order_by(Message.created_at.asc(), Message.id.asc())
            if offset_value:
                stmt = stmt.offset(offset_value)
            if limit_value is not None:
                stmt = stmt.limit(limit_value)
            rows = session.execute(stmt).scalars().all()

        return [self._serialize_message(row) for row in rows]

    def stream_conversation_messages(
        self,
        *,
        conversation_id: Any,
        tenant_id: Any,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
        include_deleted: bool = False,
        batch_size: int = 200,
    ) -> Iterator[Dict[str, Any]]:
        conversation_uuid = _coerce_uuid(conversation_id)
        tenant_key = self._normalize_tenant(tenant_id)
        limit_value = None if limit is None else max(int(limit), 0)
        offset_value = max(int(offset), 0)
        order_value = str(order or "desc").lower()
        order_desc = order_value not in {"asc", "ascending"}
        batch_size = max(int(batch_size), 1)

        with self._session_scope() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_uuid)
                .where(Message.tenant_id == tenant_key)
            )
            if not include_deleted:
                stmt = stmt.where(Message.deleted_at.is_(None))
            if order_desc:
                stmt = stmt.order_by(Message.created_at.desc(), Message.id.desc())
            else:
                stmt = stmt.order_by(Message.created_at.asc(), Message.id.asc())

            remaining = limit_value
            current_offset = offset_value
            while True:
                fetch_size = batch_size
                if remaining is not None:
                    if remaining <= 0:
                        break
                    fetch_size = min(fetch_size, remaining)
                windowed = stmt.offset(current_offset).limit(fetch_size)
                rows = session.execute(windowed).scalars().all()
                if not rows:
                    break
                for row in rows:
                    yield self._serialize_message(row)
                current_offset += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                    if remaining <= 0:
                        break
                if len(rows) < fetch_size:
                    break

    def query_messages_by_text(
        self,
        *,
        conversation_ids: Sequence[Any],
        tenant_id: Any,
        query_text: Optional[str] = None,
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

        tenant_key = self._normalize_tenant(tenant_id)
        normalized_text = (str(query_text or "").strip())
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

    def fetch_message_events(
        self,
        *,
        message_id: Any,
        tenant_id: Any,
    ) -> List[Dict[str, Any]]:
        tenant_key = self._normalize_tenant(tenant_id)
        message_uuid = _coerce_uuid(message_id)
        with self._session_scope() as session:
            rows = session.execute(
                select(MessageEvent)
                .where(MessageEvent.message_id == message_uuid)
                .where(MessageEvent.tenant_id == tenant_key)
                .order_by(MessageEvent.created_at.asc())
            ).scalars().all()

        events: List[Dict[str, Any]] = []
        for event in rows:
            payload = {
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

    # ------------------------------------------------------------------
    # Retention helpers

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
                        normalized_tenant = self._normalize_tenant(tenant_key)
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
