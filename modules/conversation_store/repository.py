"""Repository helpers for working with the conversation store."""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ._compat import Session, sessionmaker
from .accounts import AccountStore
from .conversations import ConversationStore
from .graph import GraphStore
from .schema import create_conversation_engine, create_schema as _create_schema
from .vectors import VectorStore


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

        self._vectors = VectorStore(self._session_scope)
        self._conversations = ConversationStore(
            self._session_scope,
            self._vectors,
            retention=self._retention,
        )
        self._accounts = AccountStore(self._session_scope)
        self._graph = GraphStore(self._session_scope)

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

    # ------------------------------------------------------------------
    # Schema bootstrap

    def create_schema(self) -> None:
        _create_schema(self._session_factory)

    # ------------------------------------------------------------------
    # Account helpers

    def create_user_account(
        self,
        username: str,
        password_hash: str,
        email: str,
        *,
        tenant_id: Optional[Any] = None,
        name: Optional[str] = None,
        dob: Optional[str] = None,
        user_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self._accounts.create_user_account(
            username,
            password_hash,
            email,
            tenant_id=tenant_id,
            name=name,
            dob=dob,
            user_id=user_id,
        )

    def attach_credential(
        self, username: str, user_id: Any, *, tenant_id: Optional[Any] = None
    ) -> Optional[str]:
        return self._accounts.attach_credential(username, user_id, tenant_id=tenant_id)

    def get_user_account(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        return self._accounts.get_user_account(username, tenant_id=tenant_id)

    def get_user_account_by_email(
        self, email: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        return self._accounts.get_user_account_by_email(email, tenant_id=tenant_id)

    def get_username_for_email(
        self, email: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[str]:
        return self._accounts.get_username_for_email(email, tenant_id=tenant_id)

    def list_user_accounts(
        self, *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        return self._accounts.list_user_accounts(tenant_id=tenant_id)

    def search_user_accounts(
        self, query_text: Optional[str], *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        return self._accounts.search_user_accounts(query_text, tenant_id=tenant_id)

    def update_user_account(
        self,
        username: str,
        *,
        tenant_id: Optional[Any] = None,
        **updates: Any,
    ) -> Optional[Dict[str, Any]]:
        return self._accounts.update_user_account(username, tenant_id=tenant_id, **updates)

    def delete_user_account(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> bool:
        return self._accounts.delete_user_account(username, tenant_id=tenant_id)

    def set_user_password(self, username: str, password_hash: str) -> bool:
        return self._accounts.set_user_password(username, password_hash)

    def update_last_login(
        self, username: str, timestamp: Any, *, tenant_id: Optional[Any] = None
    ) -> bool:
        return self._accounts.update_last_login(username, timestamp, tenant_id=tenant_id)

    def set_lockout_state(
        self,
        username: str,
        *,
        failed_attempts: Optional[List[Any]] = None,
        lockout_until: Optional[Any] = None,
    ) -> bool:
        return self._accounts.set_lockout_state(
            username,
            failed_attempts=failed_attempts,
            lockout_until=lockout_until,
        )

    def clear_lockout_state(self, username: str) -> bool:
        return self._accounts.clear_lockout_state(username)

    def get_lockout_state(self, username: str) -> Optional[Dict[str, Any]]:
        return self._accounts.get_lockout_state(username)

    def get_all_lockout_states(self) -> List[Dict[str, Any]]:
        return self._accounts.get_all_lockout_states()

    def record_login_attempt(
        self,
        username: Optional[str],
        timestamp: Any,
        successful: bool,
        reason: Optional[str],
    ) -> None:
        self._accounts.record_login_attempt(username, timestamp, successful, reason)

    def get_login_attempts(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self._accounts.get_login_attempts(username, limit)

    def prune_login_attempts(self, username: str, limit: int) -> None:
        self._accounts.prune_login_attempts(username, limit)

    def upsert_password_reset_token(
        self,
        username: str,
        token_hash: str,
        expires_at: Any,
        created_at: Optional[Any] = None,
    ) -> None:
        self._accounts.upsert_password_reset_token(
            username,
            token_hash,
            expires_at,
            created_at=created_at,
        )

    def get_password_reset_token(self, username: str) -> Optional[Dict[str, Any]]:
        return self._accounts.get_password_reset_token(username)

    def delete_password_reset_token(self, username: str) -> None:
        self._accounts.delete_password_reset_token(username)

    # ------------------------------------------------------------------
    # Conversation helpers

    def ensure_user(
        self,
        external_id: Any,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[Any] = None,
    ) -> Any:
        return self._conversations.ensure_user(
            external_id,
            display_name=display_name,
            metadata=metadata,
            tenant_id=tenant_id,
        )

    def get_user_profile(
        self, username: str, *, tenant_id: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        return self._conversations.get_user_profile(username, tenant_id=tenant_id)

    def list_user_profiles(
        self, *, tenant_id: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        return self._conversations.list_user_profiles(tenant_id=tenant_id)

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
        return self._conversations.upsert_user_profile(
            username,
            tenant_id=tenant_id,
            display_name=display_name,
            profile=profile,
            documents=documents,
            merge_documents=merge_documents,
        )

    def ensure_session(
        self,
        user_id: Any | None,
        external_session_id: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        return self._conversations.ensure_session(
            user_id,
            external_session_id,
            metadata=metadata,
        )

    def ensure_conversation(
        self,
        conversation_id: Any,
        *,
        tenant_id: Any,
        session_id: Any | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._conversations.ensure_conversation(
            conversation_id,
            tenant_id=tenant_id,
            session_id=session_id,
            metadata=metadata,
        )

    def load_recent_messages(
        self,
        conversation_id: Any,
        *,
        tenant_id: Any,
        limit: Optional[int] = None,
        before: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return self._conversations.load_recent_messages(
            conversation_id,
            tenant_id=tenant_id,
            limit=limit,
            before=before,
        )

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
        return self._conversations.add_message(
            conversation_id,
            tenant_id=tenant_id,
            role=role,
            content=content,
            message_type=message_type,
            status=status,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            extra=extra,
            client_message_id=client_message_id,
            created_at=created_at,
            assets=assets,
            events=events,
            vectors=vectors,
        )

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
        return self._conversations.record_edit(
            message_id,
            tenant_id=tenant_id,
            content=content,
            metadata=metadata,
            extra=extra,
            events=events,
            vectors=vectors,
            assets=assets,
        )

    def soft_delete_message(self, message_id: Any, *, tenant_id: Any) -> bool:
        return self._conversations.soft_delete_message(message_id, tenant_id=tenant_id)

    def hard_delete_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        self._conversations.hard_delete_conversation(conversation_id, tenant_id=tenant_id)

    def archive_conversation(self, conversation_id: Any, *, tenant_id: Any) -> bool:
        return self._conversations.archive_conversation(conversation_id, tenant_id=tenant_id)

    def reset_conversation(self, conversation_id: Any, *, tenant_id: Any) -> None:
        self._conversations.reset_conversation(conversation_id, tenant_id=tenant_id)

    def append_episodic_memory(self, **kwargs: Any) -> Dict[str, Any]:
        return self._conversations.append_episodic_memory(**kwargs)

    def query_episodic_memories(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._conversations.query_episodic_memories(**kwargs)

    def prune_episodic_memories(self, **kwargs: Any) -> int:
        return self._conversations.prune_episodic_memories(**kwargs)

    def get_conversation(
        self, conversation_id: Any, *, tenant_id: Any
    ) -> Optional[Dict[str, Any]]:
        return self._conversations.get_conversation(conversation_id, tenant_id=tenant_id)

    def get_message(
        self, message_id: Any, *, tenant_id: Any
    ) -> Optional[Dict[str, Any]]:
        return self._conversations.get_message(message_id, tenant_id=tenant_id)

    def list_conversations_for_tenant(
        self,
        *,
        tenant_id: Any,
        limit: Optional[int] = None,
        offset: int = 0,
        order: str = "desc",
        include_archived: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._conversations.list_conversations_for_tenant(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            order=order,
            include_archived=include_archived,
        )

    def list_known_tenants(self) -> List[Any]:
        return self._conversations.list_known_tenants()

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
        return self._conversations.fetch_messages(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            order=order,
            include_deleted=include_deleted,
        )

    def stream_conversation_messages(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        return self._conversations.stream_conversation_messages(**kwargs)

    def query_messages_by_text(self, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        return self._conversations.query_messages_by_text(**kwargs)

    def fetch_message_events(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._conversations.fetch_message_events(**kwargs)

    def prune_expired_messages(self, *, now: Optional[datetime] = None) -> Dict[str, int]:
        return self._conversations.prune_expired_messages(now=now)

    def prune_archived_conversations(
        self, *, now: Optional[datetime] = None
    ) -> Dict[str, int]:
        return self._conversations.prune_archived_conversations(now=now)

    def run_retention(self, *, now: Optional[datetime] = None) -> Dict[str, Dict[str, int]]:
        return self._conversations.run_retention(now=now)

    # ------------------------------------------------------------------
    # Graph helpers

    def upsert_graph_nodes(
        self,
        *,
        tenant_id: Any,
        nodes: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._graph.upsert_graph_nodes(tenant_id=tenant_id, nodes=nodes)

    def upsert_graph_edges(
        self,
        *,
        tenant_id: Any,
        edges: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._graph.upsert_graph_edges(tenant_id=tenant_id, edges=edges)

    def query_graph(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_types: Optional[Sequence[Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        return self._graph.query_graph(
            tenant_id=tenant_id,
            node_keys=node_keys,
            edge_types=edge_types,
        )

    def delete_graph_entries(
        self,
        *,
        tenant_id: Any,
        node_keys: Optional[Sequence[Any]] = None,
        edge_keys: Optional[Sequence[Any]] = None,
        edge_ids: Optional[Sequence[Any]] = None,
    ) -> Dict[str, int]:
        return self._graph.delete_graph_entries(
            tenant_id=tenant_id,
            node_keys=node_keys,
            edge_keys=edge_keys,
            edge_ids=edge_ids,
        )

    # ------------------------------------------------------------------
    # Vector helpers

    def query_message_vectors(self, **kwargs: Any):
        return self._vectors.query_message_vectors(**kwargs)

    def upsert_message_vectors(
        self,
        message_id: Any,
        vectors: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._vectors.upsert_message_vectors(message_id, vectors)

    def fetch_message_vectors(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return self._vectors.fetch_message_vectors(**kwargs)

    def delete_message_vectors(self, **kwargs: Any) -> int:
        return self._vectors.delete_message_vectors(**kwargs)


__all__ = [
    "ConversationStoreRepository",
    "create_conversation_engine",
]
