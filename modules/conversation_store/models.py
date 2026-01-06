"""SQLAlchemy models for the persistent conversation store."""

from __future__ import annotations

import uuid
import importlib.util
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.dialects.postgresql import TSVECTOR as _PG_TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as _PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

# Lazy import for pgvector - only required when using PostgreSQL with vector columns
_PGVector = None


def _get_pgvector():
    """Lazily import pgvector.sqlalchemy.Vector when needed."""
    global _PGVector
    if _PGVector is None:
        if importlib.util.find_spec("pgvector") is None:
            raise ImportError(
                "pgvector is required for PostgreSQL vector support. Install with "
                "`pip install pgvector psycopg[binary]` to enable PostgreSQL vector columns."
            )
        from pgvector.sqlalchemy import Vector as PGVector
        _PGVector = PGVector
    return _PGVector

from modules.store_common.model_utils import generate_uuid, utcnow


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class PortableJSON(TypeDecorator):
    """Dialect-aware JSON type that prefers JSONB when available."""

    impl = JSON
    cache_ok = True

    _json_impl = JSON()
    _jsonb_impl = _PG_JSONB()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(self._jsonb_impl)
        return dialect.type_descriptor(self._json_impl)


class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type."""

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(uuid.UUID(str(value)))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


class TextSearchVector(TypeDecorator):
    """Cross-dialect text search vector placeholder."""

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PG_TSVECTOR())
        return dialect.type_descriptor(Text())


class EmbeddingVector(TypeDecorator):
    """Store vector embeddings across SQL dialects."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            PGVector = _get_pgvector()
            return dialect.type_descriptor(PGVector())
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        return value


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    external_id: Mapped[Optional[str]] = mapped_column(String(255), unique=False, nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    credentials: Mapped[Optional["UserCredential"]] = relationship(
        "UserCredential",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", name="uq_users_tenant_external"),
        Index(
            "uq_users_external_single",
            "external_id",
            unique=True,
            postgresql_where=text("tenant_id IS NULL"),
            sqlite_where=text("tenant_id IS NULL"),
        ),
    )


class UserCredential(Base):
    __tablename__ = "user_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"))
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    dob: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_attempts: Mapped[List[Any]] = mapped_column(PortableJSON(), nullable=False, default=list)
    lockout_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    user: Mapped[Optional["User"]] = relationship("User", back_populates="credentials")
    login_attempts: Mapped[List["UserLoginAttempt"]] = relationship(
        "UserLoginAttempt",
        back_populates="credential",
        cascade="all, delete-orphan",
    )
    reset_token: Mapped[Optional["PasswordResetToken"]] = relationship(
        "PasswordResetToken",
        back_populates="credential",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "username", name="uq_user_credentials_tenant_username"),
        UniqueConstraint("tenant_id", "email", name="uq_user_credentials_tenant_email"),
        Index("ix_user_credentials_username", "username"),
        Index("ix_user_credentials_email", "email"),
        Index(
            "uq_user_credentials_username_single",
            "username",
            unique=True,
            postgresql_where=text("tenant_id IS NULL"),
            sqlite_where=text("tenant_id IS NULL"),
        ),
        Index(
            "uq_user_credentials_email_single",
            "email",
            unique=True,
            postgresql_where=text("tenant_id IS NULL"),
            sqlite_where=text("tenant_id IS NULL"),
        ),
    )


class UserLoginAttempt(Base):
    __tablename__ = "user_login_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    credential_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("user_credentials.id", ondelete="SET NULL"),
        nullable=True,
    )
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    successful: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    credential: Mapped[Optional["UserCredential"]] = relationship("UserCredential", back_populates="login_attempts")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    credential_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user_credentials.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    username: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    credential: Mapped["UserCredential"] = relationship("UserCredential", back_populates="reset_token")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    external_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    user_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"))
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[Optional["User"]] = relationship("User", back_populates="sessions")
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation", back_populates="session", cascade="all, delete-orphan"
    )



class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    session_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("sessions.id", ondelete="SET NULL"))
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[Optional["Session"]] = relationship("Session", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_conversations_tenant_created_at", "tenant_id", "created_at"),
    )



class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"))
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    message_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="text",
        server_default="text",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="sent",
        server_default="sent",
    )
    content: Mapped[Any] = mapped_column(PortableJSON(), nullable=False)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    extra: Mapped[Dict[str, Any]] = mapped_column(PortableJSON(), nullable=False, default=dict)
    client_message_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message_text_tsv: Mapped[Optional[Any]] = mapped_column(TextSearchVector(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
    user: Mapped[Optional["User"]] = relationship("User")
    assets: Mapped[List["MessageAsset"]] = relationship(
        "MessageAsset", back_populates="message", cascade="all, delete-orphan"
    )
    vectors: Mapped[List["MessageVector"]] = relationship(
        "MessageVector", back_populates="message", cascade="all, delete-orphan"
    )
    events: Mapped[List["MessageEvent"]] = relationship(
        "MessageEvent", back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "client_message_id",
            name="uq_messages_conversation_client_id",
        ),
        Index(
            "ix_messages_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
        Index(
            "ix_messages_tenant_conversation_created_at",
            "tenant_id",
            "conversation_id",
            "created_at",
        ),
        Index("ix_messages_message_type", "message_type"),
        Index("ix_messages_status", "status"),
    )



class EpisodicMemory(Base):
    __tablename__ = "episodic_memories"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    conversation_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    message_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[Any] = mapped_column(PortableJSON(), nullable=False)
    tags: Mapped[List[Any]] = mapped_column(PortableJSON(), nullable=False, default=list)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation")
    message: Mapped[Optional["Message"]] = relationship("Message")
    user: Mapped[Optional["User"]] = relationship("User")

    __table_args__ = (
        Index("ix_episodic_memories_tenant_occurred_at", "tenant_id", "occurred_at"),
        Index("ix_episodic_memories_conversation_occurred_at", "conversation_id", "occurred_at"),
    )



class MessageAsset(Base):
    __tablename__ = "message_assets"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False)
    uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    message: Mapped["Message"] = relationship("Message", back_populates="assets")

    __table_args__ = (
        Index(
            "ix_message_assets_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
        Index(
            "ix_message_assets_tenant_created_at",
            "tenant_id",
            "created_at",
        ),
    )



class MessageVector(Base):
    __tablename__ = "message_vectors"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    provider: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    vector_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    embedding: Mapped[Optional[Any]] = mapped_column(EmbeddingVector(), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    dimensions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    message: Mapped["Message"] = relationship("Message", back_populates="vectors")

    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "provider",
            "embedding_model",
            "embedding_model_version",
            name="uq_message_vectors_message_model",
        ),
        Index(
            "ix_message_vectors_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
        Index(
            "ix_message_vectors_conversation_vector_key",
            "conversation_id",
            "vector_key",
        ),
        Index(
            "ix_message_vectors_tenant_created_at",
            "tenant_id",
            "created_at",
        ),
    )



class MessageEvent(Base):
    __tablename__ = "message_events"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[UUID] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    message: Mapped["Message"] = relationship("Message", back_populates="events")

    __table_args__ = (
        Index(
            "ix_message_events_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
        Index(
            "ix_message_events_tenant_created_at",
            "tenant_id",
            "created_at",
        ),
    )


class GraphNode(Base):
    __tablename__ = "memory_graph_nodes"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    node_key: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    node_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    outgoing_edges: Mapped[List["GraphEdge"]] = relationship(
        "GraphEdge",
        back_populates="source",
        cascade="all, delete-orphan",
        foreign_keys="GraphEdge.source_id",
    )
    incoming_edges: Mapped[List["GraphEdge"]] = relationship(
        "GraphEdge",
        back_populates="target",
        cascade="all, delete-orphan",
        foreign_keys="GraphEdge.target_id",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "node_key",
            name="uq_memory_graph_nodes_tenant_key",
        ),
        Index(
            "ix_memory_graph_nodes_tenant_key",
            "tenant_id",
            "node_key",
        ),
    )


class GraphEdge(Base):
    __tablename__ = "memory_graph_edges"

    id: Mapped[UUID] = mapped_column(GUID(), primary_key=True, default=generate_uuid)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    edge_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    edge_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    meta: Mapped[Dict[str, Any]] = mapped_column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    source: Mapped["GraphNode"] = relationship(
        "GraphNode",
        back_populates="outgoing_edges",
        foreign_keys=[source_id],
    )
    target: Mapped["GraphNode"] = relationship(
        "GraphNode",
        back_populates="incoming_edges",
        foreign_keys=[target_id],
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "edge_key",
            name="uq_memory_graph_edges_tenant_key",
        ),
        UniqueConstraint(
            "tenant_id",
            "source_id",
            "target_id",
            "edge_type",
            name="uq_memory_graph_edges_relation",
        ),
        Index(
            "ix_memory_graph_edges_tenant_relation",
            "tenant_id",
            "source_id",
            "target_id",
        ),
    )


def _attach_metadata_property(model_cls):
    def _get(instance):
        value = getattr(instance, "meta")
        if value is None:
            value = {}
            setattr(instance, "meta", value)
        return value

    def _set(instance, value):
        setattr(instance, "meta", dict(value or {}))

    setattr(model_cls, "metadata", property(_get, _set))


for _model in (
    User,
    Session,
    Conversation,
    Message,
    EpisodicMemory,
    MessageAsset,
    MessageVector,
    MessageEvent,
    GraphNode,
    GraphEdge,
):
    _attach_metadata_property(_model)
