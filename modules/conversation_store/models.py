"""SQLAlchemy models for the persistent conversation store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID, TSVECTOR
from sqlalchemy.types import JSON, TypeDecorator

try:  # pragma: no cover - optional dependency for pgvector support
    from pgvector.sqlalchemy import Vector as PGVector  # type: ignore
except Exception:  # pragma: no cover - fallback when pgvector is unavailable
    PGVector = None  # type: ignore
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class PortableJSON(TypeDecorator):
    """Dialect-aware JSON type that prefers JSONB when available."""

    impl = JSON
    cache_ok = True

    _json_impl = JSON()
    _jsonb_impl = JSONB()

    def load_dialect_impl(self, dialect):  # type: ignore[override]
        if dialect.name == "postgresql":
            return dialect.type_descriptor(self._jsonb_impl)
        return dialect.type_descriptor(self._json_impl)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    external_id = Column(String(255), unique=False, nullable=True)
    tenant_id = Column(String(255), nullable=True, index=True)
    display_name = Column(String(255), nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    credentials = relationship(
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    tenant_id = Column(String(255), nullable=True, index=True)
    username = Column(String(255), nullable=False)
    password_hash = Column(String(512), nullable=False)
    email = Column(String(320), nullable=False)
    name = Column(String(255), nullable=True)
    dob = Column(String(32), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_attempts = Column(PortableJSON(), nullable=False, default=list)
    lockout_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    user = relationship("User", back_populates="credentials")
    login_attempts = relationship(
        "UserLoginAttempt",
        back_populates="credential",
        cascade="all, delete-orphan",
    )
    reset_token = relationship(
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    credential_id = Column(
        Integer,
        ForeignKey("user_credentials.id", ondelete="SET NULL"),
        nullable=True,
    )
    username = Column(String(255), nullable=True, index=True)
    attempted_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    successful = Column(Boolean, nullable=False, default=False)
    reason = Column(String(255), nullable=True)

    credential = relationship("UserCredential", back_populates="login_attempts")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    credential_id = Column(
        Integer,
        ForeignKey("user_credentials.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    username = Column(String(255), nullable=False, unique=True)
    token_hash = Column(String(128), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    credential = relationship("UserCredential", back_populates="reset_token")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    external_id = Column(String(255), unique=True, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="sessions")
    conversations = relationship(
        "Conversation", back_populates="session", cascade="all, delete-orphan"
    )



class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"))
    title = Column(String(255), nullable=True)
    tenant_id = Column(String(255), nullable=False, index=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("Session", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_conversations_tenant_created_at", "tenant_id", "created_at"),
    )



class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id = Column(String(255), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    role = Column(String(32), nullable=False)
    message_type = Column(
        String(32),
        nullable=False,
        default="text",
        server_default="text",
    )
    status = Column(
        String(32),
        nullable=False,
        default="sent",
        server_default="sent",
    )
    content = Column(PortableJSON(), nullable=False)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    extra = Column(PortableJSON(), nullable=False, default=dict)
    client_message_id = Column(String(255), nullable=True)
    message_text_tsv = Column(TSVECTOR, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User")
    assets = relationship(
        "MessageAsset", back_populates="message", cascade="all, delete-orphan"
    )
    vectors = relationship(
        "MessageVector", back_populates="message", cascade="all, delete-orphan"
    )
    events = relationship(
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id = Column(String(255), nullable=False, index=True)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(255), nullable=True)
    content = Column(PortableJSON(), nullable=False)
    tags = Column(PortableJSON(), nullable=False, default=list)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    conversation = relationship("Conversation")
    message = relationship("Message")
    user = relationship("User")

    __table_args__ = (
        Index("ix_episodic_memories_tenant_occurred_at", "tenant_id", "occurred_at"),
        Index("ix_episodic_memories_conversation_occurred_at", "conversation_id", "occurred_at"),
    )



class MessageAsset(Base):
    __tablename__ = "message_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id = Column(String(255), nullable=False, index=True)
    asset_type = Column(String(64), nullable=False)
    uri = Column(Text, nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    message = relationship("Message", back_populates="assets")

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



if PGVector is not None:  # pragma: no cover - exercised when pgvector installed
    _VECTOR_TYPE = PGVector()
else:
    _VECTOR_TYPE = ARRAY(Float)


class MessageVector(Base):
    __tablename__ = "message_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id = Column(String(255), nullable=False, index=True)
    provider = Column(String(128), nullable=True)
    vector_key = Column(String(255), nullable=False, unique=True)
    embedding = Column(_VECTOR_TYPE, nullable=True)
    embedding_model = Column(String(128), nullable=True)
    embedding_model_version = Column(String(64), nullable=True)
    embedding_checksum = Column(String(128), nullable=True)
    dimensions = Column(Integer, nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    message = relationship("Message", back_populates="vectors")

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

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id = Column(String(255), nullable=False, index=True)
    event_type = Column(String(64), nullable=False)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    message = relationship("Message", back_populates="events")

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

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id = Column(String(255), nullable=False, index=True)
    node_key = Column(String(255), nullable=False)
    label = Column(String(255), nullable=True)
    node_type = Column(String(64), nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    outgoing_edges = relationship(
        "GraphEdge",
        back_populates="source",
        cascade="all, delete-orphan",
        foreign_keys="GraphEdge.source_id",
    )
    incoming_edges = relationship(
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    tenant_id = Column(String(255), nullable=False, index=True)
    edge_key = Column(String(255), nullable=True)
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id = Column(
        UUID(as_uuid=True),
        ForeignKey("memory_graph_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    edge_type = Column(String(64), nullable=True)
    weight = Column(Float, nullable=True)
    meta = Column("metadata", PortableJSON(), nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    source = relationship(
        "GraphNode",
        back_populates="outgoing_edges",
        foreign_keys=[source_id],
    )
    target = relationship(
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

