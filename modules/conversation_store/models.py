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
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID, TSVECTOR

try:  # pragma: no cover - optional dependency for pgvector support
    from pgvector.sqlalchemy import Vector as PGVector  # type: ignore
except Exception:  # pragma: no cover - fallback when pgvector is unavailable
    PGVector = None  # type: ignore
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    external_id = Column(String(255), unique=True, nullable=True)
    display_name = Column(String(255), nullable=True)
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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


class UserCredential(Base):
    __tablename__ = "user_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    username = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(512), nullable=False)
    email = Column(String(320), nullable=False, unique=True)
    name = Column(String(255), nullable=True)
    dob = Column(String(32), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_attempts = Column(JSONB, nullable=False, default=list)
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
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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
    content = Column(JSONB, nullable=False)
    meta = Column("metadata", JSONB, nullable=False, default=dict)
    extra = Column(JSONB, nullable=False, default=dict)
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
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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
    meta = Column("metadata", JSONB, nullable=False, default=dict)
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


for _model in (User, Session, Conversation, Message, MessageAsset, MessageVector, MessageEvent):
    _attach_metadata_property(_model)

