"""SQLAlchemy models for the persistent conversation store."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
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
    metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    metadata = Column(JSONB, nullable=False, default=dict)
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
    metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    session = relationship("Session", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    role = Column(String(32), nullable=False)
    content = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=False, default=dict)
    extra = Column(JSONB, nullable=False, default=dict)
    client_message_id = Column(String(255), nullable=True)
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
    asset_type = Column(String(64), nullable=False)
    uri = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    message = relationship("Message", back_populates="assets")

    __table_args__ = (
        Index(
            "ix_message_assets_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
    )


class MessageVector(Base):
    __tablename__ = "message_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    provider = Column(String(128), nullable=True)
    embedding = Column(ARRAY(Float), nullable=True)
    metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    message = relationship("Message", back_populates="vectors")

    __table_args__ = (
        Index(
            "ix_message_vectors_conversation_created_at",
            "conversation_id",
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
    event_type = Column(String(64), nullable=False)
    metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    message = relationship("Message", back_populates="events")

    __table_args__ = (
        Index(
            "ix_message_events_conversation_created_at",
            "conversation_id",
            "created_at",
        ),
    )
