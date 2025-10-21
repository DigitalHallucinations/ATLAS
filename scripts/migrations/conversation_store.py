"""Alembic-style helpers for the conversation store schema."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from modules.conversation_store import Base

metadata = Base.metadata


def upgrade(connection: Connection) -> None:
    """Apply the base conversation store schema."""

    metadata.create_all(connection)


def downgrade(connection: Connection) -> None:
    """Drop the conversation store schema."""

    metadata.drop_all(connection)
