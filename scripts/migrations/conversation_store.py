"""Alembic-style helpers for the conversation store schema."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from modules.conversation_store import Base

metadata = Base.metadata


def upgrade(connection: Connection) -> None:
    """Apply the base conversation store schema."""

    metadata.create_all(connection)

    inspector = inspect(connection)
    columns = {column["name"] for column in inspector.get_columns("messages")}

    if "message_type" not in columns:
        connection.execute(
            text(
                "ALTER TABLE messages"
                " ADD COLUMN message_type VARCHAR(32) NOT NULL DEFAULT 'text'"
            )
        )
    if "status" not in columns:
        connection.execute(
            text("ALTER TABLE messages ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'sent'")
        )

    connection.execute(
        text(
            "UPDATE messages SET message_type = 'text' WHERE message_type IS NULL"
        )
    )
    connection.execute(
        text(
            "UPDATE messages SET status = 'deleted' WHERE deleted_at IS NOT NULL AND (status IS NULL OR status = 'sent')"
        )
    )

    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_messages_message_type ON messages (message_type)"
        )
    )
    connection.execute(
        text("CREATE INDEX IF NOT EXISTS ix_messages_status ON messages (status)")
    )


def downgrade(connection: Connection) -> None:
    """Drop the conversation store schema."""

    metadata.drop_all(connection)
