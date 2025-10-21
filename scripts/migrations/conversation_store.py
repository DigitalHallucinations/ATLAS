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

    existing_conversation_columns = {
        column["name"] for column in inspector.get_columns("conversations")
    }
    if "tenant_id" not in existing_conversation_columns:
        connection.execute(
            text("ALTER TABLE conversations ADD COLUMN tenant_id VARCHAR(255)")
        )

    existing_message_columns = columns
    if "tenant_id" not in existing_message_columns:
        connection.execute(
            text("ALTER TABLE messages ADD COLUMN tenant_id VARCHAR(255)")
        )

    existing_asset_columns = {
        column["name"] for column in inspector.get_columns("message_assets")
    }
    if "tenant_id" not in existing_asset_columns:
        connection.execute(
            text("ALTER TABLE message_assets ADD COLUMN tenant_id VARCHAR(255)")
        )

    existing_vector_columns = {
        column["name"] for column in inspector.get_columns("message_vectors")
    }
    if "tenant_id" not in existing_vector_columns:
        connection.execute(
            text("ALTER TABLE message_vectors ADD COLUMN tenant_id VARCHAR(255)")
        )

    existing_event_columns = {
        column["name"] for column in inspector.get_columns("message_events")
    }
    if "tenant_id" not in existing_event_columns:
        connection.execute(
            text("ALTER TABLE message_events ADD COLUMN tenant_id VARCHAR(255)")
        )

    connection.execute(
        text(
            """
            UPDATE conversations
               SET tenant_id = COALESCE(NULLIF("metadata"->>'tenant_id', ''), id::text)
             WHERE tenant_id IS NULL OR tenant_id = ''
            """
        )
    )

    connection.execute(
        text(
            """
            UPDATE messages AS m
               SET tenant_id = COALESCE(
                   NULLIF(m."metadata"->>'tenant_id', ''),
                   (SELECT tenant_id FROM conversations c WHERE c.id = m.conversation_id)
               )
             WHERE tenant_id IS NULL OR tenant_id = ''
            """
        )
    )

    connection.execute(
        text(
            """
            UPDATE message_assets AS a
               SET tenant_id = COALESCE(
                   NULLIF(a."metadata"->>'tenant_id', ''),
                   (SELECT tenant_id FROM messages m WHERE m.id = a.message_id),
                   (SELECT tenant_id FROM conversations c WHERE c.id = a.conversation_id)
               )
             WHERE tenant_id IS NULL OR tenant_id = ''
            """
        )
    )

    connection.execute(
        text(
            """
            UPDATE message_vectors AS v
               SET tenant_id = COALESCE(
                   NULLIF(v."metadata"->>'tenant_id', ''),
                   (SELECT tenant_id FROM messages m WHERE m.id = v.message_id),
                   (SELECT tenant_id FROM conversations c WHERE c.id = v.conversation_id)
               )
             WHERE tenant_id IS NULL OR tenant_id = ''
            """
        )
    )

    connection.execute(
        text(
            """
            UPDATE message_events AS e
               SET tenant_id = COALESCE(
                   NULLIF(e."metadata"->>'tenant_id', ''),
                   (SELECT tenant_id FROM messages m WHERE m.id = e.message_id),
                   (SELECT tenant_id FROM conversations c WHERE c.id = e.conversation_id)
               )
             WHERE tenant_id IS NULL OR tenant_id = ''
            """
        )
    )

    connection.execute(
        text(
            "ALTER TABLE conversations ALTER COLUMN tenant_id SET NOT NULL"
        )
    )
    connection.execute(
        text("ALTER TABLE messages ALTER COLUMN tenant_id SET NOT NULL")
    )
    connection.execute(
        text("ALTER TABLE message_assets ALTER COLUMN tenant_id SET NOT NULL")
    )
    connection.execute(
        text("ALTER TABLE message_vectors ALTER COLUMN tenant_id SET NOT NULL")
    )
    connection.execute(
        text("ALTER TABLE message_events ALTER COLUMN tenant_id SET NOT NULL")
    )

    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_conversations_tenant_created_at"
            " ON conversations (tenant_id, created_at)"
        )
    )
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_messages_tenant_conversation_created_at"
            " ON messages (tenant_id, conversation_id, created_at)"
        )
    )
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_message_assets_tenant_created_at"
            " ON message_assets (tenant_id, created_at)"
        )
    )
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_message_vectors_tenant_created_at"
            " ON message_vectors (tenant_id, created_at)"
        )
    )
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_message_events_tenant_created_at"
            " ON message_events (tenant_id, created_at)"
        )
    )


def downgrade(connection: Connection) -> None:
    """Drop the conversation store schema."""

    metadata.drop_all(connection)
