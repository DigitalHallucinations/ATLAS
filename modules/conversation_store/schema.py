"""Schema bootstrap helpers for the conversation store."""

from __future__ import annotations

import contextlib
from typing import Any

from ._compat import Engine, create_engine, inspect, sessionmaker, text
from .models import Base


def create_conversation_engine(url: str, **engine_kwargs: Any) -> Engine:
    """Create a SQLAlchemy engine for the conversation store."""

    return create_engine(url, future=True, **engine_kwargs)


def resolve_engine(session_factory: sessionmaker) -> Engine:
    """Resolve the engine bound to ``session_factory``."""

    engine: Engine | None = getattr(session_factory, "bind", None)
    if engine is not None:
        return engine

    with contextlib.ExitStack() as stack:
        try:
            session = stack.enter_context(session_factory())
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError("Session factory is not bound to an engine") from exc
        engine = session.get_bind()

    if engine is None:
        raise RuntimeError("Session factory is not bound to an engine")

    return engine


def create_schema(session_factory: sessionmaker) -> None:
    """Create conversation store tables if they do not already exist."""

    from modules.job_store import ensure_job_schema
    from modules.task_store import ensure_task_schema

    engine = resolve_engine(session_factory)

    Base.metadata.create_all(engine)
    ensure_task_schema(engine)
    ensure_job_schema(engine)

    inspector = inspect(engine)
    dialect_name = getattr(engine.dialect, "name", "")

    with engine.begin() as connection:
        user_columns = {column["name"] for column in inspector.get_columns("users")}
        if "tenant_id" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN tenant_id VARCHAR(255)"))
        if dialect_name == "postgresql":
            connection.execute(text("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_external_id_key"))
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_users_tenant_id ON users (tenant_id)")
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_tenant_external ON users (tenant_id, external_id)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_external_single ON users (external_id) WHERE tenant_id IS NULL"
            )
        )

        credential_columns = {column["name"] for column in inspector.get_columns("user_credentials")}
        if "tenant_id" not in credential_columns:
            connection.execute(text("ALTER TABLE user_credentials ADD COLUMN tenant_id VARCHAR(255)"))
            connection.execute(
                text("UPDATE user_credentials SET tenant_id = NULL WHERE tenant_id IS NULL")
            )
        if dialect_name == "postgresql":
            connection.execute(
                text(
                    "ALTER TABLE user_credentials DROP CONSTRAINT IF EXISTS user_credentials_username_key"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE user_credentials DROP CONSTRAINT IF EXISTS user_credentials_email_key"
                )
            )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_user_credentials_tenant_id ON user_credentials (tenant_id)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_credentials_tenant_username ON user_credentials (tenant_id, username)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_credentials_tenant_email ON user_credentials (tenant_id, email)"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_credentials_username_single ON user_credentials (username) WHERE tenant_id IS NULL"
            )
        )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_credentials_email_single ON user_credentials (email) WHERE tenant_id IS NULL"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_user_credentials_username ON user_credentials (username)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_user_credentials_email ON user_credentials (email)")
        )

        if dialect_name == "postgresql":
            message_columns = {column["name"] for column in inspector.get_columns("messages")}
            if "message_text_tsv" not in message_columns:
                connection.execute(text("ALTER TABLE messages ADD COLUMN message_text_tsv tsvector"))
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


__all__ = [
    "create_conversation_engine",
    "create_schema",
    "resolve_engine",
]
