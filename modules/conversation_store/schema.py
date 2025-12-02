"""Schema bootstrap helpers for the conversation store."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Iterable

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


def _parse_version(version: str | None) -> tuple[int, ...] | None:
    """Parse a PostgreSQL extension version string into numeric components."""

    if not version:
        return None

    parts: list[int] = []
    for piece in re.split(r"[^0-9]+", str(version)):
        if not piece:
            continue
        try:
            parts.append(int(piece))
        except ValueError:
            continue
    return tuple(parts) if parts else None


def _supports_hnsw(version: str | None) -> bool:
    parsed = _parse_version(version)
    if parsed is None:
        return False
    return parsed >= (0, 5, 0)


def _ensure_pgvector_extension(engine: Engine, *, logger=None) -> tuple[bool, str | None]:
    """Ensure the pgvector extension is available and return its version."""

    dialect_name = getattr(engine.dialect, "name", "")
    if dialect_name != "postgresql":
        return False, None

    try:
        with engine.begin() as connection:
            version = connection.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            if version is None:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                version = connection.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                ).scalar_one_or_none()
    except Exception as exc:
        if logger:
            logger.warning("Unable to ensure pgvector extension is installed: %s", exc)
        return False, None

    if logger:
        if version:
            logger.info("pgvector extension detected (version %s)", version)
        else:
            logger.warning("pgvector extension could not be detected even after install attempt")

    return version is not None, str(version) if version is not None else None


def _ensure_hnsw_indexes(
    engine: Engine,
    *,
    logger=None,
    opclasses: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Create HNSW indexes for embedding columns when supported."""

    dialect_name = getattr(engine.dialect, "name", "")
    if dialect_name != "postgresql":
        return

    inspector = inspect(engine)
    vector_columns = {column["name"] for column in inspector.get_columns("message_vectors")}
    if "embedding" not in vector_columns:
        return

    operations = opclasses or (
        ("ix_message_vectors_embedding_hnsw_cosine", "vector_cosine_ops"),
        ("ix_message_vectors_embedding_hnsw_l2", "vector_l2_ops"),
    )

    with engine.begin() as connection:
        for index_name, opclass in operations:
            try:
                connection.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON message_vectors USING hnsw (embedding {opclass})"
                    )
                )
            except Exception as exc:
                if logger:
                    logger.warning(
                        "Skipping creation of HNSW index %s due to error: %s", index_name, exc
                    )
                continue
            if logger:
                logger.info("Ensured HNSW index '%s' with opclass %s", index_name, opclass)


def create_schema(session_factory: sessionmaker, *, logger=None) -> None:
    """Create conversation store tables if they do not already exist."""

    from modules.job_store import ensure_job_schema
    from modules.task_store import ensure_task_schema

    engine = resolve_engine(session_factory)
    dialect_name = getattr(engine.dialect, "name", "")

    has_pgvector, pgvector_version = _ensure_pgvector_extension(engine, logger=logger)

    Base.metadata.create_all(engine)
    ensure_task_schema(engine)
    ensure_job_schema(engine)

    inspector = inspect(engine)

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

    if dialect_name == "postgresql" and has_pgvector:
        if _supports_hnsw(pgvector_version):
            _ensure_hnsw_indexes(engine, logger=logger)
        elif logger:
            logger.warning(
                "pgvector version %s does not support HNSW indexes; upgrade to >= 0.5.0 to enable acceleration",
                pgvector_version or "unknown",
            )


__all__ = [
    "create_conversation_engine",
    "create_schema",
    "resolve_engine",
]
