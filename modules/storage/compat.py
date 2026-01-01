"""Configuration conversion utilities for StorageManager.

This module provides utilities for converting between configuration formats:

1. StorageArchitecture <-> StorageSettings (for config system interop)
2. DatabaseState <-> StorageSettings (for setup wizard)

These are not legacy compatibility layers - they are format converters
that enable different parts of the system to work with StorageManager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ATLAS.config.storage import StorageArchitecture, PerformanceMode
    from ATLAS.setup.controller import DatabaseState
    from modules.storage.manager import StorageManager
    from modules.storage.settings import StorageSettings


def storage_architecture_to_settings(
    architecture: "StorageArchitecture",
    *,
    database_url: Optional[str] = None,
    mongo_url: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> "StorageSettings":
    """
    Convert a StorageArchitecture to StorageSettings.

    Args:
        architecture: The legacy StorageArchitecture configuration.
        database_url: Optional explicit database URL override.
        mongo_url: Optional MongoDB URL if using MongoDB backend.
        extra: Additional settings to merge.

    Returns:
        StorageSettings instance.
    """
    from modules.storage.settings import (
        StorageSettings,
        SQLBackendSettings,
        SQLDialect,
        VectorBackendSettings,
        VectorBackendType,
    )

    # Map conversation backend to SQL dialect
    backend = architecture.conversation_backend.lower()
    if backend in ("sqlite", "sqlite3"):
        sql_dialect = SQLDialect.SQLITE
    elif backend in ("postgresql", "postgres", "pg"):
        sql_dialect = SQLDialect.POSTGRESQL
    else:
        # Default to PostgreSQL for unknown backends
        sql_dialect = SQLDialect.POSTGRESQL

    # Build SQL settings
    sql_settings = SQLBackendSettings(dialect=sql_dialect)
    if database_url:
        sql_settings = SQLBackendSettings(dialect=sql_dialect, url=database_url)

    # Map vector store adapter
    vector_adapter = architecture.vector_store_adapter.lower()
    if vector_adapter in ("pgvector", "pg_vector"):
        vector_backend = VectorBackendType.PGVECTOR
    elif vector_adapter in ("pinecone",):
        vector_backend = VectorBackendType.PINECONE
    elif vector_adapter in ("chroma", "chromadb"):
        vector_backend = VectorBackendType.CHROMA
    elif vector_adapter in ("none", "disabled"):
        vector_backend = VectorBackendType.NONE
    else:
        # in_memory, mongodb, etc. - not directly supported
        vector_backend = VectorBackendType.NONE

    vector_settings = VectorBackendSettings(backend=vector_backend)

    settings = StorageSettings(
        sql=sql_settings,
        vectors=vector_settings,
    )

    # Apply extras if provided
    if extra:
        import dataclasses
        current = dataclasses.asdict(settings)
        _deep_merge(current, dict(extra))
        settings = StorageSettings.from_dict(current)

    return settings


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override into base (modifies base in place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def storage_settings_to_architecture(
    settings: "StorageSettings",
) -> "StorageArchitecture":
    """
    Convert StorageSettings back to StorageArchitecture.

    Args:
        settings: The StorageSettings instance.

    Returns:
        StorageArchitecture for legacy compatibility.
    """
    from ATLAS.config.storage import StorageArchitecture, PerformanceMode
    from modules.storage.settings import SQLDialect, VectorBackendType

    # Map SQL dialect to conversation backend
    if settings.sql.dialect == SQLDialect.SQLITE:
        conversation_backend = "sqlite"
    else:
        conversation_backend = "postgresql"

    # Map vector backend
    if settings.vectors.backend == VectorBackendType.PGVECTOR:
        vector_adapter = "pgvector"
    elif settings.vectors.backend == VectorBackendType.PINECONE:
        vector_adapter = "pinecone"
    elif settings.vectors.backend == VectorBackendType.CHROMA:
        vector_adapter = "chroma"
    else:
        vector_adapter = "in_memory"

    return StorageArchitecture(
        performance_mode=PerformanceMode.BALANCED,
        conversation_backend=conversation_backend,
        kv_reuse_conversation_store=True,
        vector_store_adapter=vector_adapter,
    )


def create_storage_settings_from_config(
    config_manager: "ConfigManager",
) -> "StorageSettings":
    """
    Extract StorageSettings from a ConfigManager instance.

    This reads the existing configuration and converts it to
    the new StorageSettings format.

    Args:
        config_manager: The ConfigManager instance.

    Returns:
        StorageSettings derived from current configuration.
    """
    from modules.storage.settings import (
        StorageSettings,
        SQLBackendSettings,
        SQLDialect,
        RetentionSettings,
    )

    # Get the storage architecture
    architecture = config_manager.get_storage_architecture()

    # Get database URL if available
    database_url: str | None = None
    try:
        # Try to get the conversation store engine to extract URL
        getter = getattr(config_manager, "get_conversation_store_database_url", None)
        if callable(getter):
            result = getter()
            database_url = str(result) if result else None
        else:
            # Fallback - try to read from config
            env = getattr(config_manager, "env_config", {}) or {}
            result = env.get("DATABASE_URL") or env.get("CONVERSATION_STORE_URL")
            database_url = str(result) if result else None
    except Exception:
        pass

    # Convert architecture to settings
    settings = storage_architecture_to_settings(
        architecture,
        database_url=database_url,
    )

    # Apply retention settings
    try:
        retention = config_manager.get_conversation_retention_policies()
        days = retention.get("days")
        history_limit = retention.get("history_limit")
        settings.retention = RetentionSettings(
            conversation_days=int(days) if days is not None else 30,
            conversation_history_limit=int(history_limit) if history_limit is not None else 100,
        )
    except Exception:
        pass

    # Apply tenant context setting
    try:
        require_tenant = config_manager.config.get("require_tenant_context", False)
        settings.require_tenant_context = bool(require_tenant)
    except Exception:
        pass

    return settings


def database_state_to_settings(
    state: "DatabaseState",
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> "StorageSettings":
    """
    Convert setup wizard DatabaseState to StorageSettings.

    Args:
        state: The DatabaseState from the setup wizard.
        extra: Additional settings to merge.

    Returns:
        StorageSettings instance.
    """
    from modules.storage.settings import (
        StorageSettings,
        SQLBackendSettings,
        SQLDialect,
    )

    # Map backend to dialect
    backend = state.backend.lower()
    if backend in ("sqlite", "sqlite3"):
        sql_dialect = SQLDialect.SQLITE
        # For SQLite, the database field is the file path
        sql_settings = SQLBackendSettings(
            dialect=sql_dialect,
            database=state.database or ":memory:",
        )
    elif backend in ("postgresql", "postgres", "pg"):
        sql_dialect = SQLDialect.POSTGRESQL
        # Build from components or use DSN
        if state.dsn:
            sql_settings = SQLBackendSettings(
                dialect=sql_dialect,
                url=state.dsn,
            )
        else:
            sql_settings = SQLBackendSettings(
                dialect=sql_dialect,
                host=state.host or "localhost",
                port=state.port or 5432,
                database=state.database or "atlas",
                username=state.user,
                password=state.password,
            )
    else:
        # Unknown backend, default to PostgreSQL
        sql_dialect = SQLDialect.POSTGRESQL
        sql_settings = SQLBackendSettings(
            dialect=sql_dialect,
            host=state.host or "localhost",
            port=state.port or 5432,
            database=state.database or "atlas",
            username=state.user,
            password=state.password,
        )

    settings = StorageSettings(sql=sql_settings)

    # Apply extras if provided
    if extra:
        import dataclasses
        current = dataclasses.asdict(settings)
        _deep_merge(current, dict(extra))
        settings = StorageSettings.from_dict(current)

    return settings


def storage_settings_to_database_state(
    settings: "StorageSettings",
) -> "DatabaseState":
    """
    Convert StorageSettings back to DatabaseState for setup wizard.

    Args:
        settings: The StorageSettings instance.

    Returns:
        DatabaseState for setup wizard compatibility.
    """
    from ATLAS.setup.controller import DatabaseState
    from modules.storage.settings import SQLDialect

    # Map SQL dialect to backend
    if settings.sql.dialect == SQLDialect.SQLITE:
        backend = "sqlite"
    else:
        backend = "postgresql"

    # Build DSN if we have components
    dsn = ""
    if settings.sql.url:
        dsn = settings.sql.url
    else:
        dsn = settings.sql.get_url()

    return DatabaseState(
        backend=backend,
        host=settings.sql.host or "localhost",
        port=settings.sql.port or 5432,
        database=settings.sql.database or "atlas",
        user=settings.sql.username or "",
        password=settings.sql.password or "",
        dsn=dsn,
        options="",
    )
