"""PostgreSQL backend adapter for StorageManager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker

logger = setup_logger(__name__)


class PostgreSQLBackend:
    """PostgreSQL-specific backend operations.

    Handles PostgreSQL-specific schema management, extensions,
    and connection verification.
    """

    def __init__(self, engine: "Engine", session_factory: "sessionmaker") -> None:
        self._engine = engine
        self._session_factory = session_factory

    @property
    def engine(self) -> "Engine":
        """Get the underlying SQLAlchemy engine."""
        return self._engine

    @property
    def session_factory(self) -> "sessionmaker":
        """Get the session factory."""
        return self._session_factory

    async def ensure_extensions(self, extensions: Optional[Set[str]] = None) -> Dict[str, bool]:
        """Ensure required PostgreSQL extensions are installed.

        Args:
            extensions: Set of extension names to ensure. Defaults to common ones.

        Returns:
            Dict mapping extension name to whether it was successfully enabled.
        """
        if extensions is None:
            extensions = {"uuid-ossp", "pgcrypto"}

        results: Dict[str, bool] = {}

        def _ensure() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                for ext in extensions:
                    try:
                        conn.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                        conn.commit()
                        results[ext] = True
                        logger.debug(f"Extension {ext} ensured")
                    except Exception as exc:
                        logger.warning(f"Failed to ensure extension {ext}: {exc}")
                        results[ext] = False

        await asyncio.to_thread(_ensure)
        return results

    async def ensure_pgvector(self) -> bool:
        """Ensure the pgvector extension is installed."""
        results = await self.ensure_extensions({"vector"})
        return results.get("vector", False)

    async def get_table_names(self, schema: str = "public") -> Set[str]:
        """Get all table names in the specified schema."""

        def _get_tables() -> Set[str]:
            from sqlalchemy import inspect

            inspector = inspect(self._engine)
            return set(inspector.get_table_names(schema=schema))

        return await asyncio.to_thread(_get_tables)

    async def verify_tables(
        self, required: Set[str], schema: str = "public"
    ) -> Dict[str, bool]:
        """Verify that required tables exist.

        Returns:
            Dict mapping table name to whether it exists.
        """
        existing = await self.get_table_names(schema)
        return {table: table in existing for table in required}

    async def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL asynchronously."""

        def _execute() -> Any:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                conn.commit()
                return result.fetchall() if result.returns_rows else None

        return await asyncio.to_thread(_execute)

    async def get_database_size(self) -> int:
        """Get the current database size in bytes."""

        def _get_size() -> int:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text("SELECT pg_database_size(current_database())")
                )
                row = result.fetchone()
                return int(row[0]) if row else 0

        return await asyncio.to_thread(_get_size)

    async def get_connection_count(self) -> int:
        """Get the number of active connections to the database."""

        def _get_count() -> int:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database()"
                    )
                )
                row = result.fetchone()
                return int(row[0]) if row else 0

        return await asyncio.to_thread(_get_count)


__all__ = ["PostgreSQLBackend"]
