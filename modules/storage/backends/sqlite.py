"""SQLite backend adapter for StorageManager."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker

logger = setup_logger(__name__)


class SQLiteBackend:
    """SQLite-specific backend operations.

    Handles SQLite-specific file management, pragmas, and schema operations.
    """

    def __init__(
        self,
        engine: "Engine",
        session_factory: "sessionmaker",
        db_path: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._db_path = db_path

    @property
    def engine(self) -> "Engine":
        """Get the underlying SQLAlchemy engine."""
        return self._engine

    @property
    def session_factory(self) -> "sessionmaker":
        """Get the session factory."""
        return self._session_factory

    @property
    def db_path(self) -> Optional[str]:
        """Get the database file path (None for in-memory)."""
        return self._db_path

    @property
    def is_memory(self) -> bool:
        """Check if this is an in-memory database."""
        return self._db_path is None or self._db_path == ":memory:"

    async def ensure_directory(self) -> bool:
        """Ensure the database directory exists."""
        if self.is_memory:
            return True

        db_path = self._db_path
        if db_path is None:
            return True

        def _ensure() -> bool:
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return True

        return await asyncio.to_thread(_ensure)

    async def configure_pragmas(
        self,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
        cache_size: int = -64000,  # 64MB
        foreign_keys: bool = True,
    ) -> None:
        """Configure SQLite pragmas for optimal performance."""

        def _configure() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text(f"PRAGMA journal_mode={journal_mode}"))
                conn.execute(text(f"PRAGMA synchronous={synchronous}"))
                conn.execute(text(f"PRAGMA cache_size={cache_size}"))
                conn.execute(text(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}"))
                conn.commit()

            logger.debug(
                "SQLite pragmas configured",
                extra={
                    "journal_mode": journal_mode,
                    "synchronous": synchronous,
                    "cache_size": cache_size,
                    "foreign_keys": foreign_keys,
                },
            )

        await asyncio.to_thread(_configure)

    async def get_table_names(self) -> Set[str]:
        """Get all table names in the database."""

        def _get_tables() -> Set[str]:
            from sqlalchemy import inspect

            inspector = inspect(self._engine)
            return set(inspector.get_table_names())

        return await asyncio.to_thread(_get_tables)

    async def verify_tables(self, required: Set[str]) -> Dict[str, bool]:
        """Verify that required tables exist."""
        existing = await self.get_table_names()
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

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim space and defragment the database."""
        if self.is_memory:
            return

        def _vacuum() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("VACUUM"))
                conn.commit()

            logger.info("SQLite VACUUM completed", extra={"path": self._db_path})

        await asyncio.to_thread(_vacuum)

    async def get_database_size(self) -> int:
        """Get the database file size in bytes."""
        if self.is_memory:
            return 0

        db_path = self._db_path
        if db_path is None:
            return 0

        def _get_size() -> int:
            try:
                return os.path.getsize(db_path)
            except OSError:
                return 0

        return await asyncio.to_thread(_get_size)

    async def get_page_count(self) -> int:
        """Get the number of pages in the database."""

        def _get_count() -> int:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(text("PRAGMA page_count"))
                row = result.fetchone()
                return int(row[0]) if row else 0

        return await asyncio.to_thread(_get_count)

    async def checkpoint(self) -> None:
        """Checkpoint the WAL file (if using WAL mode)."""

        def _checkpoint() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                conn.commit()

        await asyncio.to_thread(_checkpoint)


__all__ = ["SQLiteBackend"]
