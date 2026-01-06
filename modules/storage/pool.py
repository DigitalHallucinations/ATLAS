"""Connection pool wrappers with async lifecycle management."""

from __future__ import annotations

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

from .settings import SQLBackendSettings, MongoBackendSettings, SQLDialect

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, sessionmaker

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# SQL Connection Pool
# ---------------------------------------------------------------------------


@dataclass
class SQLPool:
    """Async-aware wrapper around SQLAlchemy engine and session factory.

    Manages engine lifecycle with proper async initialization and shutdown.
    """

    settings: SQLBackendSettings
    _engine: Optional["Engine"] = field(default=None, init=False, repr=False)
    _session_factory: Optional["sessionmaker"] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize the connection pool asynchronously."""
        async with self._lock:
            if self._initialized:
                return

            await asyncio.to_thread(self._create_engine)
            self._initialized = True
            logger.info(
                "SQL pool initialized",
                extra={"dialect": self.settings.dialect.value, "event": "pool_init"},
            )

    def _create_engine(self) -> None:
        """Create SQLAlchemy engine synchronously (runs in thread)."""
        if importlib.util.find_spec("sqlalchemy") is None:
            raise ImportError(
                "SQLAlchemy is required. Install via `pip install SQLAlchemy`."
            )

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        url = self.settings.get_url()
        engine_kwargs: Dict[str, Any] = {"future": True}

        if self.settings.dialect == SQLDialect.POSTGRESQL:
            engine_kwargs.update(self.settings.pool.to_engine_kwargs())
        else:
            # SQLite doesn't support connection pooling the same way
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self._engine = create_engine(url, **engine_kwargs)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    async def shutdown(self) -> None:
        """Dispose of the engine and clean up connections."""
        async with self._lock:
            if not self._initialized or self._engine is None:
                return

            await asyncio.to_thread(self._engine.dispose)
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info(
                "SQL pool shutdown complete",
                extra={"dialect": self.settings.dialect.value, "event": "pool_shutdown"},
            )

    @property
    def engine(self) -> "Engine":
        """Get the SQLAlchemy engine. Raises if not initialized."""
        if self._engine is None:
            raise RuntimeError("SQL pool not initialized. Call initialize() first.")
        return self._engine

    @property
    def session_factory(self) -> "sessionmaker":
        """Get the session factory. Raises if not initialized."""
        if self._session_factory is None:
            raise RuntimeError("SQL pool not initialized. Call initialize() first.")
        return self._session_factory

    @property
    def is_initialized(self) -> bool:
        """Check if the pool has been initialized."""
        return self._initialized

    @asynccontextmanager
    async def session(self) -> AsyncIterator["Session"]:
        """Provide a transactional scope around a series of operations.

        Usage::

            async with pool.session() as session:
                session.add(entity)
                # commits on exit, rolls back on exception
        """
        if not self._initialized or self._session_factory is None:
            raise RuntimeError("SQL pool not initialized. Call initialize() first.")

        factory = self._session_factory
        session = await asyncio.to_thread(factory)
        try:
            yield session
            await asyncio.to_thread(session.commit)
        except Exception:
            await asyncio.to_thread(session.rollback)
            raise
        finally:
            await asyncio.to_thread(session.close)

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the database connection is healthy."""
        if not self._initialized or self._engine is None:
            return False

        engine = self._engine

        try:

            def _check() -> bool:
                from sqlalchemy import text

                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True

            return await asyncio.wait_for(
                asyncio.to_thread(_check),
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "SQL health check failed",
                extra={"error": str(exc), "event": "health_check_failed"},
            )
            return False


# ---------------------------------------------------------------------------
# MongoDB Connection Pool
# ---------------------------------------------------------------------------


@dataclass
class MongoPool:
    """Async-aware wrapper around PyMongo MongoClient.

    Manages client lifecycle with proper async initialization and shutdown.
    """

    settings: MongoBackendSettings
    _client: Optional[Any] = field(default=None, init=False, repr=False)
    _database: Optional[Any] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize the MongoDB client asynchronously."""
        async with self._lock:
            if self._initialized:
                return

            await asyncio.to_thread(self._create_client)
            self._initialized = True
            logger.info(
                "MongoDB pool initialized",
                extra={"database": self.settings.database, "event": "pool_init"},
            )

    def _create_client(self) -> None:
        """Create PyMongo client synchronously (runs in thread)."""
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError(
                "PyMongo is required for MongoDB support. "
                "Install via `pip install pymongo`."
            ) from exc

        url = self.settings.get_url()
        client_kwargs = self.settings.get_client_kwargs()

        client = MongoClient(url, **client_kwargs)
        self._client = client
        self._database = client[self.settings.database]

    async def shutdown(self) -> None:
        """Close the MongoDB client and clean up connections."""
        async with self._lock:
            if not self._initialized or self._client is None:
                return

            await asyncio.to_thread(self._client.close)
            self._client = None
            self._database = None
            self._initialized = False
            logger.info(
                "MongoDB pool shutdown complete",
                extra={"database": self.settings.database, "event": "pool_shutdown"},
            )

    @property
    def client(self) -> Any:
        """Get the MongoClient. Raises if not initialized."""
        if self._client is None:
            raise RuntimeError("MongoDB pool not initialized. Call initialize() first.")
        return self._client

    @property
    def database(self) -> Any:
        """Get the database instance. Raises if not initialized."""
        if self._database is None:
            raise RuntimeError("MongoDB pool not initialized. Call initialize() first.")
        return self._database

    @property
    def is_initialized(self) -> bool:
        """Check if the pool has been initialized."""
        return self._initialized

    def get_collection(self, name: str) -> Any:
        """Get a collection from the database."""
        return self.database[name]

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the MongoDB connection is healthy."""
        if not self._initialized or self._client is None:
            return False

        client = self._client

        try:

            def _check() -> bool:
                client.admin.command("ping")
                return True

            return await asyncio.wait_for(
                asyncio.to_thread(_check),
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "MongoDB health check failed",
                extra={"error": str(exc), "event": "health_check_failed"},
            )
            return False


__all__ = [
    "SQLPool",
    "MongoPool",
]
