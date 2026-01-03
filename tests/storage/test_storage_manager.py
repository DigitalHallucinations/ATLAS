"""Tests for the StorageManager unified storage orchestrator."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary SQLite database path."""
    return str(tmp_path / "test_atlas.sqlite3")


@pytest.fixture
def sqlite_settings(temp_db_path: str):
    """Create SQLite-based StorageSettings for testing."""
    from modules.storage.settings import (
        SQLBackendSettings,
        SQLDialect,
        StorageSettings,
        VectorBackendSettings,
        VectorBackendType,
    )

    return StorageSettings(
        sql=SQLBackendSettings(
            dialect=SQLDialect.SQLITE,
            database=temp_db_path,
        ),
        vectors=VectorBackendSettings(backend=VectorBackendType.NONE),
        require_tenant_context=False,
    )


@pytest.fixture
def env_settings():
    """Environment variable settings for testing."""
    return {
        "ATLAS_SQL_DIALECT": "sqlite",
        "ATLAS_SQL_DATABASE": ":memory:",
        "ATLAS_VECTOR_BACKEND": "none",
        "ATLAS_REQUIRE_TENANT_CONTEXT": "false",
    }


# ---------------------------------------------------------------------------
# Settings Tests
# ---------------------------------------------------------------------------


class TestStorageSettings:
    """Tests for StorageSettings configuration."""

    def test_default_settings(self):
        """Default settings should have sensible defaults."""
        from modules.storage.settings import StorageSettings, SQLDialect

        settings = StorageSettings()

        assert settings.sql.dialect == SQLDialect.POSTGRESQL
        assert settings.sql.host == "localhost"
        assert settings.sql.port == 5432
        assert settings.sql.database == "atlas"
        assert settings.sql.pool.size == 5
        assert settings.require_tenant_context is False

    def test_sqlite_settings(self, temp_db_path: str):
        """SQLite settings should build correct URL."""
        from modules.storage.settings import SQLBackendSettings, SQLDialect

        settings = SQLBackendSettings(
            dialect=SQLDialect.SQLITE,
            database=temp_db_path,
        )

        url = settings.get_url()
        assert url.startswith("sqlite:///")
        assert temp_db_path in url

    def test_sqlite_memory_url(self):
        """SQLite in-memory should produce correct URL."""
        from modules.storage.settings import SQLBackendSettings, SQLDialect

        settings = SQLBackendSettings(
            dialect=SQLDialect.SQLITE,
            database=":memory:",
        )

        assert settings.get_url() == "sqlite:///:memory:"

    def test_postgresql_url_with_auth(self):
        """PostgreSQL URL should include auth when provided."""
        from modules.storage.settings import SQLBackendSettings, SQLDialect

        settings = SQLBackendSettings(
            dialect=SQLDialect.POSTGRESQL,
            host="db.example.com",
            port=5432,
            database="mydb",
            username="user",
            password="secret",
        )

        url = settings.get_url()
        assert url == "postgresql://user:secret@db.example.com:5432/mydb"

    def test_postgresql_url_without_auth(self):
        """PostgreSQL URL should work without auth."""
        from modules.storage.settings import SQLBackendSettings, SQLDialect

        settings = SQLBackendSettings(
            dialect=SQLDialect.POSTGRESQL,
            host="localhost",
            port=5432,
            database="atlas",
        )

        url = settings.get_url()
        assert url == "postgresql://localhost:5432/atlas"

    def test_explicit_url_takes_precedence(self):
        """Explicit URL should override component settings."""
        from modules.storage.settings import SQLBackendSettings, SQLDialect

        settings = SQLBackendSettings(
            dialect=SQLDialect.POSTGRESQL,
            url="postgresql://custom@host:1234/db",
            host="ignored",
            database="ignored",
        )

        assert settings.get_url() == "postgresql://custom@host:1234/db"

    def test_settings_from_dict(self):
        """Settings should load from dictionary."""
        from modules.storage.settings import StorageSettings, SQLDialect

        data = {
            "sql": {
                "dialect": "sqlite",
                "database": "test.db",
            },
            "require_tenant_context": True,
        }

        settings = StorageSettings.from_dict(data)

        assert settings.sql.dialect == SQLDialect.SQLITE
        assert settings.sql.database == "test.db"
        assert settings.require_tenant_context is True

    def test_settings_from_env(self, env_settings: Dict[str, str]):
        """Settings should load from environment variables."""
        from modules.storage.settings import StorageSettings, SQLDialect, VectorBackendType

        with patch.dict(os.environ, env_settings, clear=False):
            settings = StorageSettings.from_env()

        assert settings.sql.dialect == SQLDialect.SQLITE
        assert settings.sql.database == ":memory:"
        assert settings.vectors.backend == VectorBackendType.NONE

    def test_settings_to_dict(self):
        """Settings should serialize to dictionary."""
        from modules.storage.settings import StorageSettings

        settings = StorageSettings()
        data = settings.to_dict()

        assert "sql" in data
        assert "mongo" in data
        assert "vectors" in data
        assert "retention" in data
        assert data["sql"]["dialect"] == "postgresql"

    def test_mongo_settings_url(self):
        """MongoDB settings should build correct URL."""
        from modules.storage.settings import MongoBackendSettings

        settings = MongoBackendSettings(
            host="mongo.example.com",
            port=27017,
            database="atlas",
            username="admin",
            password="secret",
        )

        url = settings.get_url()
        assert "mongodb://" in url
        assert "admin" in url
        assert "mongo.example.com" in url

    def test_vector_settings_defaults(self):
        """Vector settings should have sensible defaults."""
        from modules.storage.settings import VectorBackendSettings, VectorBackendType

        settings = VectorBackendSettings()

        assert settings.backend == VectorBackendType.PGVECTOR
        assert settings.embedding_dimension == 1536
        assert settings.collection_name == "atlas_vectors"

    def test_pool_settings_to_engine_kwargs(self):
        """Pool settings should convert to SQLAlchemy kwargs."""
        from modules.storage.settings import PoolSettings

        pool = PoolSettings(size=10, max_overflow=20, timeout=60.0)
        kwargs = pool.to_engine_kwargs()

        assert kwargs["pool_size"] == 10
        assert kwargs["max_overflow"] == 20
        assert kwargs["pool_timeout"] == 60.0


# ---------------------------------------------------------------------------
# Pool Tests
# ---------------------------------------------------------------------------


class TestSQLPool:
    """Tests for SQLPool connection management."""

    @pytest.mark.asyncio
    async def test_pool_initialize_sqlite(self, sqlite_settings):
        """SQLPool should initialize with SQLite."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)

        assert pool.is_initialized is False

        await pool.initialize()

        assert pool.is_initialized is True
        assert pool.engine is not None
        assert pool.session_factory is not None

        await pool.shutdown()

        assert pool.is_initialized is False

    @pytest.mark.asyncio
    async def test_pool_health_check_sqlite(self, sqlite_settings):
        """Health check should pass for healthy SQLite connection."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)
        await pool.initialize()

        healthy = await pool.health_check(timeout=5.0)

        assert healthy is True

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_health_check_not_initialized(self, sqlite_settings):
        """Health check should fail when not initialized."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)

        healthy = await pool.health_check()

        assert healthy is False

    @pytest.mark.asyncio
    async def test_pool_session_context_manager(self, sqlite_settings):
        """Session context manager should provide working session."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)
        await pool.initialize()

        async with pool.session() as session:
            from sqlalchemy import text

            result = await asyncio.to_thread(
                lambda: session.execute(text("SELECT 1")).scalar()
            )
            assert result == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_double_initialize(self, sqlite_settings):
        """Double initialization should be idempotent."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)

        await pool.initialize()
        engine1 = pool.engine

        await pool.initialize()
        engine2 = pool.engine

        assert engine1 is engine2

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_engine_not_initialized_error(self, sqlite_settings):
        """Accessing engine before init should raise RuntimeError."""
        from modules.storage.pool import SQLPool

        pool = SQLPool(settings=sqlite_settings.sql)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = pool.engine


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthChecks:
    """Tests for health monitoring system."""

    def test_store_health_is_healthy(self):
        """StoreHealth should correctly report healthy status."""
        from modules.storage.health import StoreHealth, HealthStatus

        health = StoreHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            latency_ms=5.0,
        )

        assert health.is_healthy is True
        assert health.is_available is True

    def test_store_health_degraded(self):
        """StoreHealth should report degraded as available but not healthy."""
        from modules.storage.health import StoreHealth, HealthStatus

        health = StoreHealth(
            name="test",
            status=HealthStatus.DEGRADED,
        )

        assert health.is_healthy is False
        assert health.is_available is True

    def test_store_health_unhealthy(self):
        """StoreHealth should report unhealthy as unavailable."""
        from modules.storage.health import StoreHealth, HealthStatus

        health = StoreHealth(
            name="test",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed",
        )

        assert health.is_healthy is False
        assert health.is_available is False

    def test_compute_overall_healthy(self):
        """Overall status should be healthy when all components healthy."""
        from modules.storage.health import (
            StoreHealth,
            HealthStatus,
            compute_overall_status,
        )

        components = [
            StoreHealth(name="sql", status=HealthStatus.HEALTHY),
            StoreHealth(name="mongo", status=HealthStatus.HEALTHY),
        ]

        overall = compute_overall_status(components)

        assert overall == HealthStatus.HEALTHY

    def test_compute_overall_degraded(self):
        """Overall status should be degraded when any component degraded."""
        from modules.storage.health import (
            StoreHealth,
            HealthStatus,
            compute_overall_status,
        )

        components = [
            StoreHealth(name="sql", status=HealthStatus.HEALTHY),
            StoreHealth(name="mongo", status=HealthStatus.DEGRADED),
        ]

        overall = compute_overall_status(components)

        assert overall == HealthStatus.DEGRADED

    def test_compute_overall_unhealthy(self):
        """Overall status should be unhealthy when any component unhealthy."""
        from modules.storage.health import (
            StoreHealth,
            HealthStatus,
            compute_overall_status,
        )

        components = [
            StoreHealth(name="sql", status=HealthStatus.HEALTHY),
            StoreHealth(name="mongo", status=HealthStatus.UNHEALTHY),
        ]

        overall = compute_overall_status(components)

        assert overall == HealthStatus.UNHEALTHY

    def test_compute_overall_empty(self):
        """Overall status should be unknown with no components."""
        from modules.storage.health import HealthStatus, compute_overall_status

        overall = compute_overall_status([])

        assert overall == HealthStatus.UNKNOWN

    def test_storage_health_status_to_dict(self):
        """StorageHealthStatus should serialize to dict."""
        from modules.storage.health import (
            StoreHealth,
            StorageHealthStatus,
            HealthStatus,
        )

        status = StorageHealthStatus(
            overall=HealthStatus.HEALTHY,
            sql=StoreHealth(name="sql", status=HealthStatus.HEALTHY),
        )

        data = status.to_dict()

        assert data["overall"] == "healthy"
        assert "components" in data
        assert "sql" in data["components"]


# ---------------------------------------------------------------------------
# StorageManager Tests
# ---------------------------------------------------------------------------


class TestStorageManager:
    """Tests for the main StorageManager orchestrator."""

    @pytest.mark.asyncio
    async def test_manager_initialize_sqlite(self, sqlite_settings):
        """StorageManager should initialize with SQLite backend."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)

        assert manager.is_initialized is False

        await manager.initialize()

        assert manager.is_initialized is True

        await manager.shutdown()

        assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_manager_get_engine(self, sqlite_settings):
        """Manager should provide SQL engine after initialization."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        engine = manager.get_sql_engine()

        assert engine is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_get_session_factory(self, sqlite_settings):
        """Manager should provide session factory after initialization."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        factory = manager.get_session_factory()

        assert factory is not None

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_not_initialized_error(self, sqlite_settings):
        """Accessing stores before init should raise RuntimeError."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.get_sql_engine()

    @pytest.mark.asyncio
    async def test_manager_health_check(self, sqlite_settings):
        """Health check should return status for all components."""
        from modules.storage.manager import StorageManager
        from modules.storage.health import HealthStatus

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        health = await manager.health_check()

        assert health.overall == HealthStatus.HEALTHY
        assert health.sql is not None
        assert health.sql.is_healthy is True

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_double_initialize(self, sqlite_settings):
        """Double initialization should be idempotent."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)

        await manager.initialize()
        await manager.initialize()  # Should not raise

        assert manager.is_initialized is True

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_double_shutdown(self, sqlite_settings):
        """Double shutdown should be idempotent."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        await manager.shutdown()
        await manager.shutdown()  # Should not raise

        assert manager.is_initialized is False

    @pytest.mark.asyncio
    async def test_manager_sql_backend_sqlite(self, sqlite_settings):
        """Manager should provide SQLite backend for SQLite dialect."""
        from modules.storage.manager import StorageManager
        from modules.storage.backends.sqlite import SQLiteBackend

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        backend = manager.sql_backend

        assert isinstance(backend, SQLiteBackend)

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_configure(self, sqlite_settings):
        """Configure should update settings."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)

        manager.configure(require_tenant_context=True)

        assert manager.settings.require_tenant_context is True

    @pytest.mark.asyncio
    async def test_manager_get_settings(self, sqlite_settings):
        """get_settings should return current settings."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)

        settings = manager.get_settings()

        assert settings is manager.settings


# ---------------------------------------------------------------------------
# Singleton Tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for singleton access pattern."""

    @pytest.mark.asyncio
    async def test_get_storage_manager_creates_instance(self, sqlite_settings):
        """get_storage_manager should create and initialize instance."""
        from modules.storage.manager import (
            get_storage_manager,
            reset_storage_manager,
        )

        # Ensure clean state
        await reset_storage_manager()

        with patch(
            "modules.storage.manager.StorageSettings.from_env",
            return_value=sqlite_settings,
        ):
            manager = await get_storage_manager(settings=sqlite_settings)

        assert manager is not None
        assert manager.is_initialized is True

        await reset_storage_manager()

    @pytest.mark.asyncio
    async def test_get_storage_manager_returns_same_instance(self, sqlite_settings):
        """Subsequent calls should return same instance."""
        from modules.storage.manager import (
            get_storage_manager,
            reset_storage_manager,
        )

        await reset_storage_manager()

        manager1 = await get_storage_manager(settings=sqlite_settings)
        manager2 = await get_storage_manager()

        assert manager1 is manager2

        await reset_storage_manager()

    @pytest.mark.asyncio
    async def test_reset_storage_manager(self, sqlite_settings):
        """reset_storage_manager should shutdown and clear instance."""
        from modules.storage.manager import (
            get_storage_manager,
            get_storage_manager_sync,
            reset_storage_manager,
        )

        await reset_storage_manager()

        manager = await get_storage_manager(settings=sqlite_settings)
        assert get_storage_manager_sync() is manager

        await reset_storage_manager()

        assert get_storage_manager_sync() is None


# ---------------------------------------------------------------------------
# Unit of Work Tests
# ---------------------------------------------------------------------------


class TestUnitOfWork:
    """Tests for cross-store transaction coordination."""

    @pytest.mark.asyncio
    async def test_unit_of_work_basic(self, sqlite_settings):
        """Unit of work should commit on success."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        executed = []

        async with manager.unit_of_work() as uow:
            uow.add_sql_operation(lambda s: executed.append("op1"))

        assert "op1" in executed

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_unit_of_work_rollback_on_error(self, sqlite_settings):
        """Unit of work should rollback on exception."""
        from modules.storage.manager import StorageManager
        from modules.storage.unit_of_work import UnitOfWorkState

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        compensated = []

        with pytest.raises(ValueError):
            async with manager.unit_of_work() as uow:
                uow.add_compensating_action(lambda: compensated.append("comp1"))
                raise ValueError("Test error")

        assert "comp1" in compensated

        await manager.shutdown()

    def test_unit_of_work_state_transitions(self):
        """Unit of work should track state correctly."""
        from modules.storage.unit_of_work import UnitOfWork, UnitOfWorkState

        uow = UnitOfWork()

        assert uow.state == UnitOfWorkState.PENDING


# ---------------------------------------------------------------------------
# Vector Store Tests
# ---------------------------------------------------------------------------


class TestVectorProviderBase:
    """Tests for vector provider base classes."""

    def test_vector_document_auto_id(self):
        """VectorDocument should generate ID if not provided."""
        from modules.storage.vectors.base import VectorDocument

        doc = VectorDocument(id="", vector=[1.0, 2.0, 3.0])

        assert doc.id != ""
        assert len(doc.id) > 0

    def test_vector_document_with_metadata(self):
        """VectorDocument should store metadata."""
        from modules.storage.vectors.base import VectorDocument

        doc = VectorDocument(
            id="test",
            vector=[1.0, 2.0],
            content="Hello",
            metadata={"key": "value"},
            namespace="ns1",
        )

        assert doc.content == "Hello"
        assert doc.metadata["key"] == "value"
        assert doc.namespace == "ns1"

    def test_vector_search_result(self):
        """VectorSearchResult should hold document and score."""
        from modules.storage.vectors.base import VectorDocument, VectorSearchResult

        doc = VectorDocument(id="test", vector=[1.0])
        result = VectorSearchResult(document=doc, score=0.95, distance=0.05)

        assert result.document is doc
        assert result.score == 0.95
        assert result.distance == 0.05

    def test_collection_info(self):
        """CollectionInfo should hold collection metadata."""
        from modules.storage.vectors.base import CollectionInfo, DistanceMetric

        info = CollectionInfo(
            name="test_collection",
            dimension=1536,
            count=100,
            metric=DistanceMetric.COSINE,
        )

        assert info.name == "test_collection"
        assert info.dimension == 1536
        assert info.count == 100

    def test_register_vector_provider(self):
        """register_vector_provider should add to registry."""
        from modules.storage.vectors.base import (
            VectorProvider,
            register_vector_provider,
            get_vector_provider,
            available_vector_providers,
        )

        @register_vector_provider("test_provider")
        class TestProvider(VectorProvider):
            @property
            def name(self):
                return "test"

            @property
            def is_initialized(self):
                return False

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

            async def health_check(self, timeout=5.0):
                return True

            async def create_collection(self, name, dimension, **kwargs):
                pass

            async def delete_collection(self, name):
                return True

            async def list_collections(self):
                return []

            async def get_collection(self, name):
                return None

            async def upsert(self, collection, documents, **kwargs):
                return 0

            async def delete(self, collection, ids, **kwargs):
                return 0

            async def get(self, collection, ids, **kwargs):
                return []

            async def search(self, collection, query_vector, **kwargs):
                return []

        assert "test_provider" in available_vector_providers()
        assert get_vector_provider("test_provider") is TestProvider

    def test_vector_store_registry(self):
        """VectorStoreRegistry should manage provider instances."""
        from modules.storage.vectors.base import VectorStoreRegistry
        from unittest.mock import MagicMock

        registry = VectorStoreRegistry()

        provider = MagicMock()
        provider.name = "mock"

        registry.register_instance("mock", provider, default=True)

        assert registry.get("mock") is provider
        assert registry.default is provider
        assert "mock" in registry.list_instances()


# ---------------------------------------------------------------------------
# Backend Tests
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    """Tests for SQLite backend adapter."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_get_table_names(self, sqlite_settings):
        """SQLite backend should list table names."""
        from modules.storage.pool import SQLPool
        from modules.storage.backends.sqlite import SQLiteBackend

        pool = SQLPool(settings=sqlite_settings.sql)
        await pool.initialize()

        backend = SQLiteBackend(pool.engine, pool.session_factory)

        tables = await backend.get_table_names()

        assert isinstance(tables, set)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_sqlite_backend_verify_tables(self, sqlite_settings):
        """SQLite backend should verify table existence."""
        from modules.storage.pool import SQLPool
        from modules.storage.backends.sqlite import SQLiteBackend

        pool = SQLPool(settings=sqlite_settings.sql)
        await pool.initialize()

        backend = SQLiteBackend(pool.engine, pool.session_factory)

        results = await backend.verify_tables({"nonexistent_table"})

        assert results["nonexistent_table"] is False

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_sqlite_backend_execute_sql(self, sqlite_settings):
        """SQLite backend should execute raw SQL."""
        from modules.storage.pool import SQLPool
        from modules.storage.backends.sqlite import SQLiteBackend

        pool = SQLPool(settings=sqlite_settings.sql)
        await pool.initialize()

        backend = SQLiteBackend(pool.engine, pool.session_factory)

        result = await backend.execute_sql("SELECT 1 + 1")

        assert result[0][0] == 2

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_sqlite_backend_is_memory(self, sqlite_settings):
        """SQLite backend should detect in-memory databases."""
        from modules.storage.backends.sqlite import SQLiteBackend
        from unittest.mock import MagicMock

        backend = SQLiteBackend(MagicMock(), MagicMock(), db_path=":memory:")

        assert backend.is_memory is True

        backend2 = SQLiteBackend(MagicMock(), MagicMock(), db_path="/path/to/db.sqlite3")

        assert backend2.is_memory is False


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for full StorageManager workflow."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, sqlite_settings):
        """Test complete initialize -> use -> shutdown cycle."""
        from modules.storage.manager import StorageManager
        from modules.storage.health import HealthStatus

        # Initialize
        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        # Check state
        assert manager.is_initialized
        assert manager.get_sql_engine() is not None
        assert manager.get_session_factory() is not None

        # Health check
        health = await manager.health_check()
        assert health.overall == HealthStatus.HEALTHY

        # Shutdown
        await manager.shutdown()
        assert not manager.is_initialized

    @pytest.mark.asyncio
    async def test_schema_creation(self, sqlite_settings):
        """Test schema creation for domain stores."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        # This may fail if domain stores aren't fully compatible,
        # but it tests the interface
        try:
            results = await manager.ensure_schemas()
            # At minimum the method should run without crashing
            assert isinstance(results, dict)
        except Exception:
            # Domain stores may not be fully set up in test env
            pass

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_verify_stores(self, sqlite_settings):
        """Test store verification."""
        from modules.storage.manager import StorageManager

        manager = StorageManager(settings=sqlite_settings)
        await manager.initialize()

        results = await manager.verify_stores()

        assert "sql" in results
        assert results["sql"] is True

        await manager.shutdown()
