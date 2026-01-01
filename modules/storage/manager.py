"""Core StorageManager implementation.

Provides centralized orchestration of all ATLAS storage subsystems
with async lifecycle management, lazy store initialization, and
unified health monitoring.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    TYPE_CHECKING,
)

from modules.logging.logger import setup_logger

from .backends.mongodb import MongoDBBackend
from .backends.postgresql import PostgreSQLBackend
from .backends.sqlite import SQLiteBackend
from .health import (
    HealthStatus,
    StorageHealthStatus,
    StoreHealth,
    check_mongo_health,
    check_sql_health,
    check_vector_health,
    compute_overall_status,
)
from .pool import MongoPool, SQLPool
from .settings import SQLDialect, StorageSettings, VectorBackendType
from .unit_of_work import UnitOfWork, UnitOfWorkManager
from .vectors.base import VectorProvider, VectorStoreRegistry

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker

    from modules.conversation_store import ConversationStoreRepository
    from modules.job_store.repository import JobStoreRepository
    from modules.task_store import TaskStoreRepository

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Singleton Management
# ---------------------------------------------------------------------------

_storage_manager_instance: Optional["StorageManager"] = None
_storage_manager_lock = asyncio.Lock()


async def get_storage_manager(
    settings: Optional[StorageSettings] = None,
) -> "StorageManager":
    """Get the global StorageManager singleton.

    Creates and initializes the manager on first call. Subsequent calls
    return the same instance.

    Args:
        settings: Optional settings for first-time initialization.
                 Ignored if manager already exists.

    Returns:
        The initialized StorageManager instance.

    Usage::

        storage = await get_storage_manager()
        conversations = storage.conversations
    """
    global _storage_manager_instance

    if _storage_manager_instance is not None and _storage_manager_instance.is_initialized:
        return _storage_manager_instance

    async with _storage_manager_lock:
        if _storage_manager_instance is None:
            _storage_manager_instance = StorageManager(settings=settings)

        if not _storage_manager_instance.is_initialized:
            await _storage_manager_instance.initialize()

        return _storage_manager_instance


def get_storage_manager_sync() -> Optional["StorageManager"]:
    """Get the StorageManager if already initialized (non-async).

    Returns None if not yet initialized. Use get_storage_manager()
    for guaranteed initialization.

    Returns:
        The StorageManager instance or None.
    """
    return _storage_manager_instance


async def reset_storage_manager() -> None:
    """Reset the global StorageManager singleton.

    Shuts down the current manager and clears the instance.
    Primarily for testing.
    """
    global _storage_manager_instance

    async with _storage_manager_lock:
        if _storage_manager_instance is not None:
            await _storage_manager_instance.shutdown()
            _storage_manager_instance = None


# ---------------------------------------------------------------------------
# StorageManager
# ---------------------------------------------------------------------------


@dataclass
class StorageManager:
    """Unified orchestrator for all ATLAS storage subsystems.

    Manages lifecycle, configuration, health, and access to:
    - SQL databases (PostgreSQL, SQLite)
    - MongoDB document store
    - Key-Value stores
    - Vector stores (pgvector, Pinecone, Chroma)
    - Domain stores (conversations, tasks, jobs)

    Usage::

        # Create with settings
        storage = StorageManager(settings=my_settings)
        await storage.initialize()

        # Access stores
        conversations = storage.conversations
        tasks = storage.tasks

        # Health check
        health = await storage.health_check()

        # Graceful shutdown
        await storage.shutdown()
    """

    settings: StorageSettings = field(default_factory=StorageSettings)

    # Connection pools
    _sql_pool: Optional[SQLPool] = field(default=None, init=False, repr=False)
    _mongo_pool: Optional[MongoPool] = field(default=None, init=False, repr=False)

    # Backend adapters
    _sql_backend: Optional[PostgreSQLBackend | SQLiteBackend] = field(
        default=None, init=False, repr=False
    )
    _mongo_backend: Optional[MongoDBBackend] = field(default=None, init=False, repr=False)

    # Vector store
    _vector_registry: VectorStoreRegistry = field(
        default_factory=VectorStoreRegistry, init=False, repr=False
    )
    _vector_provider: Optional[VectorProvider] = field(default=None, init=False, repr=False)

    # Domain stores (lazy-initialized)
    _conversation_store: Optional[Any] = field(default=None, init=False, repr=False)
    _task_store: Optional[Any] = field(default=None, init=False, repr=False)
    _job_store: Optional[Any] = field(default=None, init=False, repr=False)
    _kv_store: Optional[Any] = field(default=None, init=False, repr=False)

    # Unit of work manager
    _uow_manager: Optional[UnitOfWorkManager] = field(default=None, init=False, repr=False)

    # State
    _initialized: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.settings, dict):
            self.settings = StorageSettings.from_dict(self.settings)

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Check if the storage manager has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize all storage subsystems.

        Sets up connection pools, initializes backends, and prepares
        for domain store access.
        """
        async with self._lock:
            if self._initialized:
                logger.debug("StorageManager already initialized")
                return

            logger.info("Initializing StorageManager...")

            # Initialize SQL pool
            await self._init_sql_pool()

            # Initialize MongoDB pool if configured
            if self.settings.use_mongo_for_conversations or self.settings.use_mongo_for_jobs:
                await self._init_mongo_pool()

            # Initialize vector store
            await self._init_vector_store()

            # Set up unit of work manager
            if self._sql_pool is not None:
                self._uow_manager = UnitOfWorkManager(
                    session_factory=self._sql_pool.session_factory
                )

            self._initialized = True
            logger.info("StorageManager initialized successfully")

    async def _init_sql_pool(self) -> None:
        """Initialize the SQL connection pool."""
        self._sql_pool = SQLPool(settings=self.settings.sql)
        await self._sql_pool.initialize()

        # Create appropriate backend adapter
        if self.settings.sql.dialect == SQLDialect.POSTGRESQL:
            self._sql_backend = PostgreSQLBackend(
                self._sql_pool.engine,
                self._sql_pool.session_factory,
            )
        else:
            db_path = None
            if self.settings.sql.url:
                # Extract path from sqlite:///path
                url = self.settings.sql.url
                if url.startswith("sqlite:///"):
                    db_path = url[10:]
            else:
                db_path = self.settings.sql.database

            self._sql_backend = SQLiteBackend(
                self._sql_pool.engine,
                self._sql_pool.session_factory,
                db_path=db_path,
            )
            # Configure SQLite pragmas for performance
            await self._sql_backend.configure_pragmas()

    async def _init_mongo_pool(self) -> None:
        """Initialize the MongoDB connection pool."""
        self._mongo_pool = MongoPool(settings=self.settings.mongo)
        await self._mongo_pool.initialize()

        self._mongo_backend = MongoDBBackend(
            self._mongo_pool.client,
            self._mongo_pool.database,
        )

    async def _init_vector_store(self) -> None:
        """Initialize the vector store provider."""
        backend = self.settings.vectors.backend

        if backend == VectorBackendType.NONE:
            logger.info("Vector store disabled")
            return

        if backend == VectorBackendType.PGVECTOR:
            if self._sql_pool is None or self.settings.sql.dialect != SQLDialect.POSTGRESQL:
                logger.warning("pgvector requires PostgreSQL; vector store not initialized")
                return

            from .vectors.pgvector import PgVectorProvider

            self._vector_provider = PgVectorProvider(
                self._sql_pool.engine,
                self._sql_pool.session_factory,
                index_type=self.settings.vectors.pgvector_index_type,
                ivfflat_lists=self.settings.vectors.pgvector_lists,
                hnsw_m=self.settings.vectors.pgvector_m,
                hnsw_ef_construction=self.settings.vectors.pgvector_ef_construction,
            )

        elif backend == VectorBackendType.PINECONE:
            if not self.settings.vectors.pinecone_api_key:
                logger.warning("Pinecone API key not configured; vector store not initialized")
                return

            from .vectors.pinecone import PineconeProvider

            self._vector_provider = PineconeProvider(
                api_key=self.settings.vectors.pinecone_api_key,
                environment=self.settings.vectors.pinecone_environment,
                index_name=self.settings.vectors.pinecone_index_name,
                namespace=self.settings.vectors.pinecone_namespace,
                metric=self.settings.vectors.pinecone_metric,
                dimension=self.settings.vectors.embedding_dimension,
            )

        elif backend == VectorBackendType.CHROMA:
            from .vectors.chroma import ChromaProvider

            self._vector_provider = ChromaProvider(
                host=self.settings.vectors.chroma_host,
                port=self.settings.vectors.chroma_port,
                persist_directory=self.settings.vectors.chroma_persist_directory,
                auth_token=self.settings.vectors.chroma_auth_token,
                tenant=self.settings.vectors.chroma_tenant,
                database=self.settings.vectors.chroma_database,
            )

        if self._vector_provider is not None:
            await self._vector_provider.initialize()
            self._vector_registry.register_instance(
                self._vector_provider.name,
                self._vector_provider,
                default=True,
            )
            logger.info(f"Vector store initialized: {self._vector_provider.name}")

    async def shutdown(self) -> None:
        """Gracefully shutdown all storage subsystems.

        Closes connections, disposes pools, and cleans up resources.
        """
        async with self._lock:
            if not self._initialized:
                return

            logger.info("Shutting down StorageManager...")

            # Shutdown vector stores
            await self._vector_registry.shutdown_all()
            self._vector_provider = None

            # Shutdown MongoDB pool
            if self._mongo_pool is not None:
                await self._mongo_pool.shutdown()
                self._mongo_pool = None
                self._mongo_backend = None

            # Shutdown SQL pool
            if self._sql_pool is not None:
                await self._sql_pool.shutdown()
                self._sql_pool = None
                self._sql_backend = None

            # Clear cached stores
            self._conversation_store = None
            self._task_store = None
            self._job_store = None
            self._kv_store = None
            self._uow_manager = None

            self._initialized = False
            logger.info("StorageManager shutdown complete")

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------

    def get_settings(self) -> StorageSettings:
        """Get the current storage settings."""
        return self.settings

    def configure(self, **overrides: Any) -> None:
        """Update storage settings.

        Note: Changes take effect on next initialization.
        Call shutdown() then initialize() to apply changes.

        Args:
            **overrides: Settings fields to update.
        """
        import dataclasses

        current = dataclasses.asdict(self.settings)
        current.update(overrides)
        self.settings = StorageSettings.from_dict(current)
        logger.info("Storage settings updated (restart required to apply)")

    # ---------------------------------------------------------------------------
    # Connection Access
    # ---------------------------------------------------------------------------

    def get_sql_engine(self) -> "Engine":
        """Get the SQLAlchemy engine.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._sql_pool is None:
            raise RuntimeError("StorageManager not initialized")
        return self._sql_pool.engine

    def get_session_factory(self) -> "sessionmaker":
        """Get the SQL session factory.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._sql_pool is None:
            raise RuntimeError("StorageManager not initialized")
        return self._sql_pool.session_factory

    def get_mongo_client(self) -> Any:
        """Get the MongoDB client.

        Raises:
            RuntimeError: If MongoDB not configured or not initialized.
        """
        if self._mongo_pool is None:
            raise RuntimeError("MongoDB not configured or not initialized")
        return self._mongo_pool.client

    def get_mongo_database(self) -> Any:
        """Get the MongoDB database.

        Raises:
            RuntimeError: If MongoDB not configured or not initialized.
        """
        if self._mongo_pool is None:
            raise RuntimeError("MongoDB not configured or not initialized")
        return self._mongo_pool.database

    # ---------------------------------------------------------------------------
    # Backend Access
    # ---------------------------------------------------------------------------

    @property
    def sql_backend(self) -> PostgreSQLBackend | SQLiteBackend:
        """Get the SQL backend adapter."""
        if self._sql_backend is None:
            raise RuntimeError("StorageManager not initialized")
        return self._sql_backend

    @property
    def mongo_backend(self) -> Optional[MongoDBBackend]:
        """Get the MongoDB backend adapter (if configured)."""
        return self._mongo_backend

    # ---------------------------------------------------------------------------
    # Domain Store Access (Lazy Initialization)
    # ---------------------------------------------------------------------------

    @property
    def conversations(self) -> "ConversationStoreRepository":
        """Get the conversation store repository.

        Lazily initializes on first access.
        """
        if self._conversation_store is None:
            self._conversation_store = self._create_conversation_store()
        return self._conversation_store

    def _create_conversation_store(self) -> Any:
        """Create the conversation store repository."""
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        if self.settings.use_mongo_for_conversations and self._mongo_pool is not None:
            from modules.conversation_store import MongoConversationStoreRepository

            return MongoConversationStoreRepository.from_database(
                self._mongo_pool.database,
                client=self._mongo_pool.client,
            )
        else:
            from modules.conversation_store import ConversationStoreRepository

            return ConversationStoreRepository(
                self._sql_pool.session_factory,
                retention={
                    "days": self.settings.retention.conversation_days,
                    "history_limit": self.settings.retention.conversation_history_limit,
                },
                require_tenant_context=self.settings.require_tenant_context,
            )

    @property
    def tasks(self) -> "TaskStoreRepository":
        """Get the task store repository.

        Lazily initializes on first access.
        """
        if self._task_store is None:
            self._task_store = self._create_task_store()
        return self._task_store

    def _create_task_store(self) -> Any:
        """Create the task store repository."""
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        from modules.task_store.repository import TaskStoreRepository

        return TaskStoreRepository(
            self._sql_pool.session_factory,
            require_tenant_context=self.settings.require_tenant_context,
        )

    @property
    def jobs(self) -> "JobStoreRepository":
        """Get the job store repository.

        Lazily initializes on first access.
        """
        if self._job_store is None:
            self._job_store = self._create_job_store()
        return self._job_store

    def _create_job_store(self) -> Any:
        """Create the job store repository."""
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        if self.settings.use_mongo_for_jobs and self._mongo_pool is not None:
            from modules.job_store.mongo_repository import MongoJobStoreRepository

            return MongoJobStoreRepository(
                self._mongo_pool.database,
                require_tenant_context=self.settings.require_tenant_context,
            )
        else:
            from modules.job_store.repository import JobStoreRepository

            return JobStoreRepository(
                self._sql_pool.session_factory,
                require_tenant_context=self.settings.require_tenant_context,
            )

    @property
    def kv(self) -> Any:
        """Get the key-value store service.

        Lazily initializes on first access.
        """
        if self._kv_store is None:
            self._kv_store = self._create_kv_store()
        return self._kv_store

    def _create_kv_store(self) -> Any:
        """Create the KV store service."""
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        from modules.Tools.Base_Tools.kv_store import KeyValueStoreService

        # Determine which adapter to use based on SQL dialect
        if self.settings.sql.dialect == SQLDialect.POSTGRESQL:
            adapter_name = "postgres"
        else:
            adapter_name = "sqlite"

        return KeyValueStoreService(
            adapter_name=adapter_name,
            engine=self._sql_pool.engine,
            namespace_quota_bytes=self.settings.kv.namespace_quota_bytes,
            global_quota_bytes=self.settings.kv.global_quota_bytes,
        )

    @property
    def vectors(self) -> Optional[VectorProvider]:
        """Get the vector store provider.

        Returns None if vector store is not configured.
        """
        return self._vector_provider

    @property
    def vector_registry(self) -> VectorStoreRegistry:
        """Get the vector store registry for managing multiple providers."""
        return self._vector_registry

    def get_domain_store_factory(
        self,
        *,
        default_retention: Optional[Dict[str, Any]] = None,
        require_tenant_context: Optional[bool] = None,
    ) -> "DomainStoreFactory":
        """Get a factory for creating domain stores.

        This is an alternative to the lazy properties (conversations, tasks, etc.)
        that provides more control over store configuration.

        Args:
            default_retention: Retention policy for applicable stores.
            require_tenant_context: Whether to require tenant context.

        Returns:
            DomainStoreFactory instance.
        """
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        from .adapters import DomainStoreFactory

        return DomainStoreFactory(
            self,
            default_retention=default_retention or {
                "days": self.settings.retention.conversation_days,
                "history_limit": self.settings.retention.conversation_history_limit,
            },
            require_tenant_context=(
                require_tenant_context
                if require_tenant_context is not None
                else self.settings.require_tenant_context
            ),
        )

    # ---------------------------------------------------------------------------
    # Schema Management
    # ---------------------------------------------------------------------------

    async def ensure_schemas(self) -> Dict[str, bool]:
        """Ensure all required database schemas exist.

        Creates tables and indexes for all domain stores.

        Returns:
            Dict mapping store name to success status.
        """
        if not self._initialized:
            raise RuntimeError("StorageManager not initialized")

        results: Dict[str, bool] = {}

        # Conversation store schema
        try:
            await asyncio.to_thread(self.conversations.create_schema)
            results["conversations"] = True
        except Exception as exc:
            logger.error(f"Failed to create conversation schema: {exc}")
            results["conversations"] = False

        # Task store schema
        try:
            from modules.task_store.models import ensure_task_schema

            await asyncio.to_thread(ensure_task_schema, self._sql_pool.engine)
            results["tasks"] = True
        except Exception as exc:
            logger.error(f"Failed to create task schema: {exc}")
            results["tasks"] = False

        # Job store schema
        try:
            from modules.job_store.models import ensure_job_schema

            await asyncio.to_thread(ensure_job_schema, self._sql_pool.engine)
            results["jobs"] = True
        except Exception as exc:
            logger.error(f"Failed to create job schema: {exc}")
            results["jobs"] = False

        # Vector store (pgvector extension)
        if self._vector_provider is not None and self.settings.vectors.backend == VectorBackendType.PGVECTOR:
            try:
                if isinstance(self._sql_backend, PostgreSQLBackend):
                    await self._sql_backend.ensure_pgvector()
                results["vectors"] = True
            except Exception as exc:
                logger.error(f"Failed to ensure pgvector extension: {exc}")
                results["vectors"] = False

        return results

    async def verify_stores(self) -> Dict[str, bool]:
        """Verify that all stores are accessible and functional.

        Returns:
            Dict mapping store name to verification status.
        """
        results: Dict[str, bool] = {}

        # Verify SQL connection
        if self._sql_pool is not None:
            results["sql"] = await self._sql_pool.health_check()

        # Verify MongoDB connection
        if self._mongo_pool is not None:
            results["mongo"] = await self._mongo_pool.health_check()

        # Verify vector store
        if self._vector_provider is not None:
            results["vectors"] = await self._vector_provider.health_check()

        return results

    # ---------------------------------------------------------------------------
    # Health Monitoring
    # ---------------------------------------------------------------------------

    async def health_check(self) -> StorageHealthStatus:
        """Perform comprehensive health check of all storage subsystems.

        Returns:
            Aggregate health status with per-component details.
        """
        timeout = self.settings.health_check_timeout_seconds
        components: list[StoreHealth] = []

        # SQL health
        sql_health: Optional[StoreHealth] = None
        if self._sql_pool is not None:
            sql_health = await check_sql_health(self._sql_pool, timeout=timeout)
            components.append(sql_health)

        # MongoDB health
        mongo_health: Optional[StoreHealth] = None
        if self._mongo_pool is not None:
            mongo_health = await check_mongo_health(self._mongo_pool, timeout=timeout)
            components.append(mongo_health)

        # Vector store health
        vector_health: Optional[StoreHealth] = None
        if self._vector_provider is not None:
            vector_health = await check_vector_health(
                self._vector_provider, timeout=timeout
            )
            components.append(vector_health)

        # Compute overall status
        overall = compute_overall_status(components)

        return StorageHealthStatus(
            overall=overall,
            sql=sql_health,
            mongo=mongo_health,
            vector_store=vector_health,
        )

    # ---------------------------------------------------------------------------
    # Unit of Work
    # ---------------------------------------------------------------------------

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[UnitOfWork]:
        """Create a unit of work for cross-store atomic operations.

        Usage::

            async with storage.unit_of_work() as uow:
                uow.add_sql_operation(lambda s: s.add(entity))
                # Commits on success, rolls back on exception
        """
        if self._uow_manager is None:
            raise RuntimeError("StorageManager not initialized")

        async with self._uow_manager.create() as uow:
            yield uow


__all__ = [
    "StorageManager",
    "get_storage_manager",
    "get_storage_manager_sync",
    "reset_storage_manager",
]
