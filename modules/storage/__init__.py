"""Unified storage orchestration for ATLAS.

This module provides centralized management of all persistence subsystems:
- SQL databases (PostgreSQL, SQLite)
- Document stores (MongoDB)
- Key-Value stores
- Vector stores (pgvector, Pinecone, Chroma)
- Domain stores (conversations, tasks, jobs)

Usage::

    from modules.storage import get_storage_manager, StorageManager

    # Get the singleton instance
    storage = await get_storage_manager()

    # Or create a configured instance
    storage = StorageManager(settings=my_settings)
    await storage.initialize()

    # Access domain stores
    conversations = storage.conversations
    tasks = storage.tasks
    jobs = storage.jobs

    # Access vector store
    vectors = storage.vectors

    # Health check
    health = await storage.health_check()

    # Graceful shutdown
    await storage.shutdown()
"""

from .settings import (
    StorageSettings,
    SQLBackendSettings,
    MongoBackendSettings,
    VectorBackendSettings,
    PoolSettings,
)
from .health import StorageHealthStatus, StoreHealth
from .manager import StorageManager, get_storage_manager, get_storage_manager_sync, reset_storage_manager
from .adapters import (
    DomainStoreFactory,
    create_conversation_repository,
    create_task_repository,
    create_job_repository,
    create_kv_store,
)
from .compat import (
    storage_architecture_to_settings,
    storage_settings_to_architecture,
    create_storage_settings_from_config,
    database_state_to_settings,
    storage_settings_to_database_state,
)

__all__ = [
    # Core
    "StorageManager",
    "get_storage_manager",
    "get_storage_manager_sync",
    "reset_storage_manager",
    # Settings
    "StorageSettings",
    "SQLBackendSettings",
    "MongoBackendSettings",
    "VectorBackendSettings",
    "PoolSettings",
    # Health
    "StorageHealthStatus",
    "StoreHealth",
    # Adapters
    "DomainStoreFactory",
    "create_conversation_repository",
    "create_task_repository",
    "create_job_repository",
    "create_kv_store",
    # Config Converters
    "storage_architecture_to_settings",
    "storage_settings_to_architecture",
    "create_storage_settings_from_config",
    "database_state_to_settings",
    "storage_settings_to_database_state",
]
