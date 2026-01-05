"""Bridge adapters connecting StorageManager to existing domain stores.

This module provides adapters that allow existing repositories
(ConversationStoreRepository, TaskStoreRepository, etc.) to be
constructed from a StorageManager instance instead of requiring
their own connection setup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from modules.storage.manager import StorageManager


class StorageAdapterMixin:
    """Mixin providing StorageManager integration for domain stores."""

    _storage_manager: Optional["StorageManager"] = None

    @classmethod
    def from_storage_manager(
        cls,
        storage_manager: "StorageManager",
        **kwargs: Any,
    ):
        """
        Create an instance using StorageManager for connection management.

        Args:
            storage_manager: The initialized StorageManager instance.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A new instance of the repository.
        """
        raise NotImplementedError("Subclasses must implement from_storage_manager")


def create_conversation_repository(
    storage_manager: "StorageManager",
    *,
    retention: Optional[Dict[str, Any]] = None,
    require_tenant_context: bool = False,
):
    """
    Create a ConversationStoreRepository backed by StorageManager.

    Args:
        storage_manager: The initialized StorageManager instance.
        retention: Retention policy configuration.
        require_tenant_context: Whether to require tenant context for operations.

    Returns:
        ConversationStoreRepository instance.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    from modules.conversation_store import ConversationStoreRepository

    if not storage_manager.is_initialized:
        raise RuntimeError("StorageManager must be initialized before creating repositories")

    session_factory = storage_manager.get_session_factory()

    return ConversationStoreRepository(
        session_factory,
        retention=retention,
        require_tenant_context=require_tenant_context,
    )


def create_task_repository(
    storage_manager: "StorageManager",
):
    """
    Create a TaskStoreRepository backed by StorageManager.

    Args:
        storage_manager: The initialized StorageManager instance.

    Returns:
        TaskStoreRepository instance.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    from modules.task_store import TaskStoreRepository

    if not storage_manager.is_initialized:
        raise RuntimeError("StorageManager must be initialized before creating repositories")

    session_factory = storage_manager.get_session_factory()

    repository = TaskStoreRepository(session_factory)
    # Schema creation is handled by StorageManager.ensure_schemas()
    return repository


def create_job_repository(
    storage_manager: "StorageManager",
):
    """
    Create a JobStoreRepository backed by StorageManager.

    Args:
        storage_manager: The initialized StorageManager instance.

    Returns:
        JobStoreRepository or MongoJobStoreRepository instance.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    from modules.job_store import JobStoreRepository

    if not storage_manager.is_initialized:
        raise RuntimeError("StorageManager must be initialized before creating repositories")

    session_factory = storage_manager.get_session_factory()

    repository = JobStoreRepository(session_factory)
    return repository


def create_kv_store(
    storage_manager: "StorageManager",
    *,
    namespace: str = "default",
):
    """
    Create a KV store backed by StorageManager.

    Args:
        storage_manager: The initialized StorageManager instance.
        namespace: The namespace for key isolation.

    Returns:
        KVStore instance.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    # Import here to avoid circular dependency issues
    try:
        from modules.store_common.kv_store import KVStore
    except ImportError:
        # Fallback for different module structure
        from modules.kv_store import KVStore  # type: ignore

    if not storage_manager.is_initialized:
        raise RuntimeError("StorageManager must be initialized before creating stores")

    session_factory = storage_manager.get_session_factory()

    return KVStore(session_factory, namespace=namespace)


class DomainStoreFactory:
    """
    Factory class for creating domain stores from a StorageManager.

    This provides a single point of access for creating all domain
    store instances with consistent configuration.
    """

    def __init__(
        self,
        storage_manager: "StorageManager",
        *,
        default_retention: Optional[Dict[str, Any]] = None,
        require_tenant_context: bool = False,
    ) -> None:
        """
        Initialize the factory.

        Args:
            storage_manager: The StorageManager instance to use.
            default_retention: Default retention policy for applicable stores.
            require_tenant_context: Default tenant context requirement.
        """
        self._storage_manager = storage_manager
        self._default_retention = default_retention or {}
        self._require_tenant_context = require_tenant_context

        # Cached instances
        self._conversation_repo = None
        self._task_repo = None
        self._job_repo = None
        self._kv_stores: Dict[str, Any] = {}

    @property
    def storage_manager(self) -> "StorageManager":
        """Return the underlying StorageManager."""
        return self._storage_manager

    def get_conversation_repository(
        self,
        *,
        retention: Optional[Dict[str, Any]] = None,
        require_tenant_context: Optional[bool] = None,
    ):
        """
        Get or create the conversation repository.

        Args:
            retention: Override retention policy.
            require_tenant_context: Override tenant context requirement.

        Returns:
            ConversationStoreRepository instance.
        """
        if self._conversation_repo is None:
            self._conversation_repo = create_conversation_repository(
                self._storage_manager,
                retention=retention or self._default_retention,
                require_tenant_context=(
                    require_tenant_context
                    if require_tenant_context is not None
                    else self._require_tenant_context
                ),
            )
        return self._conversation_repo

    def get_task_repository(self):
        """
        Get or create the task repository.

        Returns:
            TaskStoreRepository instance.
        """
        if self._task_repo is None:
            self._task_repo = create_task_repository(self._storage_manager)
        return self._task_repo

    def get_job_repository(self):
        """
        Get or create the job repository.

        Returns:
            JobStoreRepository instance.
        """
        if self._job_repo is None:
            self._job_repo = create_job_repository(self._storage_manager)
        return self._job_repo

    def get_kv_store(self, namespace: str = "default"):
        """
        Get or create a KV store for the given namespace.

        Args:
            namespace: The namespace for key isolation.

        Returns:
            KVStore instance.
        """
        if namespace not in self._kv_stores:
            self._kv_stores[namespace] = create_kv_store(
                self._storage_manager,
                namespace=namespace,
            )
        return self._kv_stores[namespace]

    def clear_caches(self) -> None:
        """Clear all cached repository instances."""
        self._conversation_repo = None
        self._task_repo = None
        self._job_repo = None
        self._kv_stores.clear()


def create_budget_store(
    storage_manager: "StorageManager",
):
    """
    Create a BudgetStore backed by StorageManager.

    Args:
        storage_manager: The initialized StorageManager instance.

    Returns:
        BudgetStore instance configured with SQL backend.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    from modules.budget.persistence import SQLBudgetStore

    if not storage_manager.is_initialized:
        raise RuntimeError("StorageManager must be initialized before creating stores")

    session_factory = storage_manager.get_session_factory()
    engine = storage_manager.get_sql_engine()

    return SQLBudgetStore(
        session_factory=session_factory,
        engine=engine,
    )
