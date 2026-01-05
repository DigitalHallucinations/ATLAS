"""Bridge adapters connecting StorageManager vector providers to existing vector store tools.

This module provides adapters that allow the new VectorProvider implementations
to work with the existing VectorStoreAdapter protocol used by the tools system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from core.config import ConfigManager
    from modules.storage.vectors.base import VectorProvider

logger = setup_logger(__name__)


class VectorProviderAdapter:
    """
    Adapter that wraps a VectorProvider to implement VectorStoreAdapter protocol.

    This bridges the new unified VectorProvider interface with the existing
    tool-based VectorStoreAdapter protocol.
    """

    def __init__(
        self,
        provider: "VectorProvider",
        *,
        collection_prefix: str = "",
        default_dimension: int = 1536,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            provider: The VectorProvider instance to wrap.
            collection_prefix: Optional prefix for collection names.
            default_dimension: Default embedding dimension for new collections.
        """
        self._provider = provider
        self._collection_prefix = collection_prefix
        self._default_dimension = default_dimension
        self._collection_cache: dict[str, bool] = {}

    def _collection_name(self, namespace: str) -> str:
        """Convert a namespace to a collection name."""
        if self._collection_prefix:
            return f"{self._collection_prefix}_{namespace}"
        return namespace

    async def _ensure_collection(self, namespace: str) -> str:
        """Ensure the collection exists for the namespace."""
        collection = self._collection_name(namespace)

        if collection in self._collection_cache:
            return collection

        try:
            info = await self._provider.get_collection(collection)
            if info is None:
                await self._provider.create_collection(
                    collection,
                    dimension=self._default_dimension,
                )
                logger.debug("Created collection '%s' for namespace '%s'", collection, namespace)
            self._collection_cache[collection] = True
        except Exception as exc:
            logger.warning("Failed to ensure collection '%s': %s", collection, exc)
            # Try to continue anyway
            self._collection_cache[collection] = True

        return collection

    async def upsert_vectors(
        self,
        namespace: str,
        vectors: Sequence[Any],  # Sequence[VectorRecord]
    ) -> Any:  # UpsertResponse
        """Upsert vectors to the store."""
        from modules.Tools.Base_Tools.vector_store import UpsertResponse
        from modules.storage.vectors.base import VectorDocument

        collection = await self._ensure_collection(namespace)

        # Convert VectorRecord to VectorDocument
        documents = []
        for vec in vectors:
            doc = VectorDocument(
                id=vec.id,
                vector=list(vec.values),
                metadata=dict(vec.metadata) if vec.metadata else {},
                namespace=namespace,
            )
            documents.append(doc)

        upserted_count = await self._provider.upsert(collection, documents)
        ids = tuple(vec.id for vec in vectors)

        return UpsertResponse(
            namespace=namespace,
            ids=ids,
            upserted_count=upserted_count,
        )

    async def query_vectors(
        self,
        namespace: str,
        query: Sequence[float],
        *,
        top_k: int = 10,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        include_values: bool = False,
    ) -> Any:  # QueryResponse
        """Query vectors from the store."""
        from modules.Tools.Base_Tools.vector_store import QueryMatch, QueryResponse

        collection = await self._ensure_collection(namespace)

        results = await self._provider.search(
            collection,
            query_vector=list(query),
            top_k=top_k,
            filter=dict(metadata_filter) if metadata_filter else None,
            include_vectors=include_values,
            namespace=namespace,
        )

        matches = []
        for result in results:
            match = QueryMatch(
                id=result.document.id,
                score=result.score,
                values=tuple(result.document.vector) if include_values and result.document.vector else None,
                metadata=result.document.metadata or {},
            )
            matches.append(match)

        return QueryResponse(
            namespace=namespace,
            matches=tuple(matches),
        )

    async def delete_namespace(self, namespace: str) -> Any:  # DeleteResponse
        """Delete all vectors in a namespace."""
        from modules.Tools.Base_Tools.vector_store import DeleteResponse

        collection = self._collection_name(namespace)

        try:
            deleted = await self._provider.delete_collection(collection)
            self._collection_cache.pop(collection, None)
        except Exception as exc:
            logger.warning("Failed to delete collection '%s': %s", collection, exc)
            deleted = False

        return DeleteResponse(
            namespace=namespace,
            deleted=deleted,
        )


def create_storage_manager_adapter(
    config_manager: Optional["ConfigManager"] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> VectorProviderAdapter:
    """
    Factory function to create a VectorProviderAdapter from StorageManager.

    This is designed to be registered with register_vector_store_adapter.

    Args:
        config_manager: Optional ConfigManager (for compatibility).
        config: Optional configuration dict.

    Returns:
        VectorProviderAdapter instance.

    Raises:
        RuntimeError: If StorageManager is not initialized.
    """
    from modules.storage.manager import get_storage_manager_sync

    manager = get_storage_manager_sync()
    if manager is None:
        raise RuntimeError("StorageManager not initialized")

    provider = manager.vectors
    if provider is None:
        raise RuntimeError("Vector provider not configured in StorageManager")

    config = config or {}

    return VectorProviderAdapter(
        provider,
        collection_prefix=config.get("collection_prefix", ""),
        default_dimension=config.get("default_dimension", 1536),
    )


def register_storage_manager_adapter(name: str = "storage_manager") -> None:
    """
    Register the StorageManager vector adapter with the tool system.

    Args:
        name: The adapter name to register under.
    """
    from modules.Tools.Base_Tools.vector_store import register_vector_store_adapter

    register_vector_store_adapter(name, create_storage_manager_adapter)
    logger.info("Registered StorageManager vector adapter as '%s'", name)
