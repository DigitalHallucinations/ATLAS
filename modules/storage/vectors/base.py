"""Base vector store provider abstraction.

Defines the common interface for all vector store backends (pgvector,
Pinecone, Chroma, etc.) allowing seamless switching between providers.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class VectorNotFoundError(VectorStoreError):
    """Raised when a vector document is not found."""

    pass


class VectorCollectionError(VectorStoreError):
    """Raised when collection operations fail."""

    pass


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    INNER_PRODUCT = "inner_product"


@dataclass(slots=True)
class VectorDocument:
    """A document with its vector embedding.

    Attributes:
        id: Unique identifier for the document.
        vector: The embedding vector as a list of floats.
        content: Optional text content of the document.
        metadata: Optional metadata dictionary.
        namespace: Optional namespace for multi-tenant isolation.
        created_at: Timestamp when the document was created.
    """

    id: str
    vector: List[float]
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass(slots=True)
class VectorSearchResult:
    """Result from a vector similarity search.

    Attributes:
        document: The matching document.
        score: Similarity score (interpretation depends on metric).
        distance: Raw distance value (lower is more similar for most metrics).
    """

    document: VectorDocument
    score: float
    distance: Optional[float] = None


@dataclass(slots=True)
class CollectionInfo:
    """Information about a vector collection.

    Attributes:
        name: Collection name.
        dimension: Vector dimension.
        count: Number of documents in the collection.
        metric: Distance metric used.
        index_type: Type of index (e.g., "ivfflat", "hnsw").
        metadata: Additional collection metadata.
    """

    name: str
    dimension: int
    count: int = 0
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorProvider(ABC):
    """Abstract base class for vector store providers.

    All vector store implementations (pgvector, Pinecone, Chroma)
    must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'pgvector', 'pinecone')."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and close connections."""
        ...

    @abstractmethod
    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the vector store is healthy and accessible."""
        ...

    # --- Collection Management ---

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionInfo:
        """Create a new vector collection.

        Args:
            name: Collection name.
            dimension: Vector dimension.
            metric: Distance metric to use.
            metadata: Optional collection metadata.

        Returns:
            Information about the created collection.
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a vector collection.

        Args:
            name: Collection name.

        Returns:
            True if deleted, False if collection didn't exist.
        """
        ...

    @abstractmethod
    async def list_collections(self) -> List[CollectionInfo]:
        """List all vector collections.

        Returns:
            List of collection information objects.
        """
        ...

    @abstractmethod
    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a specific collection.

        Args:
            name: Collection name.

        Returns:
            Collection info or None if not found.
        """
        ...

    # --- Document Operations ---

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        documents: Sequence[VectorDocument],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Insert or update documents in a collection.

        Args:
            collection: Collection name.
            documents: Documents to upsert.
            namespace: Optional namespace for multi-tenant isolation.

        Returns:
            Number of documents upserted.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete documents by ID.

        Args:
            collection: Collection name.
            ids: Document IDs to delete.
            namespace: Optional namespace.

        Returns:
            Number of documents deleted.
        """
        ...

    @abstractmethod
    async def get(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Get documents by ID.

        Args:
            collection: Collection name.
            ids: Document IDs to retrieve.
            namespace: Optional namespace.

        Returns:
            List of found documents.
        """
        ...

    # --- Search Operations ---

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        *,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            collection: Collection name.
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            namespace: Optional namespace filter.
            filter: Optional metadata filter.
            include_metadata: Whether to include metadata in results.
            include_vectors: Whether to include vectors in results.

        Returns:
            List of search results ordered by similarity.
        """
        ...

    async def search_by_text(
        self,
        collection: str,
        query_text: str,
        *,
        embed_fn: Callable[[str], List[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search by text using an embedding function.

        This is a convenience method that embeds the query text and
        delegates to the vector search.

        Args:
            collection: Collection name.
            query_text: Text to search for.
            embed_fn: Function to convert text to embedding.
            top_k: Number of results to return.
            namespace: Optional namespace filter.
            filter: Optional metadata filter.

        Returns:
            List of search results ordered by similarity.
        """
        query_vector = embed_fn(query_text)
        return await self.search(
            collection,
            query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

ProviderT = TypeVar("ProviderT", bound=VectorProvider)

_PROVIDER_REGISTRY: Dict[str, Type[VectorProvider]] = {}


def register_vector_provider(name: str) -> Callable[[Type[ProviderT]], Type[ProviderT]]:
    """Decorator to register a vector provider class.

    Usage::

        @register_vector_provider("pgvector")
        class PgVectorProvider(VectorProvider):
            ...
    """

    def decorator(cls: Type[ProviderT]) -> Type[ProviderT]:
        _PROVIDER_REGISTRY[name.lower()] = cls
        logger.debug(f"Registered vector provider: {name}")
        return cls

    return decorator


def get_vector_provider(name: str) -> Optional[Type[VectorProvider]]:
    """Get a registered vector provider class by name.

    Args:
        name: Provider name (case-insensitive).

    Returns:
        Provider class or None if not found.
    """
    return _PROVIDER_REGISTRY.get(name.lower())


def available_vector_providers() -> List[str]:
    """List all registered vector provider names."""
    return list(_PROVIDER_REGISTRY.keys())


class VectorStoreRegistry:
    """Registry managing vector provider instances.

    Provides a central access point for vector store operations
    with lazy initialization of providers.
    """

    def __init__(self) -> None:
        self._instances: Dict[str, VectorProvider] = {}
        self._default_provider: Optional[str] = None

    def register_instance(
        self,
        name: str,
        provider: VectorProvider,
        *,
        default: bool = False,
    ) -> None:
        """Register a provider instance.

        Args:
            name: Unique name for this instance.
            provider: The provider instance.
            default: Whether this should be the default provider.
        """
        self._instances[name] = provider
        if default or self._default_provider is None:
            self._default_provider = name
        logger.debug(f"Registered vector provider instance: {name}")

    def get(self, name: Optional[str] = None) -> Optional[VectorProvider]:
        """Get a provider instance by name.

        Args:
            name: Provider name, or None for the default.

        Returns:
            Provider instance or None if not found.
        """
        if name is None:
            name = self._default_provider
        return self._instances.get(name) if name else None

    @property
    def default(self) -> Optional[VectorProvider]:
        """Get the default provider instance."""
        return self.get(self._default_provider)

    def list_instances(self) -> List[str]:
        """List all registered instance names."""
        return list(self._instances.keys())

    async def shutdown_all(self) -> None:
        """Shutdown all registered providers."""
        for name, provider in self._instances.items():
            try:
                await provider.shutdown()
                logger.debug(f"Shutdown vector provider: {name}")
            except Exception as exc:
                logger.warning(f"Error shutting down vector provider {name}: {exc}")

        self._instances.clear()
        self._default_provider = None


__all__ = [
    "VectorStoreError",
    "VectorNotFoundError",
    "VectorCollectionError",
    "DistanceMetric",
    "VectorDocument",
    "VectorSearchResult",
    "CollectionInfo",
    "VectorProvider",
    "VectorStoreRegistry",
    "register_vector_provider",
    "get_vector_provider",
    "available_vector_providers",
]
