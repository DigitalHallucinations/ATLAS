"""Pinecone vector store provider.

Uses the Pinecone managed vector database service.
Requires: pip install pinecone-client
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from modules.logging.logger import setup_logger

from .base import (
    CollectionInfo,
    DistanceMetric,
    VectorCollectionError,
    VectorDocument,
    VectorProvider,
    VectorSearchResult,
    VectorStoreError,
    register_vector_provider,
)

logger = setup_logger(__name__)


# Mapping from our metric enum to Pinecone metric names
_METRIC_TO_PINECONE = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.EUCLIDEAN: "euclidean",
    DistanceMetric.DOT_PRODUCT: "dotproduct",
    DistanceMetric.INNER_PRODUCT: "dotproduct",
}


@register_vector_provider("pinecone")
class PineconeProvider(VectorProvider):
    """Pinecone managed vector database provider.

    Supports serverless and pod-based Pinecone indexes for
    high-performance vector similarity search.
    """

    def __init__(
        self,
        api_key: str,
        *,
        environment: str = "",
        index_name: str = "atlas",
        namespace: str = "",
        metric: str = "cosine",
        dimension: int = 1536,
    ) -> None:
        """Initialize the Pinecone provider.

        Args:
            api_key: Pinecone API key.
            environment: Pinecone environment (for pod-based, empty for serverless).
            index_name: Name of the Pinecone index.
            namespace: Default namespace for operations.
            metric: Distance metric ("cosine", "euclidean", "dotproduct").
            dimension: Vector dimension.
        """
        self._api_key = api_key
        self._environment = environment
        self._index_name = index_name
        self._default_namespace = namespace
        self._metric = metric
        self._dimension = dimension
        self._client: Any = None
        self._index: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "pinecone"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the Pinecone client and connect to the index."""
        async with self._lock:
            if self._initialized:
                return

            await asyncio.to_thread(self._create_client)
            self._initialized = True
            logger.info(f"Pinecone provider initialized with index: {self._index_name}")

    def _create_client(self) -> None:
        """Create the Pinecone client (runs in thread)."""
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError(
                "Pinecone client is required. Install via `pip install pinecone-client`."
            ) from exc

        self._client = Pinecone(api_key=self._api_key)

        # Check if index exists, if so connect to it
        existing_indexes = [idx.name for idx in self._client.list_indexes()]
        if self._index_name in existing_indexes:
            self._index = self._client.Index(self._index_name)

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        async with self._lock:
            self._index = None
            self._client = None
            self._initialized = False
            logger.info("Pinecone provider shutdown")

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if Pinecone is accessible."""
        if not self._initialized or self._client is None:
            return False

        try:

            def _check() -> bool:
                indexes = self._client.list_indexes()
                return True

            return await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout)
        except Exception as exc:
            logger.warning(f"Pinecone health check failed: {exc}")
            return False

    async def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionInfo:
        """Create a new Pinecone index.

        Note: In Pinecone, "collections" map to indexes or namespaces.
        This creates an index if it doesn't exist.
        """
        pinecone_metric = _METRIC_TO_PINECONE.get(metric, "cosine")

        def _create() -> None:
            from pinecone import ServerlessSpec

            existing = [idx.name for idx in self._client.list_indexes()]
            if name in existing:
                logger.info(f"Pinecone index {name} already exists")
                return

            # Create serverless index by default
            self._client.create_index(
                name=name,
                dimension=dimension,
                metric=pinecone_metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Created Pinecone index: {name}")

        await asyncio.to_thread(_create)

        # Connect to the index
        self._index_name = name
        self._index = self._client.Index(name)

        return CollectionInfo(
            name=name,
            dimension=dimension,
            count=0,
            metric=metric,
            metadata=metadata or {},
        )

    async def delete_collection(self, name: str) -> bool:
        """Delete a Pinecone index."""

        def _delete() -> bool:
            try:
                self._client.delete_index(name)
                logger.info(f"Deleted Pinecone index: {name}")
                return True
            except Exception as exc:
                logger.warning(f"Failed to delete Pinecone index {name}: {exc}")
                return False

        return await asyncio.to_thread(_delete)

    async def list_collections(self) -> List[CollectionInfo]:
        """List all Pinecone indexes."""

        def _list() -> List[CollectionInfo]:
            collections = []
            for idx in self._client.list_indexes():
                collections.append(
                    CollectionInfo(
                        name=idx.name,
                        dimension=idx.dimension,
                        count=0,  # Would need describe_index_stats for count
                        metric=DistanceMetric(idx.metric)
                        if idx.metric in [m.value for m in DistanceMetric]
                        else DistanceMetric.COSINE,
                    )
                )
            return collections

        return await asyncio.to_thread(_list)

    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get info about a specific index."""
        collections = await self.list_collections()
        for coll in collections:
            if coll.name == name:
                return coll
        return None

    async def upsert(
        self,
        collection: str,
        documents: Sequence[VectorDocument],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Upsert vectors to Pinecone."""
        if not documents:
            return 0

        ns = namespace or self._default_namespace

        def _upsert() -> int:
            # Ensure we're connected to the right index
            index = self._client.Index(collection)

            vectors = []
            for doc in documents:
                vector_data = {
                    "id": doc.id,
                    "values": doc.vector,
                }
                if doc.metadata or doc.content:
                    metadata = dict(doc.metadata) if doc.metadata else {}
                    if doc.content:
                        metadata["_content"] = doc.content
                    vector_data["metadata"] = metadata

                vectors.append(vector_data)

            # Upsert in batches of 100
            batch_size = 100
            total = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                index.upsert(vectors=batch, namespace=ns)
                total += len(batch)

            return total

        return await asyncio.to_thread(_upsert)

    async def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors from Pinecone."""
        if not ids:
            return 0

        ns = namespace or self._default_namespace

        def _delete() -> int:
            index = self._client.Index(collection)
            index.delete(ids=list(ids), namespace=ns)
            return len(ids)

        return await asyncio.to_thread(_delete)

    async def get(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Fetch vectors by ID from Pinecone."""
        if not ids:
            return []

        ns = namespace or self._default_namespace

        def _fetch() -> List[VectorDocument]:
            index = self._client.Index(collection)
            result = index.fetch(ids=list(ids), namespace=ns)

            docs = []
            for id_, data in result.get("vectors", {}).items():
                metadata = data.get("metadata", {})
                content = metadata.pop("_content", None)

                docs.append(
                    VectorDocument(
                        id=id_,
                        vector=data.get("values", []),
                        content=content,
                        metadata=metadata,
                        namespace=ns,
                    )
                )
            return docs

        return await asyncio.to_thread(_fetch)

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
        """Query Pinecone for similar vectors."""
        ns = namespace or self._default_namespace

        def _search() -> List[VectorSearchResult]:
            index = self._client.Index(collection)

            query_response = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=ns,
                filter=filter,
                include_metadata=include_metadata,
                include_values=include_vectors,
            )

            results = []
            for match in query_response.get("matches", []):
                metadata = match.get("metadata", {})
                content = metadata.pop("_content", None) if metadata else None

                doc = VectorDocument(
                    id=match["id"],
                    vector=match.get("values", []),
                    content=content,
                    metadata=metadata,
                    namespace=ns,
                )
                results.append(
                    VectorSearchResult(
                        document=doc,
                        score=match.get("score", 0.0),
                    )
                )

            return results

        return await asyncio.to_thread(_search)


__all__ = ["PineconeProvider"]
