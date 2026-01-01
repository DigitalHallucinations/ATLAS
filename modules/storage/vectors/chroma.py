"""ChromaDB vector store provider.

Uses ChromaDB for local or cloud-based vector storage.
Requires: pip install chromadb
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


# Mapping from our metric enum to Chroma metric names
_METRIC_TO_CHROMA = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.EUCLIDEAN: "l2",
    DistanceMetric.DOT_PRODUCT: "ip",
    DistanceMetric.INNER_PRODUCT: "ip",
}


@register_vector_provider("chroma")
class ChromaProvider(VectorProvider):
    """ChromaDB vector store provider.

    Supports both local persistent storage and remote Chroma server.
    """

    def __init__(
        self,
        *,
        host: str = "",
        port: int = 8000,
        persist_directory: str = "",
        auth_token: str = "",
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> None:
        """Initialize the Chroma provider.

        Args:
            host: Chroma server host (empty for local/persistent mode).
            port: Chroma server port.
            persist_directory: Directory for local persistence (empty for in-memory).
            auth_token: Authentication token for Chroma cloud.
            tenant: Tenant name for multi-tenancy.
            database: Database name.
        """
        self._host = host
        self._port = port
        self._persist_directory = persist_directory
        self._auth_token = auth_token
        self._tenant = tenant
        self._database = database
        self._client: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "chroma"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the Chroma client."""
        async with self._lock:
            if self._initialized:
                return

            await asyncio.to_thread(self._create_client)
            self._initialized = True
            mode = "server" if self._host else ("persistent" if self._persist_directory else "ephemeral")
            logger.info(f"Chroma provider initialized in {mode} mode")

    def _create_client(self) -> None:
        """Create the Chroma client (runs in thread)."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError(
                "ChromaDB is required. Install via `pip install chromadb`."
            ) from exc

        if self._host:
            # Connect to remote Chroma server
            settings = Settings()
            if self._auth_token:
                settings = Settings(
                    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                    chroma_client_auth_credentials=self._auth_token,
                )
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                settings=settings,
            )
        elif self._persist_directory:
            # Local persistent mode
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # Ephemeral in-memory mode
            self._client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        async with self._lock:
            self._client = None
            self._initialized = False
            logger.info("Chroma provider shutdown")

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if Chroma is accessible."""
        if not self._initialized or self._client is None:
            return False

        try:

            def _check() -> bool:
                # Heartbeat check
                self._client.heartbeat()
                return True

            return await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout)
        except Exception as exc:
            logger.warning(f"Chroma health check failed: {exc}")
            return False

    async def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionInfo:
        """Create or get a Chroma collection."""
        chroma_metric = _METRIC_TO_CHROMA.get(metric, "cosine")

        def _create() -> CollectionInfo:
            collection_metadata = dict(metadata) if metadata else {}
            collection_metadata["dimension"] = dimension

            collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": chroma_metric, **collection_metadata},
            )

            return CollectionInfo(
                name=name,
                dimension=dimension,
                count=collection.count(),
                metric=metric,
                metadata=collection_metadata,
            )

        return await asyncio.to_thread(_create)

    async def delete_collection(self, name: str) -> bool:
        """Delete a Chroma collection."""

        def _delete() -> bool:
            try:
                self._client.delete_collection(name)
                logger.info(f"Deleted Chroma collection: {name}")
                return True
            except Exception as exc:
                logger.warning(f"Failed to delete Chroma collection {name}: {exc}")
                return False

        return await asyncio.to_thread(_delete)

    async def list_collections(self) -> List[CollectionInfo]:
        """List all Chroma collections."""

        def _list() -> List[CollectionInfo]:
            collections = []
            for coll in self._client.list_collections():
                metadata = coll.metadata or {}
                dimension = metadata.get("dimension", 0)
                space = metadata.get("hnsw:space", "cosine")
                metric = DistanceMetric.COSINE
                if space == "l2":
                    metric = DistanceMetric.EUCLIDEAN
                elif space == "ip":
                    metric = DistanceMetric.DOT_PRODUCT

                collections.append(
                    CollectionInfo(
                        name=coll.name,
                        dimension=dimension,
                        count=coll.count(),
                        metric=metric,
                        metadata=metadata,
                    )
                )
            return collections

        return await asyncio.to_thread(_list)

    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get info about a specific collection."""

        def _get() -> Optional[CollectionInfo]:
            try:
                coll = self._client.get_collection(name)
                metadata = coll.metadata or {}
                dimension = metadata.get("dimension", 0)
                space = metadata.get("hnsw:space", "cosine")
                metric = DistanceMetric.COSINE
                if space == "l2":
                    metric = DistanceMetric.EUCLIDEAN
                elif space == "ip":
                    metric = DistanceMetric.DOT_PRODUCT

                return CollectionInfo(
                    name=coll.name,
                    dimension=dimension,
                    count=coll.count(),
                    metric=metric,
                    metadata=metadata,
                )
            except Exception:
                return None

        return await asyncio.to_thread(_get)

    async def upsert(
        self,
        collection: str,
        documents: Sequence[VectorDocument],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Add or update documents in Chroma."""
        if not documents:
            return 0

        def _upsert() -> int:
            coll = self._client.get_collection(collection)

            ids = []
            embeddings = []
            metadatas = []
            doc_contents = []

            for doc in documents:
                ids.append(doc.id)
                embeddings.append(doc.vector)

                metadata = dict(doc.metadata) if doc.metadata else {}
                if namespace or doc.namespace:
                    metadata["_namespace"] = namespace or doc.namespace
                if doc.created_at:
                    metadata["_created_at"] = doc.created_at.isoformat()
                metadatas.append(metadata)

                doc_contents.append(doc.content or "")

            coll.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=doc_contents,
            )

            return len(ids)

        return await asyncio.to_thread(_upsert)

    async def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete documents from Chroma."""
        if not ids:
            return 0

        def _delete() -> int:
            coll = self._client.get_collection(collection)

            where_filter = None
            if namespace:
                where_filter = {"_namespace": namespace}

            coll.delete(ids=list(ids), where=where_filter)
            return len(ids)

        return await asyncio.to_thread(_delete)

    async def get(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Fetch documents by ID from Chroma."""
        if not ids:
            return []

        def _get() -> List[VectorDocument]:
            coll = self._client.get_collection(collection)

            result = coll.get(
                ids=list(ids),
                include=["embeddings", "metadatas", "documents"],
            )

            docs = []
            for i, id_ in enumerate(result.get("ids", [])):
                metadata = result.get("metadatas", [{}])[i] or {}
                namespace_val = metadata.pop("_namespace", None)
                created_at_str = metadata.pop("_created_at", None)
                created_at = None
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except ValueError:
                        pass

                embeddings = result.get("embeddings")
                vector = embeddings[i] if embeddings else []

                documents = result.get("documents")
                content = documents[i] if documents else None

                docs.append(
                    VectorDocument(
                        id=id_,
                        vector=vector,
                        content=content,
                        metadata=metadata,
                        namespace=namespace_val,
                        created_at=created_at,
                    )
                )

            return docs

        return await asyncio.to_thread(_get)

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
        """Query Chroma for similar vectors."""

        def _search() -> List[VectorSearchResult]:
            coll = self._client.get_collection(collection)

            where_filter = None
            if namespace or filter:
                where_filter = {}
                if namespace:
                    where_filter["_namespace"] = namespace
                if filter:
                    where_filter.update(filter)

            include = ["documents"]
            if include_metadata:
                include.append("metadatas")
            if include_vectors:
                include.append("embeddings")

            result = coll.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=include,
            )

            results = []
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            documents = result.get("documents", [[]])[0] if "documents" in result else [None] * len(ids)
            metadatas = result.get("metadatas", [[]])[0] if "metadatas" in result else [{}] * len(ids)
            embeddings = result.get("embeddings", [[]])[0] if include_vectors and "embeddings" in result else [None] * len(ids)

            for i, id_ in enumerate(ids):
                metadata = metadatas[i] or {}
                namespace_val = metadata.pop("_namespace", None)
                created_at_str = metadata.pop("_created_at", None)
                created_at = None
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except ValueError:
                        pass

                distance = distances[i] if distances else None
                # Convert distance to similarity score
                # For cosine, score = 1 - distance (Chroma returns distance)
                score = 1.0 - distance if distance is not None else 0.0

                doc = VectorDocument(
                    id=id_,
                    vector=embeddings[i] if embeddings[i] else [],
                    content=documents[i] if documents else None,
                    metadata=metadata,
                    namespace=namespace_val,
                    created_at=created_at,
                )
                results.append(
                    VectorSearchResult(
                        document=doc,
                        score=score,
                        distance=distance,
                    )
                )

            return results

        return await asyncio.to_thread(_search)


__all__ = ["ChromaProvider"]
