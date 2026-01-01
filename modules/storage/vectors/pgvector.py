"""pgvector (PostgreSQL) vector store provider.

Uses the pgvector extension for PostgreSQL to store and search vectors.
Requires: pip install pgvector psycopg[binary]
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

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

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import sessionmaker

logger = setup_logger(__name__)


# Mapping from our metric enum to pgvector operator
_METRIC_TO_OPERATOR = {
    DistanceMetric.COSINE: "<=>",
    DistanceMetric.EUCLIDEAN: "<->",
    DistanceMetric.DOT_PRODUCT: "<#>",
    DistanceMetric.INNER_PRODUCT: "<#>",
}

_METRIC_TO_INDEX_OPS = {
    DistanceMetric.COSINE: "vector_cosine_ops",
    DistanceMetric.EUCLIDEAN: "vector_l2_ops",
    DistanceMetric.DOT_PRODUCT: "vector_ip_ops",
    DistanceMetric.INNER_PRODUCT: "vector_ip_ops",
}


@register_vector_provider("pgvector")
class PgVectorProvider(VectorProvider):
    """PostgreSQL pgvector-based vector store provider.

    Stores vectors in PostgreSQL tables using the pgvector extension.
    Supports IVFFlat and HNSW indexes for approximate nearest neighbor search.
    """

    def __init__(
        self,
        engine: "Engine",
        session_factory: "sessionmaker",
        *,
        schema: str = "public",
        index_type: str = "ivfflat",
        ivfflat_lists: int = 100,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        """Initialize the pgvector provider.

        Args:
            engine: SQLAlchemy engine connected to PostgreSQL.
            session_factory: Session factory for database operations.
            schema: Database schema to use.
            index_type: Index type ("ivfflat" or "hnsw").
            ivfflat_lists: Number of lists for IVFFlat index.
            hnsw_m: M parameter for HNSW index.
            hnsw_ef_construction: ef_construction for HNSW index.
        """
        self._engine = engine
        self._session_factory = session_factory
        self._schema = schema
        self._index_type = index_type.lower()
        self._ivfflat_lists = ivfflat_lists
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "pgvector"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the provider and ensure pgvector extension exists."""
        async with self._lock:
            if self._initialized:
                return

            await self._ensure_extension()
            self._initialized = True
            logger.info("pgvector provider initialized")

    async def _ensure_extension(self) -> None:
        """Ensure the pgvector extension is installed."""

        def _ensure() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()

        await asyncio.to_thread(_ensure)

    async def shutdown(self) -> None:
        """Shutdown the provider (connection cleanup handled by pool)."""
        async with self._lock:
            self._initialized = False
            logger.info("pgvector provider shutdown")

    async def health_check(self, timeout: float = 5.0) -> bool:
        """Check if PostgreSQL and pgvector are accessible."""
        if not self._initialized:
            return False

        try:

            def _check() -> bool:
                from sqlalchemy import text

                with self._engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    )
                    row = result.fetchone()
                    return row is not None

            return await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout)
        except Exception as exc:
            logger.warning(f"pgvector health check failed: {exc}")
            return False

    def _table_name(self, collection: str) -> str:
        """Get the full table name for a collection."""
        safe_name = collection.replace("-", "_").replace(" ", "_")
        return f"{self._schema}.vector_{safe_name}"

    async def create_collection(
        self,
        name: str,
        dimension: int,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectionInfo:
        """Create a new vector collection (table)."""
        table_name = self._table_name(name)
        index_ops = _METRIC_TO_INDEX_OPS.get(metric, "vector_cosine_ops")

        def _create() -> None:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                # Create the table
                conn.execute(
                    text(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id TEXT PRIMARY KEY,
                            vector vector({dimension}),
                            content TEXT,
                            metadata JSONB DEFAULT '{{}}',
                            namespace TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                )

                # Create index based on type
                index_name = f"idx_{name.replace('-', '_')}_vector"
                if self._index_type == "hnsw":
                    conn.execute(
                        text(f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING hnsw (vector {index_ops})
                            WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef_construction})
                        """)
                    )
                else:
                    conn.execute(
                        text(f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING ivfflat (vector {index_ops})
                            WITH (lists = {self._ivfflat_lists})
                        """)
                    )

                # Create namespace index for multi-tenant queries
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{name.replace('-', '_')}_namespace
                        ON {table_name} (namespace)
                    """)
                )

                conn.commit()

        await asyncio.to_thread(_create)
        logger.info(f"Created pgvector collection: {name}")

        return CollectionInfo(
            name=name,
            dimension=dimension,
            count=0,
            metric=metric,
            index_type=self._index_type,
            metadata=metadata or {},
        )

    async def delete_collection(self, name: str) -> bool:
        """Drop the collection table."""
        table_name = self._table_name(name)

        def _drop() -> bool:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"DROP TABLE IF EXISTS {table_name}")
                )
                conn.commit()
                return True

        await asyncio.to_thread(_drop)
        logger.info(f"Deleted pgvector collection: {name}")
        return True

    async def list_collections(self) -> List[CollectionInfo]:
        """List all vector collections."""

        def _list() -> List[CollectionInfo]:
            from sqlalchemy import text

            collections = []
            with self._engine.connect() as conn:
                # Find tables with vector columns
                result = conn.execute(
                    text(f"""
                        SELECT table_name, 
                               (SELECT COUNT(*) FROM information_schema.columns c2 
                                WHERE c2.table_name = c.table_name 
                                AND c2.table_schema = c.table_schema
                                AND c2.column_name = 'vector') as has_vector
                        FROM information_schema.tables c
                        WHERE table_schema = '{self._schema}'
                        AND table_name LIKE 'vector_%'
                    """)
                )
                for row in result:
                    name = row[0].replace("vector_", "")
                    # Get dimension from column
                    dim_result = conn.execute(
                        text(f"""
                            SELECT atttypmod 
                            FROM pg_attribute 
                            WHERE attrelid = '{self._schema}.{row[0]}'::regclass 
                            AND attname = 'vector'
                        """)
                    )
                    dim_row = dim_result.fetchone()
                    dimension = dim_row[0] if dim_row else 0

                    # Get count
                    count_result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {self._schema}.{row[0]}")
                    )
                    count = count_result.scalar() or 0

                    collections.append(
                        CollectionInfo(
                            name=name,
                            dimension=dimension,
                            count=count,
                            index_type=self._index_type,
                        )
                    )

            return collections

        return await asyncio.to_thread(_list)

    async def get_collection(self, name: str) -> Optional[CollectionInfo]:
        """Get info about a specific collection."""
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
        """Insert or update documents."""
        if not documents:
            return 0

        table_name = self._table_name(collection)

        def _upsert() -> int:
            from sqlalchemy import text
            import json

            count = 0
            with self._engine.connect() as conn:
                for doc in documents:
                    ns = namespace or doc.namespace
                    vector_str = "[" + ",".join(str(v) for v in doc.vector) + "]"
                    metadata_json = json.dumps(doc.metadata or {})

                    conn.execute(
                        text(f"""
                            INSERT INTO {table_name} (id, vector, content, metadata, namespace, created_at)
                            VALUES (:id, :vector, :content, :metadata, :namespace, :created_at)
                            ON CONFLICT (id) DO UPDATE SET
                                vector = EXCLUDED.vector,
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                namespace = EXCLUDED.namespace
                        """),
                        {
                            "id": doc.id,
                            "vector": vector_str,
                            "content": doc.content,
                            "metadata": metadata_json,
                            "namespace": ns,
                            "created_at": doc.created_at or datetime.utcnow(),
                        },
                    )
                    count += 1

                conn.commit()
            return count

        return await asyncio.to_thread(_upsert)

    async def delete(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> int:
        """Delete documents by ID."""
        if not ids:
            return 0

        table_name = self._table_name(collection)

        def _delete() -> int:
            from sqlalchemy import text

            with self._engine.connect() as conn:
                placeholders = ",".join(f":id{i}" for i in range(len(ids)))
                params = {f"id{i}": id_ for i, id_ in enumerate(ids)}

                query = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"
                if namespace:
                    query += " AND namespace = :namespace"
                    params["namespace"] = namespace

                result = conn.execute(text(query), params)
                conn.commit()
                return result.rowcount

        return await asyncio.to_thread(_delete)

    async def get(
        self,
        collection: str,
        ids: Sequence[str],
        *,
        namespace: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Get documents by ID."""
        if not ids:
            return []

        table_name = self._table_name(collection)

        def _get() -> List[VectorDocument]:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                placeholders = ",".join(f":id{i}" for i in range(len(ids)))
                params = {f"id{i}": id_ for i, id_ in enumerate(ids)}

                query = f"""
                    SELECT id, vector::text, content, metadata, namespace, created_at
                    FROM {table_name}
                    WHERE id IN ({placeholders})
                """
                if namespace:
                    query += " AND namespace = :namespace"
                    params["namespace"] = namespace

                result = conn.execute(text(query), params)
                docs = []
                for row in result:
                    # Parse vector from text representation
                    vector_str = row[1]
                    if vector_str:
                        vector_str = vector_str.strip("[]")
                        vector = [float(x) for x in vector_str.split(",")]
                    else:
                        vector = []

                    metadata = row[3] if row[3] else {}
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    docs.append(
                        VectorDocument(
                            id=row[0],
                            vector=vector,
                            content=row[2],
                            metadata=metadata,
                            namespace=row[4],
                            created_at=row[5],
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
        """Search for similar vectors."""
        table_name = self._table_name(collection)
        # Default to cosine distance
        operator = "<=>"

        def _search() -> List[VectorSearchResult]:
            from sqlalchemy import text
            import json

            with self._engine.connect() as conn:
                vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

                select_cols = ["id", "content", "namespace", "created_at"]
                if include_metadata:
                    select_cols.append("metadata")
                if include_vectors:
                    select_cols.append("vector::text")

                select_cols.append(f"vector {operator} :query_vector AS distance")
                select_clause = ", ".join(select_cols)

                query = f"""
                    SELECT {select_clause}
                    FROM {table_name}
                """

                params: Dict[str, Any] = {"query_vector": vector_str}
                conditions = []

                if namespace:
                    conditions.append("namespace = :namespace")
                    params["namespace"] = namespace

                if filter:
                    for key, value in filter.items():
                        param_name = f"filter_{key}"
                        conditions.append(f"metadata->>'{key}' = :{param_name}")
                        params[param_name] = str(value)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += f" ORDER BY vector {operator} :query_vector LIMIT :limit"
                params["limit"] = top_k

                result = conn.execute(text(query), params)
                results = []

                for row in result:
                    idx = 0
                    doc_id = row[idx]
                    idx += 1
                    content = row[idx]
                    idx += 1
                    ns = row[idx]
                    idx += 1
                    created_at = row[idx]
                    idx += 1

                    metadata = {}
                    if include_metadata:
                        metadata = row[idx] if row[idx] else {}
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        idx += 1

                    vector = []
                    if include_vectors:
                        vector_text = row[idx]
                        if vector_text:
                            vector_text = vector_text.strip("[]")
                            vector = [float(x) for x in vector_text.split(",")]
                        idx += 1

                    distance = row[idx]
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance if distance is not None else 0.0

                    doc = VectorDocument(
                        id=doc_id,
                        vector=vector,
                        content=content,
                        metadata=metadata,
                        namespace=ns,
                        created_at=created_at,
                    )
                    results.append(
                        VectorSearchResult(document=doc, score=score, distance=distance)
                    )

                return results

        return await asyncio.to_thread(_search)


__all__ = ["PgVectorProvider"]
