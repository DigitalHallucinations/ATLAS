"""MongoDB vector store adapter leveraging Atlas/Community vector search."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from modules.logging.logger import setup_logger
from modules.Tools.Base_Tools.vector_store import (
    DeleteResponse,
    QueryMatch,
    QueryResponse,
    UpsertResponse,
    VectorRecord,
    VectorStoreAdapter,
    VectorStoreConfigurationError,
    VectorStoreOperationError,
    register_vector_store_adapter,
)

try:  # pragma: no cover - optional dependency
    from pymongo import MongoClient, UpdateOne
    from pymongo.collection import Collection
    from pymongo.errors import OperationFailure
except Exception:  # pragma: no cover - pymongo is optional at runtime
    MongoClient = None  # type: ignore[assignment]
    UpdateOne = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment]

    class OperationFailure(Exception):
        """Fallback error when :mod:`pymongo` is unavailable."""


logger = setup_logger(__name__)

DEFAULT_DATABASE = "atlas_vector_store"
DEFAULT_COLLECTION = "embeddings"
DEFAULT_INDEX_NAME = "vector_index"


def _flatten_filter_payload(payload: Mapping[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for key, value in payload.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        path = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, Mapping):
            yield from _flatten_filter_payload(value, path)
        else:
            yield path, value


def _build_match_filter(
    namespace: str,
    metadata_filter: Optional[Mapping[str, Any]],
    *,
    metadata_field: str,
) -> Dict[str, Any]:
    match_filter: Dict[str, Any] = {"namespace": namespace}
    if metadata_filter:
        for path, value in _flatten_filter_payload(metadata_filter):
            field_path = f"{metadata_field}.{path}" if metadata_field else path
            match_filter[field_path] = value
    return match_filter


def ensure_mongodb_vector_index(
    collection: Collection,
    *,
    index_name: str,
    embedding_field: str,
    num_dimensions: int,
    similarity: str = "cosine",
) -> bool:
    """Create a native MongoDB vector search index when absent."""

    try:
        existing = collection.index_information()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise VectorStoreOperationError("Failed to inspect MongoDB indexes.") from exc

    if index_name in existing:
        return False

    command = {
        "createIndexes": collection.name,
        "indexes": [
            {
                "name": index_name,
                "key": {embedding_field: "vector"},
                "vector": {
                    "path": embedding_field,
                    "numDimensions": int(num_dimensions),
                    "similarity": similarity,
                },
            }
        ],
    }

    try:
        collection.database.command(command)
        logger.info(
            "Created MongoDB vector index '%s' on collection '%s.%s'",
            index_name,
            collection.database.name,
            collection.name,
        )
        return True
    except OperationFailure as exc:  # pragma: no cover - surface provisioning issues
        if "already exists" in str(exc).lower():
            return False
        raise VectorStoreOperationError("Failed to create MongoDB vector index.") from exc


def ensure_atlas_search_index(
    collection: Collection,
    *,
    index_name: str,
    embedding_field: str,
    num_dimensions: int,
    similarity: str = "cosine",
    definition_overrides: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Create an Atlas Search knnVector index when absent."""

    list_search_indexes = getattr(collection, "list_search_indexes", None)
    if callable(list_search_indexes):
        try:
            for entry in list_search_indexes(index_name):
                if entry.get("name") == index_name:
                    return False
        except Exception:
            logger.debug("Unable to enumerate Atlas search indexes", exc_info=True)

    definition: Dict[str, Any] = {
        "mappings": {
            "dynamic": True,
            "fields": {
                embedding_field: {
                    "type": "knnVector",
                    "dimensions": int(num_dimensions),
                    "similarity": similarity,
                }
            },
        }
    }

    if isinstance(definition_overrides, Mapping):
        def _merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, Mapping) and isinstance(dst.get(key), MutableMapping):
                    _merge(dst[key], value)  # type: ignore[index]
                else:
                    dst[key] = value  # type: ignore[index]

        _merge(definition, definition_overrides)

    command = {
        "createSearchIndexes": collection.name,
        "indexes": [
            {
                "name": index_name,
                "definition": definition,
            }
        ],
    }

    try:
        collection.database.command(command)
        logger.info(
            "Created Atlas search index '%s' on collection '%s.%s'",
            index_name,
            collection.database.name,
            collection.name,
        )
        return True
    except OperationFailure as exc:  # pragma: no cover - surface provisioning issues
        if "already exists" in str(exc).lower():
            return False
        raise VectorStoreOperationError("Failed to create Atlas search index.") from exc


def drop_mongodb_vector_index(collection: Collection, *, index_name: str, atlas: bool = False) -> bool:
    """Drop an Atlas or native MongoDB vector index."""

    try:
        if atlas:
            command = {"dropSearchIndex": collection.name, "name": index_name}
            collection.database.command(command)
        else:
            collection.drop_index(index_name)
        return True
    except OperationFailure as exc:  # pragma: no cover - tolerated failures
        if "not found" in str(exc).lower():
            return False
        logger.warning("Failed to drop MongoDB vector index '%s': %s", index_name, exc)
        return False
    except Exception:
        logger.warning("Failed to drop MongoDB vector index '%s'", index_name, exc_info=True)
        return False


@dataclass
class _CollectionHandle:
    client_factory: Any
    database: str
    collection: str

    def get(self) -> Collection:
        client = self.client_factory()
        return client[self.database][self.collection]


class MongoDBVectorStoreAdapter(VectorStoreAdapter):
    """Adapter backed by MongoDB vector search capabilities."""

    def __init__(
        self,
        *,
        connection_uri: Optional[str] = None,
        client: Optional[Any] = None,
        client_factory: Optional[Any] = None,
        database: str = DEFAULT_DATABASE,
        collection: str = DEFAULT_COLLECTION,
        index_name: str = DEFAULT_INDEX_NAME,
        embedding_field: str = "embedding",
        metadata_field: str = "metadata",
        similarity: str = "cosine",
        manage_index: bool = True,
        index_type: str = "auto",
        index_overrides: Optional[Mapping[str, Any]] = None,
        vector_dimension: Optional[int] = None,
        num_candidates: Optional[int] = None,
        candidate_multiplier: int = 4,
        search_stage: Optional[str] = None,
        client_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if MongoClient is None and client is None and client_factory is None:
            raise VectorStoreConfigurationError(
                "The 'pymongo' package is required for the MongoDB vector store adapter."
            )
        if UpdateOne is None:
            raise VectorStoreConfigurationError(
                "The 'pymongo' package is required for the MongoDB vector store adapter."
            )

        if not database.strip():
            raise VectorStoreConfigurationError("MongoDB adapter requires a database name.")
        if not collection.strip():
            raise VectorStoreConfigurationError("MongoDB adapter requires a collection name.")
        if not index_name.strip():
            raise VectorStoreConfigurationError("MongoDB adapter requires an index name.")

        self._connection_uri = connection_uri
        self._client = client
        self._client_factory = client_factory
        self._client_kwargs = dict(client_kwargs or {})
        self._handle = _CollectionHandle(
            client_factory=self._resolve_client,
            database=database.strip(),
            collection=collection.strip(),
        )
        self._index_name = index_name.strip()
        self._embedding_field = embedding_field.strip() or "embedding"
        self._metadata_field = metadata_field.strip()
        self._similarity = similarity
        self._manage_index = manage_index
        self._index_type = (index_type or "auto").lower()
        self._index_overrides = index_overrides
        self._vector_dimension = vector_dimension
        self._num_candidates = num_candidates
        self._candidate_multiplier = max(1, int(candidate_multiplier))
        self._search_stage = (search_stage or "auto").lower()
        self._collection_singleton: Optional[Collection] = None
        self._client_singleton: Optional[Any] = None
        self._index_checked = False

    def _resolve_client(self) -> Any:
        if self._client_singleton is not None:
            return self._client_singleton
        if self._client is not None:
            self._client_singleton = self._client
            return self._client_singleton
        if callable(self._client_factory):
            self._client_singleton = self._client_factory()
            return self._client_singleton
        if not self._connection_uri:
            raise VectorStoreConfigurationError("MongoDB adapter requires a connection URI.")
        if MongoClient is None:  # pragma: no cover - defensive guard
            raise VectorStoreConfigurationError(
                "The 'pymongo' package is required for the MongoDB vector store adapter."
            )
        try:
            self._client_singleton = MongoClient(self._connection_uri, **self._client_kwargs)
            return self._client_singleton
        except Exception as exc:  # pragma: no cover - connection failures
            raise VectorStoreConfigurationError("Unable to create MongoDB client.") from exc

    def _get_collection(self) -> Collection:
        if self._collection_singleton is not None:
            return self._collection_singleton
        collection = self._handle.get()
        self._collection_singleton = collection
        return collection

    def _ensure_index(self, vectors: Sequence[VectorRecord]) -> None:
        if not self._manage_index or self._index_checked:
            return
        if not vectors:
            return
        dimension = self._vector_dimension or len(vectors[0].values)
        collection = self._get_collection()
        index_type = self._index_type
        try:
            if index_type == "atlas":
                ensure_atlas_search_index(
                    collection,
                    index_name=self._index_name,
                    embedding_field=self._embedding_field,
                    num_dimensions=dimension,
                    similarity=self._similarity,
                    definition_overrides=self._index_overrides,
                )
            elif index_type == "vector_search":
                ensure_mongodb_vector_index(
                    collection,
                    index_name=self._index_name,
                    embedding_field=self._embedding_field,
                    num_dimensions=dimension,
                    similarity=self._similarity,
                )
            else:
                created = ensure_mongodb_vector_index(
                    collection,
                    index_name=self._index_name,
                    embedding_field=self._embedding_field,
                    num_dimensions=dimension,
                    similarity=self._similarity,
                )
                if not created:
                    ensure_atlas_search_index(
                        collection,
                        index_name=self._index_name,
                        embedding_field=self._embedding_field,
                        num_dimensions=dimension,
                        similarity=self._similarity,
                        definition_overrides=self._index_overrides,
                    )
        except Exception:
            self._index_checked = False
            raise
        else:
            self._index_checked = True

    def _candidate_count(self, top_k: int) -> int:
        if isinstance(self._num_candidates, int) and self._num_candidates > 0:
            return max(top_k, self._num_candidates)
        return max(top_k, top_k * self._candidate_multiplier)

    async def upsert_vectors(self, namespace: str, vectors: tuple[VectorRecord, ...]) -> UpsertResponse:
        if not vectors:
            return UpsertResponse(namespace=namespace, ids=tuple(), upserted_count=0)

        self._ensure_index(vectors)

        collection = self._get_collection()
        operations = []
        for record in vectors:
            document = {
                "namespace": namespace,
                self._embedding_field: [float(value) for value in record.values],
            }
            if record.metadata:
                document[self._metadata_field or "metadata"] = dict(record.metadata)
            operations.append(
                UpdateOne(
                    {"_id": record.id, "namespace": namespace},
                    {"$set": document, "$setOnInsert": {"_id": record.id}},
                    upsert=True,
                )
            )

        try:
            await asyncio.to_thread(collection.bulk_write, operations, ordered=False)
        except Exception as exc:  # pragma: no cover - network/validation failures
            raise VectorStoreOperationError("MongoDB bulk upsert failed.") from exc

        return UpsertResponse(
            namespace=namespace,
            ids=tuple(record.id for record in vectors),
            upserted_count=len(vectors),
        )

    def _search_stage_name(self, collection: Collection) -> str:
        if self._search_stage == "atlas":
            return "search"
        if self._search_stage == "vector_search":
            return "vectorSearch"
        if self._index_type == "atlas":
            return "search"
        if self._index_type == "vector_search":
            return "vectorSearch"
        # Default to native vector search; callers can override via config.
        return "vectorSearch"

    async def query_vectors(
        self,
        namespace: str,
        query: tuple[float, ...],
        *,
        top_k: int,
        metadata_filter: Optional[Mapping[str, Any]],
        include_values: bool,
    ) -> QueryResponse:
        collection = self._get_collection()
        stage_name = self._search_stage_name(collection)
        pipeline: list[Mapping[str, Any]] = []
        if stage_name == "search":
            stage: Dict[str, Any] = {
                "$search": {
                    "index": self._index_name,
                    "knnBeta": {
                        "vector": [float(value) for value in query],
                        "path": self._embedding_field,
                        "k": self._candidate_count(top_k),
                    },
                }
            }
            pipeline.append(stage)
            pipeline.append({"$match": _build_match_filter(namespace, metadata_filter, metadata_field=self._metadata_field or "metadata")})
            pipeline.append({"$limit": top_k})
        else:
            stage = {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": self._embedding_field,
                    "queryVector": [float(value) for value in query],
                    "numCandidates": self._candidate_count(top_k),
                    "limit": top_k,
                }
            }
            pipeline.append(stage)
            pipeline.append({"$match": _build_match_filter(namespace, metadata_filter, metadata_field=self._metadata_field or "metadata")})
            pipeline.append({"$limit": top_k})

        try:
            results = await asyncio.to_thread(lambda: list(collection.aggregate(pipeline)))
        except Exception as exc:  # pragma: no cover - aggregation failures
            raise VectorStoreOperationError("MongoDB vector query failed.") from exc

        matches: list[QueryMatch] = []
        for doc in results:
            if not isinstance(doc, Mapping):
                continue
            vector_id = str(doc.get("_id", ""))
            score = doc.get("score")
            if isinstance(score, Mapping):
                score = score.get("value", 0.0)
            if score is None:
                score = doc.get("distance")
            try:
                score_value = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score_value = 0.0
            raw_values = doc.get(self._embedding_field) if include_values else None
            values: Optional[tuple[float, ...]]
            if include_values and isinstance(raw_values, Sequence):
                try:
                    values = tuple(float(v) for v in raw_values)
                except (TypeError, ValueError):  # pragma: no cover - inconsistent payloads
                    values = None
            else:
                values = None
            metadata = doc.get(self._metadata_field or "metadata", {})
            normalized_metadata: Mapping[str, Any]
            if isinstance(metadata, Mapping):
                normalized_metadata = {str(key): value for key, value in metadata.items()}
            else:
                normalized_metadata = {}
            matches.append(
                QueryMatch(
                    id=vector_id,
                    score=score_value,
                    values=values,
                    metadata=normalized_metadata,
                )
            )

        return QueryResponse(namespace=namespace, matches=tuple(matches[:top_k]), top_k=top_k)

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        collection = self._get_collection()
        try:
            result = await asyncio.to_thread(collection.delete_many, {"namespace": namespace})
        except Exception as exc:  # pragma: no cover - network failures
            raise VectorStoreOperationError("MongoDB namespace deletion failed.") from exc
        deleted_count = getattr(result, "deleted_count", 0) or 0
        return DeleteResponse(
            namespace=namespace,
            removed_ids=tuple(),
            deleted=deleted_count > 0,
        )


def _adapter_factory(config_manager: Optional[Any], config: Mapping[str, Any]) -> MongoDBVectorStoreAdapter:
    settings: Dict[str, Any] = {}
    if config_manager is not None:
        try:
            tools_config = config_manager.get_config("tools", {})
        except Exception:  # pragma: no cover - defensive guard
            tools_config = {}
        if isinstance(tools_config, Mapping):
            vector_block = tools_config.get("vector_store", {})
            if isinstance(vector_block, Mapping):
                adapters_block = vector_block.get("adapters", {})
                if isinstance(adapters_block, Mapping):
                    mongo_block = adapters_block.get("mongodb", {})
                    if isinstance(mongo_block, Mapping):
                        settings.update(mongo_block)

    for key, value in config.items():
        settings[key] = value

    connection_uri = settings.get("connection_uri")
    client_kwargs = settings.get("client_kwargs") if isinstance(settings.get("client_kwargs"), Mapping) else {}

    client_candidate = settings.get("client")
    client_factory_candidate = settings.get("client_factory")

    return MongoDBVectorStoreAdapter(
        connection_uri=connection_uri,
        client=client_candidate,
        client_factory=client_factory_candidate if callable(client_factory_candidate) else None,
        database=str(settings.get("database", DEFAULT_DATABASE)),
        collection=str(settings.get("collection", DEFAULT_COLLECTION)),
        index_name=str(settings.get("index_name", DEFAULT_INDEX_NAME)),
        embedding_field=str(settings.get("embedding_field", "embedding")),
        metadata_field=str(settings.get("metadata_field", "metadata")),
        similarity=str(settings.get("similarity", "cosine")),
        manage_index=bool(settings.get("manage_index", True)),
        index_type=str(settings.get("index_type", "auto")),
        index_overrides=settings.get("index_overrides")
        if isinstance(settings.get("index_overrides"), Mapping)
        else None,
        vector_dimension=int(settings.get("vector_dimension")) if settings.get("vector_dimension") else None,
        num_candidates=int(settings.get("num_candidates")) if settings.get("num_candidates") else None,
        candidate_multiplier=int(settings.get("candidate_multiplier", 4)),
        search_stage=str(settings.get("search_stage", "auto")),
        client_kwargs=dict(client_kwargs),
    )


register_vector_store_adapter("mongodb", _adapter_factory)


__all__ = [
    "MongoDBVectorStoreAdapter",
    "ensure_mongodb_vector_index",
    "ensure_atlas_search_index",
    "drop_mongodb_vector_index",
]

