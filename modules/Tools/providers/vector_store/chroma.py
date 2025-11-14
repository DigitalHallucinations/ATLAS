"""Chroma vector store adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

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

logger = setup_logger(__name__)


def _resolve_settings(
    provider_name: str,
    config_manager: Optional[Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if config_manager is not None:
        try:
            tools_config = config_manager.get_config("tools", {})
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorStoreConfigurationError("Unable to load vector store configuration.") from exc
        if isinstance(tools_config, Mapping):
            vector_block = tools_config.get("vector_store", {})
            if isinstance(vector_block, Mapping):
                adapters_block = vector_block.get("adapters", {})
                if isinstance(adapters_block, Mapping):
                    candidate = adapters_block.get(provider_name, {})
                    if isinstance(candidate, Mapping):
                        settings.update(candidate)
    for key, value in overrides.items():
        settings[key] = value
    return settings


def _create_client(settings: Mapping[str, Any]):
    try:
        import chromadb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise VectorStoreConfigurationError(
            "The 'chromadb' package is required for the Chroma vector store adapter."
        ) from exc

    persist_directory = settings.get("persist_directory")
    if persist_directory:
        client_cls = getattr(chromadb, "PersistentClient", None)
        if client_cls is None:
            raise VectorStoreConfigurationError(
                "Installed chromadb package does not expose PersistentClient."
            )
        return client_cls(path=str(persist_directory))

    server_host = settings.get("server_host")
    if server_host:
        http_client = getattr(chromadb, "HttpClient", None)
        if http_client is None:
            raise VectorStoreConfigurationError(
                "Installed chromadb package does not expose HttpClient."
            )
        kwargs: Dict[str, Any] = {"host": str(server_host)}
        server_port = settings.get("server_port")
        if server_port is not None:
            kwargs["port"] = int(server_port)
        if settings.get("server_ssl"):
            kwargs["ssl"] = True
        return http_client(**kwargs)

    client_factory = getattr(chromadb, "Client", None)
    if callable(client_factory):
        client_kwargs: Dict[str, Any] = {}
        raw_client_settings = settings.get("client_settings")
        if isinstance(raw_client_settings, Mapping):
            try:
                from chromadb.config import Settings  # type: ignore
            except Exception:  # pragma: no cover - fallback when Settings unavailable
                Settings = None  # type: ignore
            if Settings is not None:
                client_kwargs["settings"] = Settings(**dict(raw_client_settings))
        try:
            return client_factory(**client_kwargs)
        except TypeError:
            if "settings" in client_kwargs:
                return client_factory(client_kwargs["settings"])
            return client_factory()  # pragma: no cover - fallback path

    raise VectorStoreConfigurationError(
        "Unable to construct Chroma client. Provide a 'client' override or connection settings."
    )


def _distance_to_score(distance: Optional[float], metric: str) -> float:
    if distance is None:
        return 0.0
    metric_key = metric.lower()
    if metric_key == "cosine":
        return max(0.0, 1.0 - float(distance))
    if metric_key in {"ip", "inner_product"}:
        return float(distance)
    # Treat remaining metrics as distances and convert to similarity.
    return 1.0 / (1.0 + float(distance))


def _normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _resolve_query_field(payload: Mapping[str, Any], field: str) -> Iterable[Any]:
    value = payload.get(field, [])
    if isinstance(value, Iterable):
        return value
    return []


@dataclass
class _CollectionHandle:
    client: Any
    name: str
    metadata: Mapping[str, Any]

    def get(self) -> Any:
        get_collection = getattr(self.client, "get_collection", None)
        if callable(get_collection):
            try:
                return get_collection(name=self.name)
            except Exception:  # pragma: no cover - fall back to creation
                pass
        create = getattr(self.client, "get_or_create_collection", None)
        if not callable(create):
            raise VectorStoreConfigurationError("Chroma client does not support collection retrieval.")
        return create(name=self.name, metadata=self.metadata or None)

    def delete(self) -> bool:
        delete_collection = getattr(self.client, "delete_collection", None)
        if not callable(delete_collection):
            return False
        try:
            delete_collection(name=self.name)
            return True
        except Exception:  # pragma: no cover - tolerate delete failures
            logger.warning("Failed to delete Chroma collection '%s'", self.name, exc_info=True)
            return False


class ChromaVectorStoreAdapter(VectorStoreAdapter):
    """Adapter backed by a Chroma collection per namespace."""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        metric: str = "cosine",
        collection_metadata: Optional[Mapping[str, Any]] = None,
        namespace_separator: str = "__",
    ) -> None:
        if not collection_name.strip():
            raise VectorStoreConfigurationError("Chroma adapter requires a non-empty collection_name.")
        self._client = client
        self._base_collection = collection_name.strip()
        self._metric = metric
        self._collection_metadata = dict(collection_metadata or {})
        self._namespace_separator = namespace_separator
        self._known_ids: Dict[str, set[str]] = {}

    def _collection_for(self, namespace: str) -> _CollectionHandle:
        name = f"{self._base_collection}{self._namespace_separator}{namespace}" if self._namespace_separator else f"{self._base_collection}_{namespace}"
        return _CollectionHandle(self._client, name=name, metadata=self._collection_metadata)

    async def upsert_vectors(self, namespace: str, vectors: tuple[VectorRecord, ...]) -> UpsertResponse:
        handle = self._collection_for(namespace)
        collection = handle.get()
        ids = [record.id for record in vectors]
        embeddings = [[float(value) for value in record.values] for record in vectors]
        metadatas = [dict(record.metadata) if record.metadata else {} for record in vectors]

        upsert = getattr(collection, "upsert", None)
        if not callable(upsert):
            raise VectorStoreOperationError("Chroma collection does not implement 'upsert'.")
        upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        self._known_ids.setdefault(namespace, set()).update(ids)
        return UpsertResponse(namespace=namespace, ids=tuple(ids), upserted_count=len(ids))

    async def query_vectors(
        self,
        namespace: str,
        query: tuple[float, ...],
        *,
        top_k: int,
        metadata_filter: Optional[Mapping[str, Any]],
        include_values: bool,
    ) -> QueryResponse:
        handle = self._collection_for(namespace)
        collection = handle.get()
        include: list[str] = ["metadatas", "distances"]
        if include_values:
            include.append("embeddings")
        where = dict(metadata_filter) if metadata_filter else None
        query_method = getattr(collection, "query", None)
        if not callable(query_method):
            raise VectorStoreOperationError("Chroma collection does not implement 'query'.")

        try:
            result = query_method(
                query_embeddings=[list(query)],
                n_results=top_k,
                where=where,
                include=include,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorStoreOperationError("Chroma query execution failed.") from exc

        matches: list[QueryMatch] = []
        ids_iter = list(_resolve_query_field(result, "ids"))
        distances_iter = list(_resolve_query_field(result, "distances"))
        metadatas_iter = list(_resolve_query_field(result, "metadatas"))
        embeddings_iter = list(_resolve_query_field(result, "embeddings")) if include_values else []

        ids = ids_iter[0] if ids_iter else []
        distances = distances_iter[0] if distances_iter else []
        metadatas = metadatas_iter[0] if metadatas_iter else []
        embeddings = embeddings_iter[0] if embeddings_iter else []

        for index, vector_id in enumerate(ids):
            metadata = _normalize_metadata(metadatas[index] if index < len(metadatas) else None)
            embedding_values: Optional[tuple[float, ...]] = None
            if include_values and index < len(embeddings):
                embedding_values = tuple(float(component) for component in embeddings[index])
            distance = distances[index] if index < len(distances) else None
            score = _distance_to_score(distance, self._metric)
            matches.append(
                QueryMatch(
                    id=str(vector_id),
                    score=score,
                    values=embedding_values,
                    metadata=metadata,
                )
            )

        matches.sort(key=lambda item: (-item.score, item.id))
        limited = tuple(matches[:top_k])
        return QueryResponse(namespace=namespace, matches=limited, top_k=top_k)

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        handle = self._collection_for(namespace)
        deleted = handle.delete()
        removed_ids = tuple(self._known_ids.pop(namespace, set()))
        return DeleteResponse(namespace=namespace, removed_ids=removed_ids, deleted=deleted)


def _adapter_factory(config_manager: Optional[Any], config: Mapping[str, Any]) -> ChromaVectorStoreAdapter:
    settings = _resolve_settings("chroma", config_manager, config)
    client = settings.get("client")
    if client is None:
        client = _create_client(settings)
    collection_name = str(settings.get("collection_name", "")).strip()
    if not collection_name:
        raise VectorStoreConfigurationError("Chroma adapter requires 'collection_name' configuration.")
    metric = str(settings.get("metric", "cosine"))
    metadata = settings.get("collection_metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise VectorStoreConfigurationError("'collection_metadata' must be a mapping when provided.")
    separator = str(settings.get("namespace_separator", "__"))
    return ChromaVectorStoreAdapter(
        client=client,
        collection_name=collection_name,
        metric=metric,
        collection_metadata=metadata if isinstance(metadata, Mapping) else None,
        namespace_separator=separator,
    )


register_vector_store_adapter("chroma", _adapter_factory)

__all__ = ["ChromaVectorStoreAdapter"]
