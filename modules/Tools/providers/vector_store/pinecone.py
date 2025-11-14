"""Pinecone vector store adapter."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

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
        import pinecone  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise VectorStoreConfigurationError(
            "The 'pinecone-client' package is required for the Pinecone adapter."
        ) from exc

    api_key = settings.get("api_key")
    if not api_key:
        raise VectorStoreConfigurationError("Pinecone adapter requires an 'api_key'.")

    if hasattr(pinecone, "Pinecone"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        environment = settings.get("environment")
        if environment:
            kwargs["environment"] = environment
        host = settings.get("host")
        if host:
            kwargs["host"] = host
        project = settings.get("project_name")
        if project:
            kwargs["project_name"] = project
        return pinecone.Pinecone(**kwargs)

    init = getattr(pinecone, "init", None)
    if callable(init):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        environment = settings.get("environment")
        if environment:
            kwargs["environment"] = environment
        host = settings.get("host")
        if host:
            kwargs["host"] = host
        init(**kwargs)
        return pinecone

    raise VectorStoreConfigurationError("Unable to construct Pinecone client from installed package.")


def _normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _iter_matches(response: Any) -> Iterable[Any]:
    if response is None:
        return []
    if hasattr(response, "matches"):
        matches = getattr(response, "matches")
        if matches is not None:
            return matches
    if isinstance(response, Mapping):
        matches = response.get("matches")
        if matches is not None:
            return matches
    return []


class PineconeVectorStoreAdapter(VectorStoreAdapter):
    """Adapter for Pinecone namespaces."""

    def __init__(
        self,
        *,
        index_name: str,
        client: Optional[Any] = None,
        index_factory: Optional[Any] = None,
        namespace_prefix: str = "",
    ) -> None:
        if not index_name.strip():
            raise VectorStoreConfigurationError("Pinecone adapter requires an index_name.")
        self._index_name = index_name.strip()
        self._client = client
        self._index_factory = index_factory
        self._namespace_prefix = namespace_prefix.strip()
        self._index_singleton: Optional[Any] = None
        self._known_ids: Dict[str, set[str]] = {}

    def _full_namespace(self, namespace: str) -> str:
        if not self._namespace_prefix:
            return namespace
        return f"{self._namespace_prefix}{namespace}"

    def _get_index(self) -> Any:
        if self._index_singleton is not None:
            return self._index_singleton
        if callable(self._index_factory):
            self._index_singleton = self._index_factory()
            return self._index_singleton
        if self._client is None:
            raise VectorStoreConfigurationError("Pinecone client is not configured.")
        index_getter = getattr(self._client, "Index", None)
        if callable(index_getter):
            self._index_singleton = index_getter(self._index_name)
            return self._index_singleton
        index_getter = getattr(self._client, "index", None)
        if callable(index_getter):
            self._index_singleton = index_getter(self._index_name)
            return self._index_singleton
        raise VectorStoreConfigurationError("Unable to resolve Pinecone index factory.")

    async def upsert_vectors(self, namespace: str, vectors: tuple[VectorRecord, ...]) -> UpsertResponse:
        index = self._get_index()
        namespace_key = self._full_namespace(namespace)
        payload = []
        for record in vectors:
            entry: Dict[str, Any] = {"id": record.id, "values": [float(v) for v in record.values]}
            if record.metadata:
                entry["metadata"] = dict(record.metadata)
            payload.append(entry)
        upsert = getattr(index, "upsert", None)
        if not callable(upsert):
            raise VectorStoreOperationError("Pinecone index does not expose 'upsert'.")
        upsert(vectors=payload, namespace=namespace_key)
        self._known_ids.setdefault(namespace, set()).update(record.id for record in vectors)
        return UpsertResponse(namespace=namespace, ids=tuple(record.id for record in vectors), upserted_count=len(vectors))

    async def query_vectors(
        self,
        namespace: str,
        query: tuple[float, ...],
        *,
        top_k: int,
        metadata_filter: Optional[Mapping[str, Any]],
        include_values: bool,
    ) -> QueryResponse:
        index = self._get_index()
        namespace_key = self._full_namespace(namespace)
        query_fn = getattr(index, "query", None)
        if not callable(query_fn):
            raise VectorStoreOperationError("Pinecone index does not expose 'query'.")
        try:
            response = query_fn(
                namespace=namespace_key,
                vector=[float(v) for v in query],
                top_k=top_k,
                filter=dict(metadata_filter) if metadata_filter else None,
                include_values=include_values,
                include_metadata=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise VectorStoreOperationError("Pinecone query execution failed.") from exc

        matches: list[QueryMatch] = []
        for match in _iter_matches(response):
            if isinstance(match, Mapping):
                vector_id = str(match.get("id", ""))
                score = float(match.get("score", 0.0))
                metadata = _normalize_metadata(match.get("metadata"))
                raw_values = match.get("values") if include_values else None
            else:
                vector_id = str(getattr(match, "id", ""))
                score = float(getattr(match, "score", 0.0))
                metadata = _normalize_metadata(getattr(match, "metadata", {}))
                raw_values = getattr(match, "values", None) if include_values else None
            values = None
            if include_values and raw_values is not None:
                try:
                    values = tuple(float(component) for component in raw_values)
                except Exception:  # pragma: no cover - defensive
                    values = None
            matches.append(
                QueryMatch(
                    id=vector_id,
                    score=score,
                    values=values,
                    metadata=metadata,
                )
            )

        matches.sort(key=lambda item: (-item.score, item.id))
        limited = tuple(matches[:top_k])
        return QueryResponse(namespace=namespace, matches=limited, top_k=top_k)

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        index = self._get_index()
        namespace_key = self._full_namespace(namespace)
        delete_fn = getattr(index, "delete", None)
        if not callable(delete_fn):
            raise VectorStoreOperationError("Pinecone index does not expose 'delete'.")
        delete_fn(namespace=namespace_key, delete_all=True)
        removed_ids = tuple(self._known_ids.pop(namespace, set()))
        deleted = bool(removed_ids)
        return DeleteResponse(namespace=namespace, removed_ids=removed_ids, deleted=deleted)


def _adapter_factory(config_manager: Optional[Any], config: Mapping[str, Any]) -> PineconeVectorStoreAdapter:
    settings = _resolve_settings("pinecone", config_manager, config)
    index_name = str(settings.get("index_name", "")).strip()
    if not index_name:
        raise VectorStoreConfigurationError("Pinecone adapter requires 'index_name'.")
    namespace_prefix = str(settings.get("namespace_prefix", ""))
    index_override = settings.get("index")
    index_factory = None
    if callable(index_override):
        index_factory = index_override
        client = settings.get("client")
    else:
        client = settings.get("client")
        if client is None:
            client = _create_client(settings)
        elif not hasattr(client, "Index") and not hasattr(client, "index"):
            raise VectorStoreConfigurationError(
                "Provided Pinecone client override does not expose an index factory."
            )
    return PineconeVectorStoreAdapter(
        index_name=index_name,
        client=client,
        index_factory=index_factory,
        namespace_prefix=namespace_prefix,
    )


register_vector_store_adapter("pinecone", _adapter_factory)

__all__ = ["PineconeVectorStoreAdapter"]
