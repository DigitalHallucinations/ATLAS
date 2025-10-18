"""In-memory vector store adapter suitable for development and testing."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from modules.logging.logger import setup_logger
from modules.Tools.Base_Tools.vector_store import (
    DeleteResponse,
    QueryMatch,
    QueryResponse,
    UpsertResponse,
    VectorRecord,
    VectorStoreAdapter,
    VectorStoreOperationError,
    VectorValidationError,
    build_vector_store_service,
    register_vector_store_adapter,
)
from modules.Tools.providers.base import ToolProvider, ToolProviderSpec
from modules.Tools.providers.registry import tool_provider_registry


logger = setup_logger(__name__)


class InMemoryVectorStoreAdapter(VectorStoreAdapter):
    """Simple adapter backed by process-local dictionaries."""

    _INDICES: Dict[str, Dict[str, Dict[str, VectorRecord]]] = {}

    def __init__(self, *, index_name: str = "default") -> None:
        self._index_name = index_name or "default"
        self._index = self._INDICES.setdefault(self._index_name, {})

    async def upsert_vectors(self, namespace: str, vectors: tuple[VectorRecord, ...]) -> UpsertResponse:
        stored = self._index.setdefault(namespace, {})
        for record in vectors:
            stored[record.id] = record
        logger.debug(
            "Upserted %d vectors into namespace '%s' (index=%s)",
            len(vectors),
            namespace,
            self._index_name,
        )
        ids = tuple(record.id for record in vectors)
        return UpsertResponse(namespace=namespace, ids=ids, upserted_count=len(vectors))

    async def query_vectors(
        self,
        namespace: str,
        query: tuple[float, ...],
        *,
        top_k: int,
        metadata_filter: Optional[Mapping[str, Any]],
        include_values: bool,
    ) -> QueryResponse:
        namespace_store = self._index.get(namespace, {})
        if not namespace_store:
            return QueryResponse(namespace=namespace, matches=tuple(), top_k=top_k)

        matches: list[QueryMatch] = []
        for record in namespace_store.values():
            if metadata_filter and not _matches_filter(record.metadata, metadata_filter):
                continue
            score = _cosine_similarity(record.values, query)
            matches.append(
                QueryMatch(
                    id=record.id,
                    score=score,
                    values=record.values,
                    metadata=record.metadata,
                )
            )

        matches.sort(key=lambda item: (-item.score, item.id))
        limited = tuple(matches[:top_k])
        return QueryResponse(namespace=namespace, matches=limited, top_k=top_k)

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        namespace_store = self._index.pop(namespace, None)
        if not namespace_store:
            return DeleteResponse(namespace=namespace, removed_ids=tuple(), deleted=False)
        removed_ids = tuple(namespace_store.keys())
        logger.debug(
            "Deleted namespace '%s' from index '%s' containing %d vectors",
            namespace,
            self._index_name,
            len(removed_ids),
        )
        return DeleteResponse(namespace=namespace, removed_ids=removed_ids, deleted=True)


def _matches_filter(metadata: Mapping[str, Any], filter_map: Mapping[str, Any]) -> bool:
    for key, expected in filter_map.items():
        if key not in metadata:
            return False
        candidate = metadata[key]
        if isinstance(expected, Mapping) and isinstance(candidate, Mapping):
            if not _matches_filter(candidate, expected):
                return False
            continue
        if expected != candidate:
            return False
    return True


def _cosine_similarity(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    if len(lhs) != len(rhs):
        raise VectorValidationError("Query vector dimensionality mismatch.")
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_mag = math.sqrt(sum(a * a for a in lhs))
    rhs_mag = math.sqrt(sum(b * b for b in rhs))
    if lhs_mag == 0 or rhs_mag == 0:
        return 0.0
    return dot / (lhs_mag * rhs_mag)


def _adapter_factory(_config_manager: Optional[Any], config: Mapping[str, Any]) -> InMemoryVectorStoreAdapter:
    index_name = str(config.get("index_name", "default")).strip() or "default"
    return InMemoryVectorStoreAdapter(index_name=index_name)


register_vector_store_adapter("in_memory", _adapter_factory)


class InMemoryVectorStoreProvider(ToolProvider):
    """Provider wrapping :class:`InMemoryVectorStoreAdapter`."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        raw_adapter_name = self.config.get("adapter", "in_memory")
        adapter_name = str(raw_adapter_name).strip() or "in_memory"
        adapter_config = self.config.get("adapter_config")
        if adapter_config is not None and not isinstance(adapter_config, Mapping):
            raise VectorStoreOperationError(
                "'adapter_config' must be a mapping when provided for the in-memory vector store."
            )

        if isinstance(adapter_config, Mapping):
            adapter_config = dict(adapter_config)

        self._service = build_vector_store_service(
            adapter_name=adapter_name,
            adapter_config=adapter_config,
            config_manager=None,
        )

    async def call(self, **kwargs: Any) -> Mapping[str, Any]:
        if self.tool_name == "upsert_vectors":
            return await self._service.upsert_vectors(
                namespace=kwargs.get("namespace", ""),
                vectors=kwargs.get("vectors", ()),
            )
        if self.tool_name == "query_vectors":
            return await self._service.query_vectors(
                namespace=kwargs.get("namespace", ""),
                query=kwargs.get("query", ()),
                top_k=kwargs.get("top_k", 5),
                filter=kwargs.get("filter"),
                include_values=kwargs.get("include_values", False),
            )
        if self.tool_name == "delete_namespace":
            return await self._service.delete_namespace(namespace=kwargs.get("namespace", ""))
        raise VectorStoreOperationError(
            f"Unsupported vector store operation '{self.tool_name}' for in-memory provider."
        )


tool_provider_registry.register("vector_store_in_memory", InMemoryVectorStoreProvider)


__all__ = [
    "InMemoryVectorStoreAdapter",
    "InMemoryVectorStoreProvider",
]

