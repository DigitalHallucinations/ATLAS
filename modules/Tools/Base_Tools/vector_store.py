"""Asynchronous utilities for interacting with pluggable vector stores.

The vector store tool exposes three high-level operations that are expected to
be provider agnostic:

``upsert_vectors``
    Insert or update vector embeddings within a logical namespace.

``query_vectors``
    Retrieve the most similar embeddings for a supplied query vector.

``delete_namespace``
    Remove all embeddings stored under a namespace.

Actual persistence and similarity search behaviour is implemented by adapter
classes registered through :func:`register_vector_store_adapter`.  Providers in
``modules.Tools.providers.vector_store`` are responsible for supplying these
adapters and are typically instantiated by the tool provider router.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

from modules.logging.logger import setup_logger

try:  # ConfigManager is optional in certain test contexts
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - fallback when ConfigManager unavailable
    ConfigManager = None  # type: ignore

logger = setup_logger(__name__)


class VectorStoreError(RuntimeError):
    """Base class for vector store related failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when a vector store adapter cannot be resolved."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when an adapter returns an invalid response."""


class VectorValidationError(VectorStoreError):
    """Raised when user supplied payloads fail validation."""


@dataclass(frozen=True)
class VectorRecord:
    """Normalized representation of a vector embedding."""

    id: str
    values: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:  # pragma: no cover - dataclass enforcement
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class QueryMatch:
    """A single similarity match returned by a vector store."""

    id: str
    score: float
    values: Optional[tuple[float, ...]]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:  # pragma: no cover - dataclass enforcement
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self, *, include_values: bool) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "score": float(self.score),
        }
        if include_values and self.values is not None:
            payload["values"] = [float(value) for value in self.values]
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class UpsertResponse:
    """Adapter response describing stored vector identifiers."""

    namespace: str
    ids: tuple[str, ...]
    upserted_count: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "namespace": self.namespace,
            "ids": list(self.ids),
            "upserted_count": int(self.upserted_count),
        }


@dataclass(frozen=True)
class QueryResponse:
    """Adapter response containing similarity matches."""

    namespace: str
    matches: tuple[QueryMatch, ...]
    top_k: int

    def to_dict(self, *, include_values: bool) -> Mapping[str, Any]:
        return {
            "namespace": self.namespace,
            "top_k": int(self.top_k),
            "matches": [match.to_dict(include_values=include_values) for match in self.matches],
        }


@dataclass(frozen=True)
class DeleteResponse:
    """Adapter response after removing namespace contents."""

    namespace: str
    removed_ids: tuple[str, ...]
    deleted: bool

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "namespace": self.namespace,
            "deleted": bool(self.deleted),
            "removed_ids": list(self.removed_ids),
        }


class VectorStoreAdapter(Protocol):
    """Protocol describing the adapter surface consumed by the tool."""

    async def upsert_vectors(self, namespace: str, vectors: Sequence[VectorRecord]) -> UpsertResponse:
        ...

    async def query_vectors(
        self,
        namespace: str,
        query: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[Mapping[str, Any]],
        include_values: bool,
    ) -> QueryResponse:
        ...

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        ...


class PersistentVectorCatalog(Protocol):
    """Protocol describing optional persistent vector catalogs."""

    async def hydrate(self) -> Mapping[str, Sequence[VectorRecord]]:
        ...

    async def upsert(
        self,
        namespace: str,
        vectors: Sequence[VectorRecord],
    ) -> Sequence[VectorRecord]:
        ...

    async def query(
        self,
        namespace: str,
        query: Sequence[float],
        *,
        metadata_filter: Optional[Mapping[str, Any]],
        top_k: int,
    ) -> Sequence[QueryMatch]:
        ...

    async def delete_namespace(self, namespace: str) -> None:
        ...


AdapterFactory = Callable[[Optional[ConfigManager], Mapping[str, Any]], VectorStoreAdapter]

_ADAPTER_REGISTRY: Dict[str, AdapterFactory] = {}


def register_vector_store_adapter(name: str, factory: AdapterFactory) -> None:
    """Register an adapter factory under ``name``.

    Registration is case-insensitive.  Subsequent registrations under the same
    normalized name replace the prior factory which keeps the API useful for
    tests that monkeypatch adapters.
    """

    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name must be a non-empty string")
    _ADAPTER_REGISTRY[key] = factory
    logger.debug("Registered vector store adapter '%s'", key)


def available_vector_store_adapters() -> tuple[str, ...]:
    """Return the currently registered adapter names."""

    return tuple(sorted(_ADAPTER_REGISTRY))


def create_vector_store_adapter(
    name: str,
    *,
    config_manager: Optional[ConfigManager] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> VectorStoreAdapter:
    """Instantiate an adapter registered under ``name``."""

    key = name.strip().lower()
    factory = _ADAPTER_REGISTRY.get(key)
    if factory is None:
        raise VectorStoreConfigurationError(f"Unknown vector store adapter '{name}'.")

    adapter_config: Mapping[str, Any]
    if isinstance(config, Mapping):
        adapter_config = MappingProxyType(dict(config))
    else:
        adapter_config = MappingProxyType({})

    return factory(config_manager, adapter_config)


def _load_vector_store_settings(
    config_manager: Optional[ConfigManager],
) -> Mapping[str, Any]:
    if ConfigManager is None:
        return MappingProxyType({})

    manager = config_manager
    if manager is None:
        try:
            manager = ConfigManager()  # type: ignore[operator]
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to instantiate ConfigManager: %s", exc)
            return MappingProxyType({})

    if manager is None:  # pragma: no cover - defensive guard
        return MappingProxyType({})

    settings = manager.get_config("tools", {})
    if not isinstance(settings, Mapping):
        return MappingProxyType({})

    vector_settings = settings.get("vector_store", {})
    if not isinstance(vector_settings, Mapping):
        return MappingProxyType({})

    return MappingProxyType(dict(vector_settings))


def build_vector_store_service(
    *,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
    catalog: Optional[PersistentVectorCatalog] = None,
) -> "VectorStoreService":
    """Construct a :class:`VectorStoreService` from configuration."""

    settings = _load_vector_store_settings(config_manager)

    resolved_adapter = adapter_name or settings.get("default_adapter")
    if not isinstance(resolved_adapter, str) or not resolved_adapter.strip():
        raise VectorStoreConfigurationError(
            "No vector store adapter configured. Provide `adapter_name` or set "
            "`tools.vector_store.default_adapter`."
        )

    resolved_config: Mapping[str, Any]
    if isinstance(adapter_config, Mapping):
        resolved_config = MappingProxyType(dict(adapter_config))
    else:
        adapters_block = settings.get("adapters", {})
        if isinstance(adapters_block, Mapping):
            candidate = adapters_block.get(resolved_adapter)
            if isinstance(candidate, Mapping):
                resolved_config = MappingProxyType(dict(candidate))
            else:
                resolved_config = MappingProxyType({})
        else:
            resolved_config = MappingProxyType({})

    adapter = create_vector_store_adapter(
        resolved_adapter,
        config_manager=config_manager,
        config=resolved_config,
    )
    return VectorStoreService(adapter, catalog=catalog)


def _normalize_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise VectorValidationError("Namespace must be a string.")
    normalized = namespace.strip()
    if not normalized:
        raise VectorValidationError("Namespace must be a non-empty string.")
    return normalized


def _normalize_values(values: Sequence[Any]) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise VectorValidationError("Vector values must be a sequence of numbers.")
    normalized: list[float] = []
    for index, value in enumerate(values):
        try:
            normalized.append(float(value))
        except (TypeError, ValueError) as exc:
            raise VectorValidationError(
                f"Vector component at position {index} is not numeric."
            ) from exc
    if not normalized:
        raise VectorValidationError("Vector values must contain at least one element.")
    return tuple(normalized)


def _normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise VectorValidationError("Vector metadata must be a mapping.")

    def _normalize_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _normalize_value(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_normalize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    normalized: Dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        normalized[key_str] = _normalize_value(value)
    return MappingProxyType(normalized)


def _normalize_vector_entry(entry: Mapping[str, Any]) -> VectorRecord:
    if not isinstance(entry, Mapping):
        raise VectorValidationError("Each vector entry must be an object containing 'id' and 'values'.")

    raw_id = entry.get("id")
    if not isinstance(raw_id, str):
        if raw_id is None:
            raise VectorValidationError("Vector entries must include an 'id'.")
        raw_id = str(raw_id)
    vector_id = raw_id.strip()
    if not vector_id:
        raise VectorValidationError("Vector id must be a non-empty string.")

    raw_values = entry.get("values")
    values = _normalize_values(raw_values)

    metadata = _normalize_metadata(entry.get("metadata"))

    return VectorRecord(id=vector_id, values=values, metadata=metadata)


def _normalize_vector_payload(vectors: Sequence[Any]) -> tuple[VectorRecord, ...]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes, bytearray)):
        raise VectorValidationError("'vectors' must be an array of vector objects.")

    normalized: list[VectorRecord] = []
    for entry in vectors:
        if not isinstance(entry, Mapping):
            raise VectorValidationError("Each vector must be represented as an object.")
        normalized.append(_normalize_vector_entry(entry))

    if not normalized:
        raise VectorValidationError("At least one vector is required for upsert operations.")
    return tuple(normalized)


def _normalize_query_vector(query: Sequence[Any]) -> tuple[float, ...]:
    if not isinstance(query, Sequence) or isinstance(query, (str, bytes, bytearray)):
        raise VectorValidationError("'query' must be a sequence of numeric values.")
    return _normalize_values(query)


def _normalize_filter(filter_obj: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if filter_obj is None:
        return MappingProxyType({})
    if not isinstance(filter_obj, Mapping):
        raise VectorValidationError("'filter' must be an object when provided.")
    return MappingProxyType(dict(filter_obj))


def _normalize_top_k(value: Any) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError("'top_k' must be an integer.") from exc
    if top_k <= 0:
        raise VectorValidationError("'top_k' must be greater than zero.")
    return top_k


class VectorStoreService:
    """High level orchestration layer around a :class:`VectorStoreAdapter`."""

    def __init__(
        self,
        adapter: VectorStoreAdapter,
        *,
        catalog: Optional[PersistentVectorCatalog] = None,
    ) -> None:
        self._adapter = adapter
        self._catalog = catalog
        self._hydrated = catalog is None
        self._hydration_lock: Optional[asyncio.Lock] = None

    async def hydrate(self) -> None:
        """Ensure the underlying adapter reflects any persisted catalog state."""

        await self._ensure_hydrated()

    async def _ensure_hydrated(self) -> None:
        if self._catalog is None or self._hydrated:
            return
        if self._hydration_lock is None:
            self._hydration_lock = asyncio.Lock()
        async with self._hydration_lock:
            if self._hydrated:
                return
            namespace_map = await self._catalog.hydrate()
            for namespace, records in namespace_map.items():
                if not records:
                    continue
                await self._adapter.upsert_vectors(namespace, records)
            self._hydrated = True

    async def upsert_vectors(
        self,
        *,
        namespace: str,
        vectors: Sequence[Any],
    ) -> Mapping[str, Any]:
        await self._ensure_hydrated()
        normalized_namespace = _normalize_namespace(namespace)
        normalized_vectors = _normalize_vector_payload(vectors)
        persisted: Sequence[VectorRecord] = ()
        if self._catalog is not None:
            persisted = await self._catalog.upsert(normalized_namespace, normalized_vectors)

        merged: Dict[str, VectorRecord] = {record.id: record for record in normalized_vectors}
        for record in persisted:
            merged[record.id] = record

        final_vectors = tuple(merged.values())

        response = await self._adapter.upsert_vectors(normalized_namespace, final_vectors)
        if not isinstance(response, UpsertResponse):
            raise VectorStoreOperationError("Adapter returned an invalid upsert response.")
        return response.to_dict()

    async def query_vectors(
        self,
        *,
        namespace: str,
        query: Sequence[Any],
        top_k: Any = 5,
        filter: Optional[Mapping[str, Any]] = None,
        include_values: bool = False,
    ) -> Mapping[str, Any]:
        await self._ensure_hydrated()
        normalized_namespace = _normalize_namespace(namespace)
        normalized_query = _normalize_query_vector(query)
        normalized_top_k = _normalize_top_k(top_k)
        normalized_filter = _normalize_filter(filter)

        response = await self._adapter.query_vectors(
            normalized_namespace,
            normalized_query,
            top_k=normalized_top_k,
            metadata_filter=normalized_filter if normalized_filter else None,
            include_values=bool(include_values),
        )
        if not isinstance(response, QueryResponse):
            raise VectorStoreOperationError("Adapter returned an invalid query response.")
        matches = list(response.matches)

        if self._catalog is not None:
            catalog_filter = dict(normalized_filter) if normalized_filter else None
            persisted_matches = await self._catalog.query(
                normalized_namespace,
                normalized_query,
                metadata_filter=catalog_filter,
                top_k=normalized_top_k,
            )

            combined: Dict[str, QueryMatch] = {match.id: match for match in matches}
            for persisted in persisted_matches:
                existing = combined.get(persisted.id)
                if existing is None or persisted.score > existing.score:
                    combined[persisted.id] = persisted

            sorted_matches = sorted(
                combined.values(), key=lambda item: (-item.score, item.id)
            )
            limited_matches = tuple(sorted_matches[:normalized_top_k])
        else:
            limited_matches = tuple(matches[:normalized_top_k])

        merged_response = QueryResponse(
            namespace=response.namespace,
            matches=limited_matches,
            top_k=normalized_top_k,
        )
        return merged_response.to_dict(include_values=bool(include_values))

    async def delete_namespace(self, *, namespace: str) -> Mapping[str, Any]:
        await self._ensure_hydrated()
        normalized_namespace = _normalize_namespace(namespace)
        if self._catalog is not None:
            await self._catalog.delete_namespace(normalized_namespace)
        response = await self._adapter.delete_namespace(normalized_namespace)
        if not isinstance(response, DeleteResponse):
            raise VectorStoreOperationError("Adapter returned an invalid delete response.")
        return response.to_dict()


def _resolve_service(
    *,
    adapter: Optional[VectorStoreAdapter],
    adapter_name: Optional[str],
    adapter_config: Optional[Mapping[str, Any]],
    config_manager: Optional[ConfigManager],
    catalog: Optional[PersistentVectorCatalog],
) -> VectorStoreService:
    if adapter is not None:
        return VectorStoreService(adapter, catalog=catalog)

    return build_vector_store_service(
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
        catalog=catalog,
    )


async def upsert_vectors(
    *,
    namespace: str,
    vectors: Sequence[Any],
    adapter: Optional[VectorStoreAdapter] = None,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
    catalog: Optional[PersistentVectorCatalog] = None,
) -> Mapping[str, Any]:
    """Insert or update embeddings within ``namespace``."""

    service = _resolve_service(
        adapter=adapter,
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
        catalog=catalog,
    )
    return await service.upsert_vectors(namespace=namespace, vectors=vectors)


async def query_vectors(
    *,
    namespace: str,
    query: Sequence[Any],
    top_k: Any = 5,
    filter: Optional[Mapping[str, Any]] = None,
    include_values: bool = False,
    adapter: Optional[VectorStoreAdapter] = None,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
    catalog: Optional[PersistentVectorCatalog] = None,
) -> Mapping[str, Any]:
    """Return the most similar embeddings for ``query`` within ``namespace``."""

    service = _resolve_service(
        adapter=adapter,
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
        catalog=catalog,
    )
    return await service.query_vectors(
        namespace=namespace,
        query=query,
        top_k=top_k,
        filter=filter,
        include_values=include_values,
    )


async def delete_namespace(
    *,
    namespace: str,
    adapter: Optional[VectorStoreAdapter] = None,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
    catalog: Optional[PersistentVectorCatalog] = None,
) -> Mapping[str, Any]:
    """Delete all embeddings stored within ``namespace``."""

    service = _resolve_service(
        adapter=adapter,
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
        catalog=catalog,
    )
    return await service.delete_namespace(namespace=namespace)


__all__ = [
    "VectorStoreError",
    "VectorStoreConfigurationError",
    "VectorStoreOperationError",
    "VectorValidationError",
    "VectorRecord",
    "QueryMatch",
    "UpsertResponse",
    "QueryResponse",
    "DeleteResponse",
    "VectorStoreAdapter",
    "PersistentVectorCatalog",
    "VectorStoreService",
    "register_vector_store_adapter",
    "available_vector_store_adapters",
    "create_vector_store_adapter",
    "build_vector_store_service",
    "upsert_vectors",
    "query_vectors",
    "delete_namespace",
]

