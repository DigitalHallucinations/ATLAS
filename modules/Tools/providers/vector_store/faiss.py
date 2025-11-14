"""FAISS backed vector store adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class _NamespaceState:
    vectors: Dict[str, VectorRecord] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, record in self.vectors.items():
            payload[key] = {
                "id": record.id,
                "values": list(record.values),
                "metadata": dict(record.metadata),
            }
        return payload

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> "_NamespaceState":
        state = cls()
        for key, value in payload.items():
            if not isinstance(value, Mapping):
                continue
            vector_id = str(value.get("id", key))
            raw_values = value.get("values", [])
            try:
                values = tuple(float(component) for component in raw_values)
            except Exception:  # pragma: no cover - defensive
                continue
            metadata = value.get("metadata")
            if isinstance(metadata, Mapping):
                metadata_dict = dict(metadata)
            else:
                metadata_dict = {}
            state.vectors[vector_id] = VectorRecord(
                id=vector_id,
                values=values,
                metadata=metadata_dict,
            )
        return state


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


def _matches_filter(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in candidate:
            return False
        candidate_value = candidate[key]
        if isinstance(value, Mapping) and isinstance(candidate_value, Mapping):
            if not _matches_filter(candidate_value, value):
                return False
            continue
        if candidate_value != value:
            return False
    return True


def _cosine_score(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    return float(dot / (lhs_norm * rhs_norm))


def _l2_score(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    distance = math.sqrt(sum((a - b) * (a - b) for a, b in zip(lhs, rhs)))
    return 1.0 / (1.0 + float(distance))


class FaissVectorStoreAdapter(VectorStoreAdapter):
    """Vector store backed by FAISS (with pure-Python scoring fallback)."""

    def __init__(
        self,
        *,
        index_path: Optional[str] = None,
        metric: str = "cosine",
    ) -> None:
        self._index_path = Path(index_path) if index_path else None
        self._metric = metric.lower()
        if self._metric not in {"cosine", "l2"}:
            raise VectorStoreConfigurationError("FAISS adapter metric must be either 'cosine' or 'l2'.")
        self._namespaces: Dict[str, _NamespaceState] = {}
        if self._index_path is not None:
            self._load_state()

    def _load_state(self) -> None:
        if self._index_path is None or not self._index_path.exists():
            return
        try:
            payload = json.loads(self._index_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load FAISS adapter state: %s", exc)
            return
        if not isinstance(payload, Mapping):
            return
        for namespace, namespace_payload in payload.items():
            if isinstance(namespace_payload, Mapping):
                self._namespaces[str(namespace)] = _NamespaceState.deserialize(namespace_payload)

    def _persist_state(self) -> None:
        if self._index_path is None:
            return
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                namespace: state.serialize() for namespace, state in self._namespaces.items()
            }
            self._index_path.write_text(json.dumps(serializable))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to persist FAISS adapter state: %s", exc)

    def _state_for(self, namespace: str) -> _NamespaceState:
        return self._namespaces.setdefault(namespace, _NamespaceState())

    def _score(self, lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
        if self._metric == "cosine":
            return _cosine_score(lhs, rhs)
        return _l2_score(lhs, rhs)

    async def upsert_vectors(self, namespace: str, vectors: tuple[VectorRecord, ...]) -> UpsertResponse:
        state = self._state_for(namespace)
        for record in vectors:
            state.vectors[record.id] = record
        self._persist_state()
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
        state = self._namespaces.get(namespace)
        if state is None or not state.vectors:
            return QueryResponse(namespace=namespace, matches=tuple(), top_k=top_k)

        query_vector = tuple(float(component) for component in query)
        matches: list[QueryMatch] = []
        for record in state.vectors.values():
            if metadata_filter and not _matches_filter(record.metadata, metadata_filter):
                continue
            record_vector = tuple(float(component) for component in record.values)
            if len(record_vector) != len(query_vector):
                raise VectorStoreOperationError(
                    "FAISS adapter encountered a dimensionality mismatch during query."
                )
            score = self._score(record_vector, query_vector)
            values = tuple(float(component) for component in record.values) if include_values else None
            matches.append(
                QueryMatch(
                    id=record.id,
                    score=score,
                    values=values,
                    metadata=record.metadata,
                )
            )

        matches.sort(key=lambda item: (-item.score, item.id))
        limited = tuple(matches[:top_k])
        return QueryResponse(namespace=namespace, matches=limited, top_k=top_k)

    async def delete_namespace(self, namespace: str) -> DeleteResponse:
        state = self._namespaces.pop(namespace, None)
        if state is None:
            return DeleteResponse(namespace=namespace, removed_ids=tuple(), deleted=False)
        removed_ids = tuple(state.vectors.keys())
        self._persist_state()
        return DeleteResponse(namespace=namespace, removed_ids=removed_ids, deleted=True)


def _adapter_factory(config_manager: Optional[Any], config: Mapping[str, Any]) -> FaissVectorStoreAdapter:
    settings = _resolve_settings("faiss", config_manager, config)
    index_path = settings.get("index_path")
    if index_path is not None:
        index_path = str(index_path)
    metric = str(settings.get("metric", "cosine"))
    return FaissVectorStoreAdapter(index_path=index_path, metric=metric)


register_vector_store_adapter("faiss", _adapter_factory)

__all__ = ["FaissVectorStoreAdapter"]
