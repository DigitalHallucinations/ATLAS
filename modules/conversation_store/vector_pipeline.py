"""Background workers and catalog utilities for conversation message vectors."""

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from modules.logging.logger import setup_logger
from modules.Tools.Base_Tools.vector_store import (
    QueryMatch,
    VectorRecord,
    VectorStoreService,
)

from .repository import ConversationStoreRepository
from ._shared import _coerce_uuid, _coerce_vector_payload

logger = setup_logger(__name__)


def _maybe_uuid(value: Any) -> Optional[uuid.UUID]:
    try:
        return _coerce_uuid(value)
    except Exception:  # noqa: BLE001 - best-effort parsing
        return None


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


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    lhs_mag = math.sqrt(sum(a * a for a in lhs))
    rhs_mag = math.sqrt(sum(b * b for b in rhs))
    if lhs_mag == 0.0 or rhs_mag == 0.0:
        return 0.0
    return dot / (lhs_mag * rhs_mag)


class ConversationVectorCatalog:
    """Adapter that exposes persistent message vectors to the vector store."""

    def __init__(self, repository: ConversationStoreRepository) -> None:
        self._repository = repository

    async def hydrate(self) -> Mapping[str, tuple[VectorRecord, ...]]:
        def _load() -> Mapping[str, tuple[VectorRecord, ...]]:
            stored = self._repository.fetch_message_vectors()
            grouped: Dict[str, list[VectorRecord]] = {}
            for record in stored:
                embedding = record.get("embedding") or []
                if not embedding:
                    continue
                metadata = dict(record.get("metadata") or {})
                metadata.setdefault("conversation_id", record["conversation_id"])
                metadata.setdefault("message_id", record["message_id"])
                metadata.setdefault("vector_key", record["vector_key"])
                namespace = metadata.get("namespace") or record["conversation_id"]
                vector = VectorRecord(
                    id=record["vector_key"],
                    values=tuple(float(component) for component in embedding),
                    metadata=metadata,
                )
                grouped.setdefault(str(namespace), []).append(vector)
            return {namespace: tuple(vectors) for namespace, vectors in grouped.items()}

        return await asyncio.to_thread(_load)

    async def upsert(
        self,
        namespace: str,
        vectors: Sequence[VectorRecord],
    ) -> tuple[VectorRecord, ...]:
        def _persist() -> tuple[VectorRecord, ...]:
            grouped: Dict[uuid.UUID, list[Dict[str, Any]]] = {}
            for record in vectors:
                metadata = dict(record.metadata)
                metadata.setdefault("namespace", namespace)
                message_id = metadata.get("message_id")
                if message_id is None:
                    continue
                try:
                    message_uuid = _coerce_uuid(message_id)
                except Exception:  # noqa: BLE001 - skip malformed identifiers
                    continue
                payload = {
                    "values": list(record.values),
                    "provider": metadata.get("provider"),
                    "model": metadata.get("model") or metadata.get("embedding_model"),
                    "model_version": metadata.get("model_version")
                    or metadata.get("embedding_model_version"),
                    "checksum": metadata.get("checksum")
                    or metadata.get("embedding_checksum"),
                    "vector_key": record.id,
                    "metadata": metadata,
                }
                grouped.setdefault(message_uuid, []).append(payload)

            persisted: list[VectorRecord] = []
            for message_uuid, payloads in grouped.items():
                stored = self._repository.upsert_message_vectors(message_uuid, payloads)
                for item in stored:
                    embedding = item.get("embedding") or []
                    if not embedding:
                        continue
                    metadata = dict(item.get("metadata") or {})
                    metadata.setdefault("namespace", namespace)
                    persisted.append(
                        VectorRecord(
                            id=item["vector_key"],
                            values=tuple(float(component) for component in embedding),
                            metadata=metadata,
                        )
                    )
            return tuple(persisted)

        return await asyncio.to_thread(_persist)

    async def query(
        self,
        namespace: str,
        query: Sequence[float],
        *,
        metadata_filter: Optional[Mapping[str, Any]],
        top_k: int,
    ) -> tuple[QueryMatch, ...]:
        def _query() -> tuple[QueryMatch, ...]:
            filters: Dict[str, Any] = {}
            conversation_uuid = _maybe_uuid(namespace)
            if conversation_uuid is not None:
                filters["conversation_id"] = conversation_uuid
            stored = self._repository.fetch_message_vectors(**filters)
            matches: list[QueryMatch] = []
            for record in stored:
                metadata = dict(record.get("metadata") or {})
                record_namespace = metadata.get("namespace") or record["conversation_id"]
                if conversation_uuid is None and str(record_namespace) != str(namespace):
                    continue
                if metadata_filter and not _matches_filter(metadata, metadata_filter):
                    continue
                embedding = record.get("embedding") or []
                if not embedding:
                    continue
                values = tuple(float(component) for component in embedding)
                score = _cosine_similarity(values, query)
                matches.append(
                    QueryMatch(
                        id=record["vector_key"],
                        score=score,
                        values=values,
                        metadata=metadata,
                    )
                )
            matches.sort(key=lambda item: (-item.score, item.id))
            return tuple(matches[:top_k])

        return await asyncio.to_thread(_query)

    async def delete_namespace(self, namespace: str) -> None:
        def _delete() -> None:
            conversation_uuid = _maybe_uuid(namespace)
            if conversation_uuid is None:
                return
            conversation = self._repository.get_conversation(conversation_uuid)
            if conversation is None:
                return
            tenant_id = conversation.get("tenant_id")
            if not tenant_id:
                return
            self._repository.delete_message_vectors(
                tenant_id=tenant_id,
                conversation_id=conversation_uuid,
            )

        await asyncio.to_thread(_delete)


@dataclass(frozen=True)
class _PipelineEvent:
    message: Mapping[str, Any]
    vectors: Optional[Sequence[Mapping[str, Any]]]


class ConversationVectorPipeline:
    """Background worker that upserts embeddings for conversation messages."""

    def __init__(
        self,
        repository: ConversationStoreRepository,
        vector_service: VectorStoreService,
        *,
        embedder: Optional[Callable[[Mapping[str, Any]], Awaitable[Any]]] = None,
    ) -> None:
        self._repository = repository
        self._vector_service = vector_service
        self._embedder = embedder
        self._queue: asyncio.Queue[Optional[_PipelineEvent]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        await self._vector_service.hydrate()
        self._worker_task = asyncio.create_task(
            self._worker(), name="conversation-vector-pipeline"
        )

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    async def wait_for_idle(self) -> None:
        await self._queue.join()

    async def enqueue_message(
        self,
        message: Mapping[str, Any],
        *,
        vectors: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        normalized_message = dict(message)
        normalized_vectors = None
        if vectors is not None:
            normalized_vectors = tuple(dict(vector) for vector in vectors)
        await self._queue.put(_PipelineEvent(normalized_message, normalized_vectors))

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                self._queue.task_done()
                break
            try:
                await self._process_event(event)
            except Exception:  # noqa: BLE001 - log and continue processing
                logger.exception("Failed to process conversation vector event")
            finally:
                self._queue.task_done()

    async def _process_event(self, event: _PipelineEvent) -> None:
        message_id = event.message.get("id") or event.message.get("message_id")
        conversation_id = event.message.get("conversation_id")
        if not message_id or not conversation_id:
            logger.warning(
                "Vector pipeline received message without identifiers: %s", event.message
            )
            return

        try:
            message_uuid = _coerce_uuid(message_id)
            conversation_uuid = _coerce_uuid(conversation_id)
        except Exception as exc:  # noqa: BLE001 - report malformed identifiers
            logger.warning("Unable to parse message identifiers: %s", exc)
            return

        vectors = await self._resolve_vectors(event, message_uuid, conversation_uuid)
        if not vectors:
            return

        await self._vector_service.upsert_vectors(
            namespace=str(conversation_uuid),
            vectors=vectors,
        )

    async def _resolve_vectors(
        self,
        event: _PipelineEvent,
        message_uuid: uuid.UUID,
        conversation_uuid: uuid.UUID,
    ) -> Sequence[Mapping[str, Any]]:
        if event.vectors is not None:
            candidates = list(event.vectors)
        elif self._embedder is not None:
            generated = await self._embedder(event.message)
            candidates = self._coerce_embedder_output(generated)
        else:
            return []

        normalized: list[Dict[str, Any]] = []
        for raw in candidates:
            payload = self._normalize_vector_dict(raw, message_uuid, conversation_uuid)
            if payload is not None:
                normalized.append(payload)
        return normalized

    def _coerce_embedder_output(self, payload: Any) -> Sequence[Mapping[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, Mapping):
            return [dict(payload)]
        if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
            if not payload:
                return []
            first = payload[0]
            if isinstance(first, Mapping):
                return [dict(item) for item in payload]
            if isinstance(first, Sequence) and not isinstance(first, (bytes, bytearray, str)):
                return [
                    {"values": [float(component) for component in entry]}
                    for entry in payload
                ]
            return [{"values": [float(component) for component in payload]}]
        raise TypeError("Embedder output must be a mapping or sequence of mappings.")

    def _normalize_vector_dict(
        self,
        raw: Mapping[str, Any],
        message_uuid: uuid.UUID,
        conversation_uuid: uuid.UUID,
    ) -> Optional[Dict[str, Any]]:
        try:
            coerced = _coerce_vector_payload(message_uuid, conversation_uuid, raw)
        except ValueError:
            return None

        return {
            "id": coerced.vector_key,
            "values": list(coerced.embedding),
            "metadata": dict(coerced.metadata),
        }


__all__ = [
    "ConversationVectorCatalog",
    "ConversationVectorPipeline",
]
