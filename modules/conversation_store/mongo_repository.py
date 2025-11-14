"""MongoDB-backed repository helpers for conversation persistence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency for typing hints
    from pymongo.collection import Collection  # type: ignore
    from pymongo.mongo_client import MongoClient  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    Collection = Any  # type: ignore
    MongoClient = Any  # type: ignore


_ASCENDING = 1
_DESCENDING = -1

CollectionLike = Any


def _is_async_collection(collection: CollectionLike) -> bool:
    """Return ``True`` when *collection* is a Motor async collection."""

    module = getattr(collection, "__module__", "")
    return module.startswith("motor.")


@dataclass
class MongoConversationStoreRepository:
    """Lightweight wrapper around MongoDB collections used for conversations."""

    runs: CollectionLike
    messages: CollectionLike
    sessions: CollectionLike
    client: MongoClient | None = None

    def __post_init__(self) -> None:
        self._is_async = any(
            _is_async_collection(collection)
            for collection in (self.runs, self.messages, self.sessions)
        )

    # ------------------------------------------------------------------
    # Construction helpers

    @classmethod
    def from_database(
        cls,
        database: Any,
        *,
        client: MongoClient | None = None,
        runs_collection: str = "runs",
        messages_collection: str = "messages",
        sessions_collection: str = "sessions",
    ) -> "MongoConversationStoreRepository":
        """Build a repository from a database handle."""

        if database is None:
            raise ValueError("database handle is required to build a Mongo repository")

        runs = database.get_collection(runs_collection)
        messages = database.get_collection(messages_collection)
        sessions = database.get_collection(sessions_collection)

        return cls(runs=runs, messages=messages, sessions=sessions, client=client)

    # ------------------------------------------------------------------
    # Index management

    _RUN_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("session_id", _ASCENDING),), {}),
        ((("tenant_id", _ASCENDING), ("session_id", _ASCENDING)), {}),
        ((("status", _ASCENDING), ("created_at", _DESCENDING)), {}),
    )
    _MESSAGE_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("session_id", _ASCENDING), ("created_at", _DESCENDING)), {}),
        ((("conversation_id", _ASCENDING), ("created_at", _DESCENDING)), {}),
        ((("tenant_id", _ASCENDING), ("session_id", _ASCENDING), ("created_at", _DESCENDING)), {}),
        ((("run_id", _ASCENDING),), {}),
    )
    _SESSION_INDEXES: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]] = (
        ((("tenant_id", _ASCENDING), ("external_id", _ASCENDING)), {}),
        ((("tenant_id", _ASCENDING), ("status", _ASCENDING)), {}),
        ((("updated_at", _DESCENDING),), {}),
    )

    def provision_indexes(self) -> Sequence[Any]:
        """Create indexes for the managed collections, returning pending tasks."""

        tasks: list[Any] = []
        tasks.extend(self._ensure_collection_indexes(self.runs, self._RUN_INDEXES, unique=(1,)))
        tasks.extend(self._ensure_collection_indexes(self.messages, self._MESSAGE_INDEXES))
        tasks.extend(
            self._ensure_collection_indexes(
                self.sessions,
                self._SESSION_INDEXES,
                unique=(0,),
            )
        )
        return tuple(tasks)

    def ensure_indexes(self) -> None:
        """Synchronously ensure indexes exist when using PyMongo collections."""

        if self._is_async:
            raise RuntimeError(
                "ensure_indexes requires synchronous collections; use ensure_indexes_async instead"
            )
        self.provision_indexes()

    async def ensure_indexes_async(self) -> None:
        """Asynchronously ensure indexes exist when using Motor collections."""

        pending = self.provision_indexes()
        if not pending:
            return
        await asyncio.gather(*[task for task in pending if asyncio.iscoroutine(task)])

    def _ensure_collection_indexes(
        self,
        collection: CollectionLike,
        definitions: Sequence[tuple[Sequence[tuple[str, int]], Mapping[str, Any]]],
        *,
        unique: Sequence[int] | None = None,
    ) -> Sequence[Any]:
        """Ensure indexes are created for *collection* using *definitions*."""

        if collection is None:
            return ()

        create_index = getattr(collection, "create_index", None)
        if not callable(create_index):
            return ()

        pending: list[Any] = []
        unique_positions = set(unique or ())

        for position, (keys, options) in enumerate(definitions):
            key_spec = list(keys)
            normalized_options: MutableMapping[str, Any] = dict(options)
            if position in unique_positions:
                normalized_options.setdefault("unique", True)
            result = create_index(key_spec, **normalized_options)
            if asyncio.iscoroutine(result):
                pending.append(result)
        return tuple(pending)


__all__ = ["MongoConversationStoreRepository"]
