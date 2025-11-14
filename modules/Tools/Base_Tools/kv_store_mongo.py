"""MongoDB-backed key-value store adapter for the KV service."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - dependency optional in some environments
    from pymongo.collection import Collection  # type: ignore
    from pymongo.mongo_client import MongoClient  # type: ignore
except Exception:  # pragma: no cover - provide lightweight fallbacks when PyMongo absent
    Collection = Any  # type: ignore
    MongoClient = Any  # type: ignore

from .kv_store import (
    GlobalQuotaExceededError,
    InvalidValueTypeError,
    KeyValueStoreAdapter,
    KeyValueStoreError,
    NamespaceQuotaExceededError,
    ValueSerializationError,
    _DeleteResult,
    _GetResult,
    _StoreRecord,
    _WriteResult,
)


CollectionLike = Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class MongoKeyValueStoreAdapter(KeyValueStoreAdapter):
    """Key-value store adapter backed by a MongoDB collection."""

    collection: CollectionLike
    namespace_quota_bytes: int
    global_quota_bytes: Optional[int] = None
    client: MongoClient | None = None

    def __post_init__(self) -> None:
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Construction helpers

    @classmethod
    def from_database(
        cls,
        database: Any,
        *,
        client: MongoClient | None = None,
        collection_name: str = "kv_store",
        namespace_quota_bytes: int,
        global_quota_bytes: Optional[int],
    ) -> "MongoKeyValueStoreAdapter":
        if database is None:
            raise ValueError("database handle is required to build a Mongo adapter")
        collection = database.get_collection(collection_name)
        return cls(
            collection=collection,
            namespace_quota_bytes=namespace_quota_bytes,
            global_quota_bytes=global_quota_bytes,
            client=client,
        )

    # ------------------------------------------------------------------
    # Index helpers

    def _ensure_indexes(self) -> None:
        if self.collection is None:
            return
        create_index = getattr(self.collection, "create_index", None)
        if not callable(create_index):
            return
        create_index([("namespace", 1), ("key", 1)], unique=True)
        try:
            create_index([("expires_at", 1)], expireAfterSeconds=0)
        except Exception:  # pragma: no cover - TTL index unsupported
            pass

    # ------------------------------------------------------------------
    # Internal helpers

    def _purge_expired(self) -> None:
        now = _utc_now()
        self.collection.delete_many({"expires_at": {"$lte": now}})

    def _serialize_value(self, value: Any) -> tuple[Any, int]:
        import json

        try:
            encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueSerializationError("Values must be JSON serializable.") from exc
        size = len(encoded.encode("utf-8"))
        return json.loads(encoded), size

    def _normalize_ttl(self, ttl_seconds: Optional[float]) -> Optional[float]:
        if ttl_seconds is None:
            return None
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise KeyValueStoreError("TTL must be numeric") from exc
        if ttl <= 0:
            raise KeyValueStoreError("TTL must be greater than zero")
        return ttl

    def _compute_usage(self, namespace: Optional[str]) -> int:
        pipeline: list[Mapping[str, Any]] = []
        match: Dict[str, Any] = {}
        if namespace is not None:
            match["namespace"] = namespace
        match["$or"] = [{"expires_at": None}, {"expires_at": {"$gt": _utc_now()}}]
        pipeline.append({"$match": match})
        pipeline.append({"$group": {"_id": None, "usage": {"$sum": "$size_bytes"}}})
        result = list(self.collection.aggregate(pipeline))
        if not result:
            return 0
        return int(result[0].get("usage", 0))

    def _enforce_quotas(
        self,
        *,
        namespace: str,
        new_size: int,
        previous_size: int,
    ) -> None:
        namespace_usage = self._compute_usage(namespace) - previous_size
        if namespace_usage + new_size > self.namespace_quota_bytes:
            raise NamespaceQuotaExceededError(
                f"Storing '{namespace}:{new_size}' would exceed namespace quota of {self.namespace_quota_bytes} bytes."
            )
        if self.global_quota_bytes is not None:
            global_usage = self._compute_usage(None) - previous_size
            if global_usage + new_size > self.global_quota_bytes:
                raise GlobalQuotaExceededError("Global key-value store quota exceeded.")

    def _load_record(self, namespace: str, key: str) -> Mapping[str, Any] | None:
        return self.collection.find_one({"namespace": namespace, "key": key})

    def _store_document(
        self,
        *,
        namespace: str,
        key: str,
        value: Any,
        expires_at: Optional[datetime],
        size: int,
        previous: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        now = _utc_now()
        document = {
            "namespace": namespace,
            "key": key,
            "value": value,
            "expires_at": expires_at,
            "size_bytes": size,
            "updated_at": now,
        }
        if previous is None:
            document["_id"] = _generate_uuid()
            document["created_at"] = now
        else:
            document["_id"] = previous.get("_id") or _generate_uuid()
            document["created_at"] = previous.get("created_at", now)

        self.collection.replace_one({"_id": document["_id"]}, document, upsert=True)
        return document

    # ------------------------------------------------------------------
    # Adapter interface

    def get(self, namespace: str, key: str) -> _GetResult:
        self._purge_expired()
        record = self._load_record(namespace, key)
        if record is None:
            return _GetResult(namespace=namespace, key=key, found=False, record=None)
        expires_at = record.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= _utc_now():
            self.collection.delete_one({"_id": record.get("_id")})
            return _GetResult(namespace=namespace, key=key, found=False, record=None)
        stored = _StoreRecord(
            namespace=namespace,
            key=key,
            value=record.get("value"),
            expires_at=expires_at.timestamp() if isinstance(expires_at, datetime) else None,
        )
        return _GetResult(namespace=namespace, key=key, found=True, record=stored)

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: Optional[float]) -> _WriteResult:
        ttl = self._normalize_ttl(ttl_seconds)
        serialized, size = self._serialize_value(value)
        expires_at = (
            datetime.fromtimestamp(time.time() + ttl, tz=timezone.utc) if ttl is not None else None
        )

        self._purge_expired()
        existing = self._load_record(namespace, key)
        previous_size = int(existing.get("size_bytes", 0)) if existing else 0
        self._enforce_quotas(namespace=namespace, new_size=size, previous_size=previous_size)
        document = self._store_document(
            namespace=namespace,
            key=key,
            value=serialized,
            expires_at=expires_at,
            size=size,
            previous=existing,
        )
        record = _StoreRecord(
            namespace=namespace,
            key=key,
            value=value,
            expires_at=expires_at.timestamp() if expires_at else None,
        )
        return _WriteResult(namespace=namespace, key=key, record=record)

    def delete(self, namespace: str, key: str) -> _DeleteResult:
        self._purge_expired()
        result = self.collection.delete_one({"namespace": namespace, "key": key})
        deleted = getattr(result, "deleted_count", 0) > 0
        return _DeleteResult(namespace=namespace, key=key, deleted=deleted)

    def increment(
        self,
        namespace: str,
        key: str,
        *,
        delta: int,
        ttl_seconds: Optional[float],
        initial_value: int,
    ) -> _WriteResult:
        if not isinstance(delta, int):
            raise InvalidValueTypeError("delta must be an integer")
        if not isinstance(initial_value, int):
            raise InvalidValueTypeError("initial_value must be an integer")

        ttl = self._normalize_ttl(ttl_seconds) if ttl_seconds is not None else None
        expires_at = (
            datetime.fromtimestamp(time.time() + ttl, tz=timezone.utc) if ttl is not None else None
        )

        self._purge_expired()
        existing = self._load_record(namespace, key)
        if existing is None:
            current_value = initial_value
            previous_size = 0
        else:
            current = existing.get("value")
            if not isinstance(current, int):
                raise InvalidValueTypeError("Stored value is not an integer")
            current_value = current
            previous_size = int(existing.get("size_bytes", 0))
            if ttl is None:
                expires_at = existing.get("expires_at")

        new_value = current_value + delta
        serialized, size = self._serialize_value(new_value)
        self._enforce_quotas(namespace=namespace, new_size=size, previous_size=previous_size)

        document = self._store_document(
            namespace=namespace,
            key=key,
            value=serialized,
            expires_at=expires_at,
            size=size,
            previous=existing,
        )
        record = _StoreRecord(
            namespace=namespace,
            key=key,
            value=new_value,
            expires_at=document.get("expires_at").timestamp() if document.get("expires_at") else None,
        )
        return _WriteResult(namespace=namespace, key=key, record=record)


__all__ = ["MongoKeyValueStoreAdapter"]

