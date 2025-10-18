"""Namespaced key-value store with TTL and quota enforcement.

This module exposes asynchronous helper functions around a pluggable adapter
layer.  The default adapter is backed by an embedded SQLite database which makes
it suitable for sandboxed deployments while still supporting concurrency,
per-namespace quotas, and key expiry semantics.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from modules.logging.logger import setup_logger

try:  # ConfigManager is optional in some test environments
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - fallback when ConfigManager unavailable
    ConfigManager = None  # type: ignore


__all__ = [
    "KeyValueStoreError",
    "KeyNotFoundError",
    "NamespaceQuotaExceededError",
    "GlobalQuotaExceededError",
    "InvalidValueTypeError",
    "ValueSerializationError",
    "KeyValueStoreService",
    "register_kv_store_adapter",
    "available_kv_store_adapters",
    "build_kv_store_service",
    "kv_get",
    "kv_set",
    "kv_delete",
    "kv_increment",
]


logger = setup_logger(__name__)


class KeyValueStoreError(RuntimeError):
    """Base class for key-value store failures."""


class KeyNotFoundError(KeyValueStoreError):
    """Raised when attempting to access a missing key."""


class NamespaceQuotaExceededError(KeyValueStoreError):
    """Raised when storing a value would exceed the namespace quota."""


class GlobalQuotaExceededError(KeyValueStoreError):
    """Raised when storing a value would exceed the global quota."""


class InvalidValueTypeError(KeyValueStoreError):
    """Raised when a value cannot be incremented as requested."""


class ValueSerializationError(KeyValueStoreError):
    """Raised when a value cannot be serialized for persistence."""


@dataclass(frozen=True)
class _StoreRecord:
    namespace: str
    key: str
    value: Any
    expires_at: Optional[float]

    def ttl_seconds(self, *, now: Optional[float] = None) -> Optional[float]:
        if self.expires_at is None:
            return None
        now = now if now is not None else time.time()
        remaining = self.expires_at - now
        return max(0.0, remaining)

    def to_dict(self) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "namespace": self.namespace,
            "key": self.key,
            "value": self.value,
        }
        ttl = self.ttl_seconds()
        if ttl is not None:
            payload["ttl_seconds"] = ttl
        return MappingProxyType(payload)


@dataclass(frozen=True)
class _GetResult:
    namespace: str
    key: str
    found: bool
    record: Optional[_StoreRecord]

    def to_dict(self) -> Mapping[str, Any]:
        payload: Dict[str, Any] = {
            "namespace": self.namespace,
            "key": self.key,
            "found": self.found,
        }
        if self.found and self.record is not None:
            payload["value"] = self.record.value
            ttl = self.record.ttl_seconds()
            if ttl is not None:
                payload["ttl_seconds"] = ttl
        return MappingProxyType(payload)


@dataclass(frozen=True)
class _WriteResult:
    namespace: str
    key: str
    record: _StoreRecord

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "namespace": self.namespace,
            "key": self.key,
        }
        payload.update(dict(self.record.to_dict()))
        return MappingProxyType(payload)


@dataclass(frozen=True)
class _DeleteResult:
    namespace: str
    key: str
    deleted: bool

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "namespace": self.namespace,
                "key": self.key,
                "deleted": self.deleted,
            }
        )


class KeyValueStoreAdapter(Protocol):
    """Protocol describing the storage adapter interface."""

    def get(self, namespace: str, key: str) -> _GetResult:
        ...

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: Optional[float]) -> _WriteResult:
        ...

    def delete(self, namespace: str, key: str) -> _DeleteResult:
        ...

    def increment(
        self,
        namespace: str,
        key: str,
        *,
        delta: int,
        ttl_seconds: Optional[float],
        initial_value: int,
    ) -> _WriteResult:
        ...


AdapterFactory = Callable[[Optional[ConfigManager], Mapping[str, Any]], KeyValueStoreAdapter]


_ADAPTERS: Dict[str, AdapterFactory] = {}


def register_kv_store_adapter(name: str, factory: AdapterFactory) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Adapter name must be a non-empty string")
    _ADAPTERS[key] = factory
    logger.debug("Registered key-value store adapter '%s'", key)


def available_kv_store_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def create_kv_store_adapter(
    name: str,
    *,
    config_manager: Optional[ConfigManager] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> KeyValueStoreAdapter:
    key = name.strip().lower()
    factory = _ADAPTERS.get(key)
    if factory is None:
        raise KeyValueStoreError(f"Unknown key-value store adapter '{name}'.")
    adapter_config: Mapping[str, Any]
    if isinstance(config, Mapping):
        adapter_config = MappingProxyType(dict(config))
    else:
        adapter_config = MappingProxyType({})
    return factory(config_manager, adapter_config)


class KeyValueStoreService:
    """Asynchronous facade around a synchronous adapter implementation."""

    def __init__(self, adapter: KeyValueStoreAdapter) -> None:
        self._adapter = adapter

    async def get_value(self, namespace: str, key: str) -> Mapping[str, Any]:
        result = await asyncio.to_thread(self._adapter.get, namespace, key)
        return result.to_dict()

    async def set_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: Optional[float] = None,
    ) -> Mapping[str, Any]:
        result = await asyncio.to_thread(
            self._adapter.set,
            namespace,
            key,
            value,
            ttl_seconds=ttl_seconds,
        )
        return result.to_dict()

    async def delete_value(self, namespace: str, key: str) -> Mapping[str, Any]:
        result = await asyncio.to_thread(self._adapter.delete, namespace, key)
        return result.to_dict()

    async def increment_value(
        self,
        namespace: str,
        key: str,
        *,
        delta: int = 1,
        ttl_seconds: Optional[float] = None,
        initial_value: int = 0,
    ) -> Mapping[str, Any]:
        result = await asyncio.to_thread(
            self._adapter.increment,
            namespace,
            key,
            delta=delta,
            ttl_seconds=ttl_seconds,
            initial_value=initial_value,
        )
        return result.to_dict()


def _merge_configs(*configs: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    merged: Dict[str, Any] = {}
    for config in configs:
        if isinstance(config, Mapping):
            merged.update(dict(config))
    return MappingProxyType(merged)


def build_kv_store_service(
    *,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> KeyValueStoreService:
    default_adapter = "sqlite"
    default_config: Mapping[str, Any] = MappingProxyType({})

    if config_manager is not None:
        tools_config = config_manager.get_config("tools", {})
        if isinstance(tools_config, Mapping):
            kv_config = tools_config.get("kv_store")
            if isinstance(kv_config, Mapping):
                raw_default = kv_config.get("default_adapter", default_adapter)
                if isinstance(raw_default, str) and raw_default.strip():
                    default_adapter = raw_default.strip()
                adapters = kv_config.get("adapters")
                if isinstance(adapters, Mapping):
                    default_config = MappingProxyType(dict(adapters.get(default_adapter, {})))

    effective_adapter = (adapter_name or default_adapter).strip() or default_adapter
    effective_config = _merge_configs(default_config, adapter_config)
    adapter = create_kv_store_adapter(
        effective_adapter,
        config_manager=config_manager,
        config=effective_config,
    )
    return KeyValueStoreService(adapter)


def _coerce_positive_int(value: Any, *, minimum: int, env_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise KeyValueStoreError(f"{env_name} must be an integer") from exc
    if parsed < minimum:
        raise KeyValueStoreError(f"{env_name} must be >= {minimum}")
    return parsed


class SQLiteKeyValueStoreAdapter:
    """Embedded SQLite adapter implementing the key-value operations."""

    def __init__(
        self,
        *,
        path: Path,
        namespace_quota_bytes: int,
        global_quota_bytes: Optional[int],
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._path = path
        self._namespace_quota_bytes = namespace_quota_bytes
        self._global_quota_bytes = global_quota_bytes
        self._lock = threading.RLock()

        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_entries(expires_at)"
            )

    def _purge_expired(self, *, cursor: sqlite3.Cursor, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        cursor.execute(
            "DELETE FROM kv_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )

    def _load_row(self, cursor: sqlite3.Cursor, namespace: str, key: str, *, now: Optional[float]) -> Optional[_StoreRecord]:
        row = cursor.execute(
            "SELECT value, expires_at FROM kv_entries WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        raw_value, expires_at = row
        if expires_at is not None and expires_at <= (now if now is not None else time.time()):
            cursor.execute(
                "DELETE FROM kv_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            return None
        value = json.loads(raw_value)
        return _StoreRecord(namespace=namespace, key=key, value=value, expires_at=expires_at)

    def _compute_usage(self, cursor: sqlite3.Cursor, namespace: Optional[str]) -> int:
        if namespace is None:
            query = "SELECT COALESCE(SUM(size_bytes), 0) FROM kv_entries"
            params = ()
        else:
            query = "SELECT COALESCE(SUM(size_bytes), 0) FROM kv_entries WHERE namespace = ?"
            params = (namespace,)
        result = cursor.execute(query, params).fetchone()
        return int(result[0]) if result else 0

    def _enforce_quotas(
        self,
        cursor: sqlite3.Cursor,
        *,
        namespace: str,
        new_size: int,
        previous_size: int,
    ) -> None:
        namespace_usage = self._compute_usage(cursor, namespace) - previous_size
        if namespace_usage + new_size > self._namespace_quota_bytes:
            raise NamespaceQuotaExceededError(
                f"Storing '{namespace}:{new_size}' would exceed namespace quota of {self._namespace_quota_bytes} bytes."
            )
        if self._global_quota_bytes is not None:
            global_usage = self._compute_usage(cursor, None) - previous_size
            if global_usage + new_size > self._global_quota_bytes:
                raise GlobalQuotaExceededError(
                    "Global key-value store quota exceeded."
                )

    def _serialize_value(self, value: Any) -> tuple[str, int]:
        try:
            encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueSerializationError("Values must be JSON serializable.") from exc
        size = len(encoded.encode("utf-8"))
        return encoded, size

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

    def get(self, namespace: str, key: str) -> _GetResult:
        with self._lock, self._connection:  # type: ignore[call-arg]
            cursor = self._connection.cursor()
            now = time.time()
            self._purge_expired(cursor=cursor, now=now)
            record = self._load_row(cursor, namespace, key, now=now)
            return _GetResult(namespace=namespace, key=key, found=record is not None, record=record)

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: Optional[float]) -> _WriteResult:
        ttl = self._normalize_ttl(ttl_seconds)
        encoded, size = self._serialize_value(value)
        expires_at = (time.time() + ttl) if ttl is not None else None

        with self._lock, self._connection:  # type: ignore[call-arg]
            cursor = self._connection.cursor()
            now = time.time()
            self._purge_expired(cursor=cursor, now=now)
            existing = cursor.execute(
                "SELECT size_bytes FROM kv_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            previous_size = int(existing[0]) if existing else 0
            self._enforce_quotas(cursor, namespace=namespace, new_size=size, previous_size=previous_size)
            cursor.execute(
                """
                INSERT INTO kv_entries(namespace, key, value, expires_at, size_bytes)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    size_bytes = excluded.size_bytes
                """,
                (namespace, key, encoded, expires_at, size),
            )
            record = _StoreRecord(namespace=namespace, key=key, value=value, expires_at=expires_at)
            return _WriteResult(namespace=namespace, key=key, record=record)

    def delete(self, namespace: str, key: str) -> _DeleteResult:
        with self._lock, self._connection:  # type: ignore[call-arg]
            cursor = self._connection.cursor()
            now = time.time()
            self._purge_expired(cursor=cursor, now=now)
            cursor.execute(
                "DELETE FROM kv_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            deleted = cursor.rowcount > 0
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
        ttl = self._normalize_ttl(ttl_seconds) if ttl_seconds is not None else None
        if not isinstance(delta, int):
            raise InvalidValueTypeError("delta must be an integer")
        if not isinstance(initial_value, int):
            raise InvalidValueTypeError("initial_value must be an integer")

        with self._lock, self._connection:  # type: ignore[call-arg]
            cursor = self._connection.cursor()
            now = time.time()
            self._purge_expired(cursor=cursor, now=now)
            record = self._load_row(cursor, namespace, key, now=now)
            if record is None:
                current_value = initial_value
            else:
                if not isinstance(record.value, int):
                    raise InvalidValueTypeError("Stored value is not an integer")
                current_value = record.value
            new_value = current_value + delta
            ttl_to_use = ttl if ttl is not None else (record.ttl_seconds(now=now) if record else None)
            encoded, size = self._serialize_value(new_value)
            expires_at = (time.time() + ttl_to_use) if ttl_to_use is not None else None

            existing_size = 0
            if record is not None:
                stored = cursor.execute(
                    "SELECT size_bytes FROM kv_entries WHERE namespace = ? AND key = ?",
                    (namespace, key),
                ).fetchone()
                existing_size = int(stored[0]) if stored else 0
            self._enforce_quotas(cursor, namespace=namespace, new_size=size, previous_size=existing_size)
            cursor.execute(
                """
                INSERT INTO kv_entries(namespace, key, value, expires_at, size_bytes)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    size_bytes = excluded.size_bytes
                """,
                (namespace, key, encoded, expires_at, size),
            )
            stored_record = _StoreRecord(namespace=namespace, key=key, value=new_value, expires_at=expires_at)
            return _WriteResult(namespace=namespace, key=key, record=stored_record)


def _sqlite_adapter_factory(
    _config_manager: Optional[ConfigManager], config: Mapping[str, Any]
) -> SQLiteKeyValueStoreAdapter:
    path_setting = config.get("path") or os.environ.get("ATLAS_KV_STORE_PATH")
    if path_setting:
        path = Path(str(path_setting)).expanduser()
    else:
        default_root = Path.home() / ".atlas"
        default_root.mkdir(parents=True, exist_ok=True)
        path = default_root / "kv_store.sqlite"

    namespace_quota = config.get("namespace_quota_bytes") or os.environ.get(
        "ATLAS_KV_NAMESPACE_QUOTA_BYTES",
        1_048_576,
    )
    global_quota = config.get("global_quota_bytes") or os.environ.get("ATLAS_KV_GLOBAL_QUOTA_BYTES")

    namespace_quota_bytes = _coerce_positive_int(
        namespace_quota,
        minimum=1,
        env_name="namespace_quota_bytes",
    )
    global_quota_bytes: Optional[int]
    if global_quota is None or global_quota == "":
        global_quota_bytes = None
    else:
        global_quota_bytes = _coerce_positive_int(global_quota, minimum=1, env_name="global_quota_bytes")

    busy_timeout = config.get("busy_timeout_ms") or 5000
    busy_timeout_ms = _coerce_positive_int(busy_timeout, minimum=1, env_name="busy_timeout_ms")

    return SQLiteKeyValueStoreAdapter(
        path=path,
        namespace_quota_bytes=namespace_quota_bytes,
        global_quota_bytes=global_quota_bytes,
        busy_timeout_ms=busy_timeout_ms,
    )


register_kv_store_adapter("sqlite", _sqlite_adapter_factory)


async def kv_get(
    *,
    namespace: str,
    key: str,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = build_kv_store_service(
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
    )
    return await service.get_value(namespace, key)


async def kv_set(
    *,
    namespace: str,
    key: str,
    value: Any,
    ttl_seconds: Optional[float] = None,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = build_kv_store_service(
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
    )
    return await service.set_value(namespace, key, value, ttl_seconds=ttl_seconds)


async def kv_delete(
    *,
    namespace: str,
    key: str,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = build_kv_store_service(
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
    )
    return await service.delete_value(namespace, key)


async def kv_increment(
    *,
    namespace: str,
    key: str,
    delta: int = 1,
    ttl_seconds: Optional[float] = None,
    initial_value: int = 0,
    adapter_name: Optional[str] = None,
    adapter_config: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[ConfigManager] = None,
) -> Mapping[str, Any]:
    service = build_kv_store_service(
        adapter_name=adapter_name,
        adapter_config=adapter_config,
        config_manager=config_manager,
    )
    return await service.increment_value(
        namespace,
        key,
        delta=delta,
        ttl_seconds=ttl_seconds,
        initial_value=initial_value,
    )
