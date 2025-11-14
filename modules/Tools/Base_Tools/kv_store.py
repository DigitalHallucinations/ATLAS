"""Namespaced key-value store with TTL and quota enforcement.

This module exposes asynchronous helper functions around a pluggable adapter
layer.  The default adapter persists state in PostgreSQL using SQLAlchemy and
psycopg which aligns the KV store with the rest of ATLAS' persistence layer
while preserving concurrency, per-namespace quotas, and key expiry semantics.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from urllib.parse import urlparse

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    JSON,
    MetaData,
    Table,
    Text,
    create_engine,
    delete,
    func,
    select,
)
try:  # pragma: no cover - optional PostgreSQL extras
    from sqlalchemy.dialects.postgresql import JSONB  # type: ignore
    from sqlalchemy.dialects.postgresql import insert as pg_insert  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully when dialect missing
    JSONB = None  # type: ignore
    pg_insert = None  # type: ignore

try:  # pragma: no cover - optional SQLite extras
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully when dialect missing
    sqlite_insert = None  # type: ignore
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.sql import Select

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


def _create_kv_table(metadata: MetaData, value_type) -> Table:
    table = Table(
        "kv_entries",
        metadata,
        Column("namespace", Text, primary_key=True, nullable=False),
        Column("key", Text, primary_key=True, nullable=False),
        Column("value", value_type, nullable=False),
        Column("expires_at", Float, nullable=True),
        Column("size_bytes", Integer, nullable=False),
        comment="Namespaced key-value entries",
    )
    Index("idx_kv_entries_expires", table.c.expires_at)
    return table


def _load_adapter_defaults(
    config_manager: Optional[ConfigManager],
    adapter_name: str,
) -> Mapping[str, Any]:
    if config_manager is None:
        return {}

    kv_section = getattr(getattr(config_manager, "persistence", None), "kv_store", None)
    settings: Mapping[str, Any] | None = None

    if kv_section is not None and hasattr(kv_section, "get_settings"):
        try:
            settings = kv_section.get_settings()
        except Exception:  # pragma: no cover - fallback to legacy accessor
            settings = None

    if settings is None and hasattr(config_manager, "get_kv_store_settings"):
        try:
            settings = config_manager.get_kv_store_settings()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive fallback
            settings = None

    if isinstance(settings, Mapping):
        adapters = settings.get("adapters")
        if isinstance(adapters, Mapping):
            adapter_settings = adapters.get(adapter_name)
            if isinstance(adapter_settings, Mapping):
                return adapter_settings

    return {}


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
    default_adapter = "postgres"
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


class _SQLKeyValueStoreAdapterBase:
    """Shared SQLAlchemy-backed adapter implementation."""

    _TABLE: Table

    def __init__(
        self,
        *,
        engine: Engine,
        namespace_quota_bytes: int,
        global_quota_bytes: Optional[int],
    ) -> None:
        self._engine = engine
        self._namespace_quota_bytes = namespace_quota_bytes
        self._global_quota_bytes = global_quota_bytes
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._TABLE.metadata.create_all(self._engine, tables=[self._TABLE])
        except Exception as exc:  # pragma: no cover - defensive logging only
            raise KeyValueStoreError("Failed to initialize KV store schema") from exc

    def _purge_expired(self, connection, *, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        connection.execute(
            delete(self._TABLE).where(
                self._TABLE.c.expires_at.is_not(None),
                self._TABLE.c.expires_at <= now,
            )
        )

    def _load_row(
        self,
        connection,
        namespace: str,
        key: str,
        *,
        now: Optional[float],
        for_update: bool = False,
    ) -> Optional[_StoreRecord]:
        query: Select = select(
            self._TABLE.c.value,
            self._TABLE.c.expires_at,
        ).where(
            self._TABLE.c.namespace == namespace,
            self._TABLE.c.key == key,
        )
        if for_update:
            query = query.with_for_update()
        row = connection.execute(query).first()
        if row is None:
            return None
        value, expires_at = row
        now_value = now if now is not None else time.time()
        if expires_at is not None and expires_at <= now_value:
            connection.execute(
                delete(self._TABLE).where(
                    self._TABLE.c.namespace == namespace,
                    self._TABLE.c.key == key,
                )
            )
            return None
        return _StoreRecord(namespace=namespace, key=key, value=value, expires_at=expires_at)

    def _compute_usage(self, connection, namespace: Optional[str]) -> int:
        stmt = select(func.coalesce(func.sum(self._TABLE.c.size_bytes), 0))
        if namespace is not None:
            stmt = stmt.where(self._TABLE.c.namespace == namespace)
        result = connection.execute(stmt).scalar_one()
        return int(result)

    def _enforce_quotas(
        self,
        connection,
        *,
        namespace: str,
        new_size: int,
        previous_size: int,
    ) -> None:
        namespace_usage = self._compute_usage(connection, namespace) - previous_size
        if namespace_usage + new_size > self._namespace_quota_bytes:
            raise NamespaceQuotaExceededError(
                f"Storing '{namespace}:{new_size}' would exceed namespace quota of {self._namespace_quota_bytes} bytes."
            )
        if self._global_quota_bytes is not None:
            global_usage = self._compute_usage(connection, None) - previous_size
            if global_usage + new_size > self._global_quota_bytes:
                raise GlobalQuotaExceededError("Global key-value store quota exceeded.")

    def _serialize_value(self, value: Any) -> tuple[Any, int]:
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

    def _insert(self):  # pragma: no cover - overridden in subclasses
        raise NotImplementedError

    def _upsert(
        self,
        connection,
        *,
        namespace: str,
        key: str,
        serialized: Any,
        expires_at: Optional[float],
        size: int,
    ) -> None:
        statement = self._insert().values(
            namespace=namespace,
            key=key,
            value=serialized,
            expires_at=expires_at,
            size_bytes=size,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[self._TABLE.c.namespace, self._TABLE.c.key],
            set_={
                "value": serialized,
                "expires_at": expires_at,
                "size_bytes": size,
            },
        )
        connection.execute(statement)

    def get(self, namespace: str, key: str) -> _GetResult:
        with self._engine.begin() as connection:
            now = time.time()
            self._purge_expired(connection, now=now)
            record = self._load_row(connection, namespace, key, now=now)
            return _GetResult(namespace=namespace, key=key, found=record is not None, record=record)

    def set(self, namespace: str, key: str, value: Any, *, ttl_seconds: Optional[float]) -> _WriteResult:
        ttl = self._normalize_ttl(ttl_seconds)
        serialized, size = self._serialize_value(value)
        expires_at = (time.time() + ttl) if ttl is not None else None

        with self._engine.begin() as connection:
            now = time.time()
            self._purge_expired(connection, now=now)
            existing_size = connection.execute(
                select(self._TABLE.c.size_bytes).where(
                    self._TABLE.c.namespace == namespace,
                    self._TABLE.c.key == key,
                )
            ).scalar_one_or_none()
            previous_size = int(existing_size or 0)
            self._enforce_quotas(connection, namespace=namespace, new_size=size, previous_size=previous_size)
            self._upsert(
                connection,
                namespace=namespace,
                key=key,
                serialized=serialized,
                expires_at=expires_at,
                size=size,
            )
            record = _StoreRecord(namespace=namespace, key=key, value=value, expires_at=expires_at)
            return _WriteResult(namespace=namespace, key=key, record=record)

    def delete(self, namespace: str, key: str) -> _DeleteResult:
        with self._engine.begin() as connection:
            now = time.time()
            self._purge_expired(connection, now=now)
            result = connection.execute(
                delete(self._TABLE).where(
                    self._TABLE.c.namespace == namespace,
                    self._TABLE.c.key == key,
                )
            )
            deleted = result.rowcount is not None and result.rowcount > 0
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

        with self._engine.begin() as connection:
            now = time.time()
            self._purge_expired(connection, now=now)
            record = self._load_row(
                connection,
                namespace,
                key,
                now=now,
                for_update=True,
            )
            if record is None:
                current_value = initial_value
                existing_size = 0
            else:
                if not isinstance(record.value, int):
                    raise InvalidValueTypeError("Stored value is not an integer")
                current_value = record.value
                stored_size = connection.execute(
                    select(self._TABLE.c.size_bytes).where(
                        self._TABLE.c.namespace == namespace,
                        self._TABLE.c.key == key,
                    )
                ).scalar_one_or_none()
                existing_size = int(stored_size or 0)

            new_value = current_value + delta
            ttl_to_use = ttl if ttl is not None else (record.ttl_seconds(now=now) if record else None)
            serialized, size = self._serialize_value(new_value)
            expires_at = (time.time() + ttl_to_use) if ttl_to_use is not None else None

            self._enforce_quotas(connection, namespace=namespace, new_size=size, previous_size=existing_size)
            self._upsert(
                connection,
                namespace=namespace,
                key=key,
                serialized=serialized,
                expires_at=expires_at,
                size=size,
            )
            stored_record = _StoreRecord(namespace=namespace, key=key, value=new_value, expires_at=expires_at)
            return _WriteResult(namespace=namespace, key=key, record=stored_record)


class PostgresKeyValueStoreAdapter(_SQLKeyValueStoreAdapterBase):
    """PostgreSQL-backed adapter implementing the key-value operations."""

    _TABLE = _create_kv_table(MetaData(), JSONB if JSONB is not None else JSON)

    def _insert(self):
        if pg_insert is None:  # pragma: no cover - missing dialect
            raise KeyValueStoreError("PostgreSQL dialect support is unavailable")
        return pg_insert(self._TABLE)


class SQLiteKeyValueStoreAdapter(_SQLKeyValueStoreAdapterBase):
    """SQLite-backed adapter implementing the key-value operations."""

    _TABLE = _create_kv_table(MetaData(), JSON)

    def _insert(self):
        if sqlite_insert is None:  # pragma: no cover - missing dialect
            raise KeyValueStoreError("SQLite dialect support is unavailable")
        return sqlite_insert(self._TABLE)


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalize_postgres_url(url: str) -> str:
    try:
        parsed = make_url(url)
    except Exception:
        return url
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg")
    return str(parsed)


def _build_engine_from_config(
    config_manager: Optional[ConfigManager],
    config: Mapping[str, Any],
    *,
    adapter_name: str,
    normalize_url: Optional[Callable[[str], str]] = None,
) -> Engine:
    if adapter_name == "postgres" and config_manager is not None:
        kv_section = getattr(getattr(config_manager, "persistence", None), "kv_store", None)
        if kv_section is not None and hasattr(kv_section, "get_engine"):
            try:
                engine = kv_section.get_engine(adapter_config=config)
            except Exception as exc:  # pragma: no cover - fallback on manual construction
                logger.debug("Falling back to manual KV engine construction: %s", exc)
            else:
                if engine is not None:
                    return engine

        if hasattr(config_manager, "get_kv_store_engine"):
            try:
                engine = config_manager.get_kv_store_engine(adapter_config=config)
            except Exception as exc:  # pragma: no cover - fallback on manual construction
                logger.debug("Falling back to manual KV engine construction: %s", exc)
            else:
                if engine is not None:
                    return engine

    reuse_conversation = _normalize_bool(
        config.get("reuse_conversation_store"),
        default=False,
    )

    if reuse_conversation and config_manager is not None:
        conversation_section = getattr(getattr(config_manager, "persistence", None), "conversation", None)
        if conversation_section is not None and hasattr(conversation_section, "get_engine"):
            engine = conversation_section.get_engine()
            if engine is not None and hasattr(engine, "dialect"):
                return engine

        if hasattr(config_manager, "get_conversation_store_engine"):
            engine = config_manager.get_conversation_store_engine()
            if engine is not None and hasattr(engine, "dialect"):
                return engine

    url_value = config.get("url")
    if not isinstance(url_value, str) or not url_value.strip():
        raise KeyValueStoreError(f"{adapter_name} adapter requires a configured DSN")

    raw_url = url_value.strip()
    if raw_url.lower().startswith("mongodb://") or raw_url.lower().startswith("mongodb+srv://"):
        raise KeyValueStoreError("MongoDB DSNs must use the 'mongo' key-value adapter")
    normalized_url = normalize_url(raw_url) if normalize_url is not None else raw_url

    pool_config: Dict[str, Any] = {}
    raw_pool_override = config.get("pool")
    if isinstance(raw_pool_override, Mapping):
        pool_config.update(dict(raw_pool_override))

    engine_kwargs: Dict[str, Any] = {"future": True}
    if pool_config.get("size") is not None:
        engine_kwargs["pool_size"] = int(pool_config["size"])
    if pool_config.get("max_overflow") is not None:
        engine_kwargs["max_overflow"] = int(pool_config["max_overflow"])
    if pool_config.get("timeout") is not None:
        engine_kwargs["pool_timeout"] = float(pool_config["timeout"])

    return create_engine(normalized_url, **engine_kwargs)


def _postgres_adapter_factory(
    config_manager: Optional[ConfigManager],
    config: Mapping[str, Any],
) -> PostgresKeyValueStoreAdapter:
    defaults = _load_adapter_defaults(config_manager, "postgres")

    namespace_quota = config.get("namespace_quota_bytes")
    if namespace_quota is None:
        namespace_quota = defaults.get("namespace_quota_bytes", 1_048_576)
    namespace_quota_bytes = _coerce_positive_int(
        namespace_quota,
        minimum=1,
        env_name="namespace_quota_bytes",
    )

    global_quota = config.get("global_quota_bytes")
    if global_quota in (None, ""):
        default_global = defaults.get("global_quota_bytes")
        if default_global in (None, ""):
            global_quota_bytes: Optional[int] = None
        else:
            global_quota_bytes = _coerce_positive_int(
                default_global,
                minimum=1,
                env_name="global_quota_bytes",
            )
    else:
        global_quota_bytes = _coerce_positive_int(
            global_quota,
            minimum=1,
            env_name="global_quota_bytes",
        )

    engine = _build_engine_from_config(
        config_manager,
        config,
        adapter_name="postgres",
        normalize_url=_normalize_postgres_url,
    )

    return PostgresKeyValueStoreAdapter(
        engine=engine,
        namespace_quota_bytes=namespace_quota_bytes,
        global_quota_bytes=global_quota_bytes,
    )


register_kv_store_adapter("postgres", _postgres_adapter_factory)


def _sqlite_adapter_factory(
    config_manager: Optional[ConfigManager],
    config: Mapping[str, Any],
) -> SQLiteKeyValueStoreAdapter:
    defaults = _load_adapter_defaults(config_manager, "sqlite")

    effective_config: Dict[str, Any] = dict(config)

    url_value = effective_config.get("url")
    if not isinstance(url_value, str) or not url_value.strip():
        default_url = defaults.get("url")
        if not isinstance(default_url, str) or not default_url.strip():
            default_url = "sqlite:///atlas_kv.sqlite"
        effective_config["url"] = default_url

    namespace_quota = effective_config.get("namespace_quota_bytes")
    if namespace_quota is None:
        namespace_quota = defaults.get("namespace_quota_bytes", 1_048_576)
    namespace_quota_bytes = _coerce_positive_int(
        namespace_quota,
        minimum=1,
        env_name="namespace_quota_bytes",
    )

    global_quota = effective_config.get("global_quota_bytes")
    if global_quota in (None, ""):
        default_global = defaults.get("global_quota_bytes")
        if default_global in (None, ""):
            global_quota_bytes: Optional[int] = None
        else:
            global_quota_bytes = _coerce_positive_int(
                default_global,
                minimum=1,
                env_name="global_quota_bytes",
            )
    else:
        global_quota_bytes = _coerce_positive_int(
            global_quota,
            minimum=1,
            env_name="global_quota_bytes",
        )

    engine = _build_engine_from_config(
        config_manager,
        effective_config,
        adapter_name="sqlite",
    )

    return SQLiteKeyValueStoreAdapter(
        engine=engine,
        namespace_quota_bytes=namespace_quota_bytes,
        global_quota_bytes=global_quota_bytes,
    )


register_kv_store_adapter("sqlite", _sqlite_adapter_factory)


try:  # pragma: no cover - optional dependency path
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover - PyMongo unavailable
    MongoClient = None  # type: ignore

try:
    from .kv_store_mongo import MongoKeyValueStoreAdapter
except Exception:  # pragma: no cover - adapter unavailable
    MongoKeyValueStoreAdapter = None  # type: ignore
else:

    def _mongo_adapter_factory(
        config_manager: Optional[ConfigManager],
        config: Mapping[str, Any],
    ) -> MongoKeyValueStoreAdapter:
        if MongoKeyValueStoreAdapter is None:
            raise KeyValueStoreError("Mongo key-value adapter is unavailable")
        defaults = _load_adapter_defaults(config_manager, "mongo")

        effective: Dict[str, Any] = dict(defaults)
        effective.update(dict(config))

        url_value = effective.get("url")
        if not isinstance(url_value, str) or not url_value.strip():
            raise KeyValueStoreError("Mongo adapter requires a configured URL")
        mongo_url = url_value.strip()

        namespace_quota = effective.get("namespace_quota_bytes")
        if namespace_quota is None:
            namespace_quota = defaults.get("namespace_quota_bytes", 1_048_576)
        namespace_quota_bytes = _coerce_positive_int(
            namespace_quota,
            minimum=1,
            env_name="namespace_quota_bytes",
        )

        global_quota = effective.get("global_quota_bytes")
        if global_quota in (None, ""):
            default_global = defaults.get("global_quota_bytes")
            if default_global in (None, ""):
                global_quota_bytes: Optional[int] = None
            else:
                global_quota_bytes = _coerce_positive_int(
                    default_global,
                    minimum=1,
                    env_name="global_quota_bytes",
                )
        else:
            global_quota_bytes = _coerce_positive_int(
                global_quota,
                minimum=1,
                env_name="global_quota_bytes",
            )

        if MongoClient is None:
            raise KeyValueStoreError("PyMongo driver is unavailable for the Mongo adapter")

        try:
            client = MongoClient(mongo_url)
        except Exception as exc:
            raise KeyValueStoreError(f"Failed to connect to MongoDB: {exc}") from exc

        parsed = urlparse(mongo_url)
        database_name = effective.get("database")
        if not database_name:
            database_name = parsed.path.lstrip("/").split("?", 1)[0] or "atlas"
        collection_name = effective.get("collection") or "kv_store"

        database = client.get_database(database_name)
        adapter = MongoKeyValueStoreAdapter.from_database(
            database,
            client=client,
            collection_name=collection_name,
            namespace_quota_bytes=namespace_quota_bytes,
            global_quota_bytes=global_quota_bytes,
        )
        return adapter


    register_kv_store_adapter("mongo", _mongo_adapter_factory)
    register_kv_store_adapter("mongodb", _mongo_adapter_factory)


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
