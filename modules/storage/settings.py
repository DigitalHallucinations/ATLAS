"""Storage configuration settings owned entirely by StorageManager.

This module defines all storage-related configuration as self-contained
dataclasses, independent of the legacy ConfigManager persistence mixins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse


class SQLDialect(str, Enum):
    """Supported SQL database dialects."""

    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


class VectorBackendType(str, Enum):
    """Supported vector store backends."""

    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    NONE = "none"


@dataclass(slots=True)
class PoolSettings:
    """Connection pool configuration for SQL backends."""

    size: int = 5
    max_overflow: int = 10
    timeout: float = 30.0
    recycle: int = 1800  # seconds
    pre_ping: bool = True

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert to SQLAlchemy create_engine kwargs."""
        return {
            "pool_size": self.size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.timeout,
            "pool_recycle": self.recycle,
            "pool_pre_ping": self.pre_ping,
        }


@dataclass(slots=True)
class SQLBackendSettings:
    """Configuration for a SQL database backend."""

    dialect: SQLDialect = SQLDialect.POSTGRESQL
    url: str = ""
    host: str = "localhost"
    port: int = 5432
    database: str = "atlas"
    username: str = ""
    password: str = ""
    pool: PoolSettings = field(default_factory=PoolSettings)

    def __post_init__(self) -> None:
        if isinstance(self.dialect, str):
            self.dialect = SQLDialect(self.dialect.lower())
        if isinstance(self.pool, Mapping):
            self.pool = PoolSettings(**self.pool)

    def get_url(self) -> str:
        """Build or return the database URL."""
        if self.url:
            return self.url

        if self.dialect == SQLDialect.SQLITE:
            db_path = self.database or "atlas.sqlite3"
            if db_path == ":memory:":
                return "sqlite:///:memory:"
            path = Path(db_path)
            return f"sqlite:///{path.as_posix()}"

        # PostgreSQL
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                auth += f":{self.password}"
            auth += "@"

        port_part = f":{self.port}" if self.port else ""
        return f"postgresql://{auth}{self.host}{port_part}/{self.database}"

    @classmethod
    def from_env(cls, prefix: str = "ATLAS_SQL") -> "SQLBackendSettings":
        """Load settings from environment variables."""
        return cls(
            dialect=SQLDialect(os.getenv(f"{prefix}_DIALECT", "postgresql").lower()),
            url=os.getenv(f"{prefix}_URL", ""),
            host=os.getenv(f"{prefix}_HOST", "localhost"),
            port=int(os.getenv(f"{prefix}_PORT", "5432")),
            database=os.getenv(f"{prefix}_DATABASE", "atlas"),
            username=os.getenv(f"{prefix}_USERNAME", ""),
            password=os.getenv(f"{prefix}_PASSWORD", ""),
            pool=PoolSettings(
                size=int(os.getenv(f"{prefix}_POOL_SIZE", "5")),
                max_overflow=int(os.getenv(f"{prefix}_POOL_MAX_OVERFLOW", "10")),
                timeout=float(os.getenv(f"{prefix}_POOL_TIMEOUT", "30.0")),
                recycle=int(os.getenv(f"{prefix}_POOL_RECYCLE", "1800")),
                pre_ping=os.getenv(f"{prefix}_POOL_PRE_PING", "true").lower() == "true",
            ),
        )


@dataclass(slots=True)
class MongoBackendSettings:
    """Configuration for MongoDB backend."""

    url: str = ""
    host: str = "localhost"
    port: int = 27017
    database: str = "atlas"
    username: str = ""
    password: str = ""
    auth_source: str = "admin"
    replica_set: str = ""
    tls: bool = False
    max_pool_size: int = 100
    min_pool_size: int = 0
    connect_timeout_ms: int = 20000
    server_selection_timeout_ms: int = 30000

    def get_url(self) -> str:
        """Build or return the MongoDB connection URL."""
        if self.url:
            return self.url

        auth = ""
        if self.username:
            from urllib.parse import quote_plus

            auth = quote_plus(self.username)
            if self.password:
                auth += f":{quote_plus(self.password)}"
            auth += "@"

        base = f"mongodb://{auth}{self.host}:{self.port}/{self.database}"
        params: List[str] = []

        if self.auth_source and self.username:
            params.append(f"authSource={self.auth_source}")
        if self.replica_set:
            params.append(f"replicaSet={self.replica_set}")
        if self.tls:
            params.append("tls=true")

        if params:
            base += "?" + "&".join(params)

        return base

    def get_client_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for MongoClient initialization."""
        return {
            "maxPoolSize": self.max_pool_size,
            "minPoolSize": self.min_pool_size,
            "connectTimeoutMS": self.connect_timeout_ms,
            "serverSelectionTimeoutMS": self.server_selection_timeout_ms,
        }

    @classmethod
    def from_env(cls, prefix: str = "ATLAS_MONGO") -> "MongoBackendSettings":
        """Load settings from environment variables."""
        return cls(
            url=os.getenv(f"{prefix}_URL", ""),
            host=os.getenv(f"{prefix}_HOST", "localhost"),
            port=int(os.getenv(f"{prefix}_PORT", "27017")),
            database=os.getenv(f"{prefix}_DATABASE", "atlas"),
            username=os.getenv(f"{prefix}_USERNAME", ""),
            password=os.getenv(f"{prefix}_PASSWORD", ""),
            auth_source=os.getenv(f"{prefix}_AUTH_SOURCE", "admin"),
            replica_set=os.getenv(f"{prefix}_REPLICA_SET", ""),
            tls=os.getenv(f"{prefix}_TLS", "false").lower() == "true",
            max_pool_size=int(os.getenv(f"{prefix}_MAX_POOL_SIZE", "100")),
            min_pool_size=int(os.getenv(f"{prefix}_MIN_POOL_SIZE", "0")),
            connect_timeout_ms=int(os.getenv(f"{prefix}_CONNECT_TIMEOUT_MS", "20000")),
            server_selection_timeout_ms=int(
                os.getenv(f"{prefix}_SERVER_SELECTION_TIMEOUT_MS", "30000")
            ),
        )


@dataclass(slots=True)
class VectorBackendSettings:
    """Configuration for vector store backends."""

    backend: VectorBackendType = VectorBackendType.PGVECTOR
    collection_name: str = "atlas_vectors"
    embedding_dimension: int = 1536

    # pgvector settings (uses SQL backend connection)
    pgvector_index_type: str = "ivfflat"  # or "hnsw"
    pgvector_lists: int = 100  # for ivfflat
    pgvector_ef_construction: int = 64  # for hnsw
    pgvector_m: int = 16  # for hnsw

    # Pinecone settings
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "atlas"
    pinecone_namespace: str = ""
    pinecone_metric: str = "cosine"  # cosine, euclidean, dotproduct

    # Chroma settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_directory: str = ""
    chroma_auth_token: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = VectorBackendType(self.backend.lower())

    @classmethod
    def from_env(cls, prefix: str = "ATLAS_VECTOR") -> "VectorBackendSettings":
        """Load settings from environment variables."""
        return cls(
            backend=VectorBackendType(
                os.getenv(f"{prefix}_BACKEND", "pgvector").lower()
            ),
            collection_name=os.getenv(f"{prefix}_COLLECTION", "atlas_vectors"),
            embedding_dimension=int(os.getenv(f"{prefix}_DIMENSION", "1536")),
            # pgvector
            pgvector_index_type=os.getenv(f"{prefix}_PGVECTOR_INDEX_TYPE", "ivfflat"),
            pgvector_lists=int(os.getenv(f"{prefix}_PGVECTOR_LISTS", "100")),
            pgvector_ef_construction=int(
                os.getenv(f"{prefix}_PGVECTOR_EF_CONSTRUCTION", "64")
            ),
            pgvector_m=int(os.getenv(f"{prefix}_PGVECTOR_M", "16")),
            # Pinecone
            pinecone_api_key=os.getenv(f"{prefix}_PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv(f"{prefix}_PINECONE_ENVIRONMENT", ""),
            pinecone_index_name=os.getenv(f"{prefix}_PINECONE_INDEX_NAME", "atlas"),
            pinecone_namespace=os.getenv(f"{prefix}_PINECONE_NAMESPACE", ""),
            pinecone_metric=os.getenv(f"{prefix}_PINECONE_METRIC", "cosine"),
            # Chroma
            chroma_host=os.getenv(f"{prefix}_CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv(f"{prefix}_CHROMA_PORT", "8000")),
            chroma_persist_directory=os.getenv(f"{prefix}_CHROMA_PERSIST_DIR", ""),
            chroma_auth_token=os.getenv(f"{prefix}_CHROMA_AUTH_TOKEN", ""),
            chroma_tenant=os.getenv(f"{prefix}_CHROMA_TENANT", "default_tenant"),
            chroma_database=os.getenv(f"{prefix}_CHROMA_DATABASE", "default_database"),
        )


@dataclass(slots=True)
class KVStoreSettings:
    """Configuration for key-value store."""

    backend: str = "sql"  # "sql" uses the SQL backend, "redis" for future
    namespace_quota_bytes: int = 0  # 0 = unlimited
    global_quota_bytes: int = 0  # 0 = unlimited
    default_ttl_seconds: int = 0  # 0 = no expiry

    @classmethod
    def from_env(cls, prefix: str = "ATLAS_KV") -> "KVStoreSettings":
        """Load settings from environment variables."""
        return cls(
            backend=os.getenv(f"{prefix}_BACKEND", "sql"),
            namespace_quota_bytes=int(os.getenv(f"{prefix}_NAMESPACE_QUOTA", "0")),
            global_quota_bytes=int(os.getenv(f"{prefix}_GLOBAL_QUOTA", "0")),
            default_ttl_seconds=int(os.getenv(f"{prefix}_DEFAULT_TTL", "0")),
        )


@dataclass(slots=True)
class RetentionSettings:
    """Data retention configuration."""

    conversation_days: int = 0  # 0 = keep forever
    conversation_history_limit: int = 0  # 0 = unlimited
    task_days: int = 0
    job_days: int = 0
    vector_days: int = 0

    @classmethod
    def from_env(cls, prefix: str = "ATLAS_RETENTION") -> "RetentionSettings":
        """Load settings from environment variables."""
        return cls(
            conversation_days=int(os.getenv(f"{prefix}_CONVERSATION_DAYS", "0")),
            conversation_history_limit=int(
                os.getenv(f"{prefix}_CONVERSATION_HISTORY_LIMIT", "0")
            ),
            task_days=int(os.getenv(f"{prefix}_TASK_DAYS", "0")),
            job_days=int(os.getenv(f"{prefix}_JOB_DAYS", "0")),
            vector_days=int(os.getenv(f"{prefix}_VECTOR_DAYS", "0")),
        )


@dataclass(slots=True)
class StorageSettings:
    """Root configuration for all storage subsystems.

    StorageManager owns this configuration entirely, independent of
    the application's ConfigManager.
    """

    # Primary SQL backend (conversations, tasks, jobs, kv store)
    sql: SQLBackendSettings = field(default_factory=SQLBackendSettings)

    # Optional MongoDB backend (alternative for conversations, jobs)
    mongo: MongoBackendSettings = field(default_factory=MongoBackendSettings)
    use_mongo_for_conversations: bool = False
    use_mongo_for_jobs: bool = False

    # Vector store configuration
    vectors: VectorBackendSettings = field(default_factory=VectorBackendSettings)

    # Key-value store configuration
    kv: KVStoreSettings = field(default_factory=KVStoreSettings)

    # Retention policies
    retention: RetentionSettings = field(default_factory=RetentionSettings)

    # Tenant isolation
    require_tenant_context: bool = False
    default_tenant_id: str = "default"

    # Health check settings
    health_check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.sql, Mapping):
            self.sql = SQLBackendSettings(**self.sql)
        if isinstance(self.mongo, Mapping):
            self.mongo = MongoBackendSettings(**self.mongo)
        if isinstance(self.vectors, Mapping):
            self.vectors = VectorBackendSettings(**self.vectors)
        if isinstance(self.kv, Mapping):
            self.kv = KVStoreSettings(**self.kv)
        if isinstance(self.retention, Mapping):
            self.retention = RetentionSettings(**self.retention)

    @classmethod
    def from_env(cls) -> "StorageSettings":
        """Load all settings from environment variables."""
        return cls(
            sql=SQLBackendSettings.from_env(),
            mongo=MongoBackendSettings.from_env(),
            use_mongo_for_conversations=os.getenv(
                "ATLAS_USE_MONGO_CONVERSATIONS", "false"
            ).lower()
            == "true",
            use_mongo_for_jobs=os.getenv("ATLAS_USE_MONGO_JOBS", "false").lower()
            == "true",
            vectors=VectorBackendSettings.from_env(),
            kv=KVStoreSettings.from_env(),
            retention=RetentionSettings.from_env(),
            require_tenant_context=os.getenv(
                "ATLAS_REQUIRE_TENANT_CONTEXT", "false"
            ).lower()
            == "true",
            default_tenant_id=os.getenv("ATLAS_DEFAULT_TENANT_ID", "default"),
            health_check_interval_seconds=float(
                os.getenv("ATLAS_HEALTH_CHECK_INTERVAL", "30.0")
            ),
            health_check_timeout_seconds=float(
                os.getenv("ATLAS_HEALTH_CHECK_TIMEOUT", "5.0")
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StorageSettings":
        """Create settings from a dictionary (e.g., parsed YAML)."""
        sql_data = data.get("sql") or data.get("conversation_database") or {}
        mongo_data = data.get("mongo") or data.get("mongodb") or {}
        vector_data = data.get("vectors") or data.get("vector_store") or {}
        kv_data = data.get("kv") or data.get("kv_store") or {}
        retention_data = data.get("retention") or {}

        return cls(
            sql=SQLBackendSettings(**sql_data) if sql_data else SQLBackendSettings(),
            mongo=MongoBackendSettings(**mongo_data)
            if mongo_data
            else MongoBackendSettings(),
            use_mongo_for_conversations=data.get("use_mongo_for_conversations", False),
            use_mongo_for_jobs=data.get("use_mongo_for_jobs", False),
            vectors=VectorBackendSettings(**vector_data)
            if vector_data
            else VectorBackendSettings(),
            kv=KVStoreSettings(**kv_data) if kv_data else KVStoreSettings(),
            retention=RetentionSettings(**retention_data)
            if retention_data
            else RetentionSettings(),
            require_tenant_context=data.get("require_tenant_context", False),
            default_tenant_id=data.get("default_tenant_id", "default"),
            health_check_interval_seconds=data.get("health_check_interval_seconds", 30.0),
            health_check_timeout_seconds=data.get("health_check_timeout_seconds", 5.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize settings to a dictionary."""
        import dataclasses

        def _asdict_recursive(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return _asdict_recursive(self)


__all__ = [
    "SQLDialect",
    "VectorBackendType",
    "PoolSettings",
    "SQLBackendSettings",
    "MongoBackendSettings",
    "VectorBackendSettings",
    "KVStoreSettings",
    "RetentionSettings",
    "StorageSettings",
]
