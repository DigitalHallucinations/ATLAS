# Storage Manager

The **StorageManager** is the primary and only storage mechanism in ATLAS. It provides centralized orchestration for all persistence subsystems: databases, key-value stores, task/job stores, conversation stores, and vector stores.

> **Note**: StorageManager is the sole storage mechanism in ATLAS. All storage operations go through StorageManager.

## Overview

```Text
┌──────────────────────────────────────────────────────────────────┐
│                        StorageManager                              │
│                    (Singleton Orchestrator)                        │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │  SQLPool    │  │  MongoPool  │  │    VectorRegistry       │   │
│  │ (Engine +   │  │ (Client +   │  │   (pgvector/pinecone/   │   │
│  │  Sessions)  │  │  Database)  │  │    chroma)              │   │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘   │
│         │                │                      │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌───────────┴─────────────┐   │
│  │ PostgreSQL  │  │  MongoDB    │  │    VectorProvider       │   │
│  │  Backend    │  │  Backend    │  │    (Abstract)           │   │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Domain Store Factory                           │  │
│  │  conversations | tasks | jobs | kv | vectors               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from modules.storage import get_storage_manager, StorageManager

# Get the singleton instance (auto-initializes from environment)
async def main():
    storage = await get_storage_manager()
    
    # Access domain stores
    conversations = storage.conversations
    tasks = storage.tasks
    jobs = storage.jobs
    kv = storage.kv
    
    # Access vector store (if configured)
    vectors = storage.vectors
    
    # Health check
    health = await storage.health_check()
    print(f"Storage health: {health.overall}")
    
    # Graceful shutdown
    await storage.shutdown()
```

### Custom Configuration

```python
from modules.storage import StorageManager, StorageSettings, SQLBackendSettings, SQLDialect

# Create custom settings
settings = StorageSettings(
    sql=SQLBackendSettings(
        dialect=SQLDialect.POSTGRESQL,
        host="db.example.com",
        port=5432,
        database="mydb",
        username="user",
        password="secret",
    ),
    require_tenant_context=True,
)

# Create and initialize manager
manager = StorageManager(settings=settings)
await manager.initialize()

# Use the manager...

# Shutdown when done
await manager.shutdown()
```

### Environment Variables

StorageSettings can be loaded from environment variables:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `ATLAS_SQL_DIALECT` | `postgresql` or `sqlite` | `postgresql` |
| `ATLAS_SQL_HOST` | Database host | `localhost` |
| `ATLAS_SQL_PORT` | Database port | `5432` |
| `ATLAS_SQL_DATABASE` | Database name | `atlas` |
| `ATLAS_SQL_USERNAME` | Database user | - |
| `ATLAS_SQL_PASSWORD` | Database password | - |
| `ATLAS_SQL_URL` | Full connection URL (overrides above) | - |
| `ATLAS_VECTOR_BACKEND` | `pgvector`, `pinecone`, `chroma`, `none` | `pgvector` |
| `ATLAS_REQUIRE_TENANT_CONTEXT` | Require tenant ID for operations | `false` |

## Configuration

### StorageSettings

The main configuration object:

```python
from modules.storage.settings import (
    StorageSettings,
    SQLBackendSettings,
    MongoBackendSettings,
    VectorBackendSettings,
    PoolSettings,
    RetentionSettings,
    SQLDialect,
    VectorBackendType,
)

settings = StorageSettings(
    sql=SQLBackendSettings(
        dialect=SQLDialect.POSTGRESQL,
        host="localhost",
        port=5432,
        database="atlas",
        username="atlas",
        password="secret",
        pool=PoolSettings(
            size=5,
            max_overflow=10,
            timeout=30.0,
        ),
    ),
    mongo=MongoBackendSettings(
        enabled=False,  # Optional MongoDB support
    ),
    vectors=VectorBackendSettings(
        backend=VectorBackendType.PGVECTOR,
        embedding_dimension=1536,
        collection_name="atlas_vectors",
    ),
    retention=RetentionSettings(
        conversation_days=90,
        conversation_history_limit=100,
    ),
    require_tenant_context=False,
)
```

### SQL Backends

#### PostgreSQL

```python
from modules.storage.settings import SQLBackendSettings, SQLDialect

sql = SQLBackendSettings(
    dialect=SQLDialect.POSTGRESQL,
    host="db.example.com",
    port=5432,
    database="atlas",
    username="user",
    password="secret",
)

# Or use a connection URL
sql = SQLBackendSettings(
    dialect=SQLDialect.POSTGRESQL,
    url="postgresql://user:pass@host:5432/db",
)
```

#### SQLite

```python
sql = SQLBackendSettings(
    dialect=SQLDialect.SQLITE,
    database="/path/to/atlas.sqlite3",
)

# In-memory database
sql = SQLBackendSettings(
    dialect=SQLDialect.SQLITE,
    database=":memory:",
)
```

### Vector Backends

#### pgvector (PostgreSQL)

```python
from modules.storage.settings import VectorBackendSettings, VectorBackendType

vectors = VectorBackendSettings(
    backend=VectorBackendType.PGVECTOR,
    embedding_dimension=1536,
    collection_name="atlas_vectors",
    pgvector_options={
        "index_type": "hnsw",  # or "ivfflat"
    },
)
```

#### Pinecone

```python
vectors = VectorBackendSettings(
    backend=VectorBackendType.PINECONE,
    embedding_dimension=1536,
    pinecone_api_key="your-api-key",
    pinecone_environment="us-west1-gcp",
    pinecone_index_name="atlas",
)
```

#### ChromaDB

```python
vectors = VectorBackendSettings(
    backend=VectorBackendType.CHROMA,
    embedding_dimension=1536,
    chroma_persist_directory="/path/to/chroma",
)
```

## Domain Stores

StorageManager provides lazy-initialized access to domain stores:

### Conversations

```python
conversations = storage.conversations

# Create conversation
conv = conversations.create_conversation(
    title="My Chat",
    tenant_id="default",
)

# Store message
msg = conversations.store_message(
    conversation_id=conv["id"],
    role="user",
    content="Hello!",
    tenant_id="default",
)
```

### Tasks

```python
tasks = storage.tasks

# Create task
task = tasks.create_task(
    title="Process data",
    description="Process the uploaded file",
    owner_id="user123",
)
```

### Jobs

```python
jobs = storage.jobs

# Create scheduled job
job = jobs.create_job(
    name="daily_cleanup",
    schedule="0 3 * * *",  # Cron expression
    handler="cleanup_handler",
)
```

### Key-Value Store

```python
kv = storage.kv

# Store value
await kv.set("my_key", {"data": "value"})

# Retrieve value
value = await kv.get("my_key")
```

### Vectors

```python
from modules.storage.vectors.base import VectorDocument

vectors = storage.vectors

# Store embeddings
docs = [
    VectorDocument(
        id="doc1",
        vector=[0.1, 0.2, ...],  # 1536-dim embedding
        content="Hello world",
        metadata={"source": "chat"},
    ),
]
await vectors.upsert("my_collection", docs)

# Search
results = await vectors.search(
    "my_collection",
    query_vector=[0.1, 0.2, ...],
    top_k=10,
)
```

## Health Monitoring

```python
from modules.storage.health import HealthStatus

health = await storage.health_check()

print(f"Overall: {health.overall}")  # HEALTHY, DEGRADED, UNHEALTHY

# Component health
if health.sql:
    print(f"SQL: {health.sql.status} ({health.sql.latency_ms}ms)")

if health.mongo:
    print(f"MongoDB: {health.mongo.status}")

if health.vectors:
    print(f"Vectors: {health.vectors.status}")

# Serialize for API
health_dict = health.to_dict()
```

## Unit of Work

For cross-store transactions:

```python
async with storage.unit_of_work() as uow:
    # Queue operations
    uow.add_sql_operation(lambda session: ...)
    
    # Add compensating actions for rollback
    uow.add_compensating_action(lambda: ...)
    
    # If any operation fails, compensating actions run
```

## Config Converters

### Setup Wizard Integration

Convert between setup wizard state and StorageSettings:

```python
from modules.storage import (
    database_state_to_settings,
    storage_settings_to_database_state,
)
from core.setup.controller import DatabaseState

# Convert wizard state to settings
db_state = DatabaseState(
    backend="postgresql",
    host="localhost",
    port=5432,
    database="atlas",
    user="atlas",
    password="secret",
)
settings = database_state_to_settings(db_state)

# Convert back for wizard display
db_state = storage_settings_to_database_state(settings)
```

### StorageArchitecture Integration

Convert between StorageArchitecture and StorageSettings:

```python
from modules.storage import (
    storage_architecture_to_settings,
    storage_settings_to_architecture,
)

# Convert from architecture
settings = storage_architecture_to_settings(architecture)

# Convert back to architecture
architecture = storage_settings_to_architecture(settings)
```

### Tool System Integration

Register the StorageManager vector provider with the tool system:

```python
from modules.storage.vectors import register_storage_manager_adapter

# Register as a vector store adapter
register_storage_manager_adapter("storage_manager")

# Now tools can use it
from modules.Tools.Base_Tools.vector_store import create_vector_store_adapter

adapter = create_vector_store_adapter("storage_manager")
```

## Initialization Configuration

StorageManager is automatically initialized during ATLAS startup. Configure it via environment variables or `config.yaml`:

### Environment Setup

```bash
# Database settings
export ATLAS_SQL_DIALECT=postgresql
export ATLAS_SQL_HOST=localhost
export ATLAS_SQL_DATABASE=atlas
export ATLAS_SQL_USERNAME=atlas
export ATLAS_SQL_PASSWORD=secret

# Vector settings
export ATLAS_VECTOR_BACKEND=pgvector
```

### config.yaml

```yaml
storage:
  sql:
    dialect: postgresql
    host: localhost
    port: 5432
    database: atlas
    username: atlas
    password: secret
  vectors:
    backend: pgvector
```

## API Reference

### StorageManager

| Method | Description |
| ------ | ----------- |
| `await initialize()` | Initialize all storage subsystems |
| `await shutdown()` | Graceful shutdown |
| `await health_check()` | Get health status |
| `get_sql_engine()` | Get SQLAlchemy engine |
| `get_session_factory()` | Get session factory |
| `conversations` | Conversation repository |
| `tasks` | Task repository |
| `jobs` | Job repository |
| `kv` | Key-value store |
| `vectors` | Vector provider |

### Singleton Access

| Function | Description |
| -------- | ----------- |
| `await get_storage_manager()` | Get/create singleton |
| `get_storage_manager_sync()` | Get existing instance (no create) |
| `await reset_storage_manager()` | Shutdown and clear singleton |
