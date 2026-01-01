# Storage Module Agent Guidelines

## Overview

The `modules/storage/` module provides centralized orchestration of all ATLAS persistence subsystems. It owns configuration, lifecycle, health monitoring, and access to SQL databases, MongoDB, key-value stores, and vector stores.

## Ownership & Scope

- **Primary Owner**: Data/DB Agent
- **Secondary**: Backend Agent (for integration with ATLAS core)

## Writable Scope

Data/DB Agent has full authority over:

- `modules/storage/**` — all files in this module
- Schema migrations affecting storage tables
- Storage-related configuration in `config.yaml` under `storage:` block

## Module Structure

```text
modules/storage/
├── __init__.py           # Public API exports
├── manager.py            # StorageManager orchestrator
├── settings.py           # StorageSettings configuration
├── pool.py               # SQLPool, MongoPool wrappers
├── health.py             # Health check system
├── unit_of_work.py       # Cross-store transactions
├── adapters.py           # Domain store factory and adapters
├── compat.py             # ConfigManager/setup wizard compatibility
├── backends/
│   ├── __init__.py
│   ├── postgresql.py     # PostgreSQL-specific operations
│   ├── sqlite.py         # SQLite-specific operations
│   └── mongodb.py        # MongoDB-specific operations
└── vectors/
    ├── __init__.py       # Vector provider exports
    ├── base.py           # VectorProvider abstract base
    ├── adapter.py        # Bridge to VectorStoreAdapter protocol
    ├── pgvector.py       # PostgreSQL pgvector provider
    ├── pinecone.py       # Pinecone cloud provider
    └── chroma.py         # ChromaDB provider
```

## Key Design Principles

### 1. Async-First Lifecycle

All lifecycle methods (`initialize`, `shutdown`, `health_check`) are async.
Connection pools use async locks for thread-safe initialization.

### 2. Settings Ownership

`StorageManager` owns all storage configuration via `StorageSettings`.
It does NOT delegate to `ConfigManager` for persistence config—settings
are self-contained and can be loaded from environment, dict, or YAML.

### 3. Lazy Store Initialization

Domain stores (conversations, tasks, jobs) are created on first access,
not during `initialize()`. This allows partial initialization when not
all stores are needed.

### 4. Vector Store Abstraction

Vector stores implement the `VectorProvider` interface regardless of
backend (pgvector, Pinecone, Chroma). New providers can be added by:

1. Creating a new file in `vectors/`
2. Implementing `VectorProvider`
3. Using `@register_vector_provider("name")` decorator
4. Adding initialization logic in `manager._init_vector_store()`

### 5. Singleton Pattern

`get_storage_manager()` provides async singleton access.
`reset_storage_manager()` is for testing only.

## Coding Standards

### Error Handling

- Raise `RuntimeError` for "not initialized" states
- Use specific exception types from each submodule
- Log errors with structured `extra={}` context

### Type Hints

- All public methods must have complete type annotations
- Use `TYPE_CHECKING` for import-time-only type hints
- Prefer `Optional[X]` over `X | None` for 3.9 compatibility

### Testing Requirements

Any changes require:

- Unit tests for new functionality
- Integration tests for backend interactions
- Health check coverage for new components

## Validation Checklist

Before submitting changes:

- [ ] `pytest tests/` passes
- [ ] No new type errors (`mypy modules/storage/`)
- [ ] Health checks cover all initialized components
- [ ] Settings can be loaded from environment and dict
- [ ] Async lifecycle methods are properly awaited

## Integration Points

### Consumers

- `ATLAS/ATLAS.py` — should obtain stores via `get_storage_manager()`
- `modules/Server/` — uses stores for API routes
- `modules/orchestration/` — job scheduler accesses job store

### Dependencies

- `modules/conversation_store/` — domain repository, models
- `modules/task_store/` — domain repository, models
- `modules/job_store/` — domain repository, models
- `modules/Tools/Base_Tools/kv_store.py` — KV store service

## Migration Notes

The storage module centralizes logic previously spread across:

- `ATLAS/config/persistence.py` — pool/engine creation
- Individual store modules — session factory management
- Scattered health checks — now unified in `health.py`

Existing code should migrate to use `StorageManager` for:

1. Getting session factories
2. Accessing domain repositories
3. Performing health checks
4. Managing storage lifecycle
