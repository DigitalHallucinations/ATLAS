---
audience: Backend developers and operators
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/kv_store.py; ATLAS/config/persistence.py
---

# Key-Value Store Tool

The key-value store tool provides a namespaced state store that supports
atomically updating counters, persisting arbitrary JSON-serializable payloads,
retrieving entries, and deleting keys.  Entries can be assigned TTLs to expire
automatically, and quotas prevent unbounded growth across namespaces.

## Operations

The tool exposes four operations:

* `kv_get` – Retrieve a value from a namespace, returning the remaining TTL when
  applicable.
* `kv_set` – Write a JSON-serializable value with an optional TTL.
* `kv_delete` – Remove a key from the namespace.
* `kv_increment` – Atomically increment an integer counter, creating it when
  missing.

Each operation is backed by the configured provider; the default implementation
persists data to PostgreSQL via SQLAlchemy and psycopg which aligns with the
rest of ATLAS' storage layer and scales across multiple processes.

## Configuration

Configuration lives under the `tools.kv_store` block in `config.yaml` and can be
augmented by environment variables for deployments that cannot edit the config
file.  The following example shows how to target a dedicated PostgreSQL
database, tune connection pooling, and adjust quotas:

```yaml
tools:
  kv_store:
    default_adapter: postgres
    adapters:
      postgres:
        reuse_conversation_store: false
        url: postgresql+psycopg://atlas:atlas@db.example.com:5432/atlas_kv
        namespace_quota_bytes: 2097152  # 2 MiB per namespace
        global_quota_bytes: 16777216   # 16 MiB total
        pool:
          size: 10
          max_overflow: 20
          timeout: 30
```

For lightweight deployments the SQLite adapter can be selected instead:

```yaml
tools:
  kv_store:
    default_adapter: sqlite
    adapters:
      sqlite:
        url: sqlite:///var/lib/atlas/kv.sqlite
        reuse_conversation_store: false
```

Environment variables provide fallbacks for the PostgreSQL adapter when
configuration entries are omitted:

* `ATLAS_KV_STORE_URL` – PostgreSQL DSN used when a dedicated KV database is
  desired.
* `ATLAS_KV_REUSE_CONVERSATION` – Set to `false` to prevent reusing the
  conversation-store engine.
* `ATLAS_KV_NAMESPACE_QUOTA_BYTES` – Per-namespace quota in bytes (defaults to
  1 MiB).
* `ATLAS_KV_GLOBAL_QUOTA_BYTES` – Global quota in bytes. Leave unset to disable
  the global limit.
* `ATLAS_KV_POOL_SIZE` – Size of the primary connection pool.
* `ATLAS_KV_MAX_OVERFLOW` – Maximum overflow connections the pool can create.
* `ATLAS_KV_POOL_TIMEOUT` – Seconds to wait for a connection before timing out.

Values are JSON-serialized before storage which means the byte quotas account
for the serialized representation.  When a write would exceed a quota the
operation fails with `NamespaceQuotaExceededError` or `GlobalQuotaExceededError`.

## Providers and Extensibility

Providers register via `register_kv_store_adapter`.  The bundled
`kv_store_postgres` and `kv_store_sqlite` providers instantiate the respective SQL
adapters and enforce quotas inside the database.  Downstream deployments can
register additional adapters (such as Redis or cloud-backed stores) and reference
them from persona manifests by setting the provider name in the tool metadata.

## Policy Hooks

Manifest entries expose `requires_flags` values such as
`type.system.kv_store_access` and `type.system.kv_store_write`, allowing persona
policy definitions to grant or deny read and write access to the state store on a
per-persona basis.
