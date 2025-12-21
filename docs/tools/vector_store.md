---
audience: Backend developers and operators
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/vector_store.py; modules/Tools/providers/vector_store/; ATLAS/config/tooling.py
---

# Vector Store Tool

The vector store tool exposes three operations for managing and querying
embeddings across different providers via a common adapter layer:

* `upsert_vectors` – insert or update embeddings in a namespace.
* `query_vectors` – retrieve the most similar embeddings for a query vector.
* `delete_namespace` – remove every embedding associated with a namespace.

## Configuration

Vector store settings are read from the `tools.vector_store` block in
`config.yaml` (or environment overrides managed by `ConfigManager`).

```yaml
tools:
  vector_store:
    default_adapter: pinecone
    adapters:
      pinecone:
        api_key: ${PINECONE_API_KEY}
        environment: us-west4-gcp
        index_name: atlas-index
        namespace_prefix: tenant-
      mongodb:
        connection_uri: ${ATLAS_VECTOR_STORE_MONGODB_URI}
        database: atlas_vector_store
        collection: embeddings
        index_name: vector_index
        manage_index: true
        index_type: auto
      chroma:
        collection_name: atlas
        persist_directory: /var/lib/atlas/chroma
        metric: cosine
      faiss:
        index_path: /var/lib/atlas/faiss-index.json
        metric: l2
```

Each adapter can expose provider-specific settings such as API keys, index
names, or file paths.  When no adapter information is supplied with a tool
call, the defaults from this section are used.  All keys can be overridden
through environment variables using the standard `ConfigManager` resolution
flow.

### Pinecone

Set `api_key`, `index_name`, and optionally `environment`, `host`, or
`namespace_prefix`.  The adapter supports both the modern
`pinecone.Pinecone` client and the legacy `pinecone.init` + `pinecone.Index`
API.  Provide a custom client via `adapters.pinecone.client` in testing
scenarios.

### Chroma

Specify a `collection_name` which is used as a prefix for per-namespace
collections.  Supply either `persist_directory` for embedded deployments or
`server_host`/`server_port` for a remote instance.  Optional `metric` and
`collection_metadata` settings control similarity scoring and collection hints.

### FAISS

Configure `index_path` to persist embeddings locally (JSON encoded) and set
`metric` to `cosine` or `l2`.  The adapter maintains metadata alongside vectors
and falls back to in-memory mode when persistence is disabled.

### MongoDB / Atlas

The MongoDB adapter accepts a standard connection URI, database, collection,
and vector index name.  Connection credentials can be embedded in the
`mongodb+srv://` URI or supplied via `ATLAS_VECTOR_STORE_MONGODB_URI` in
environments where secrets management injects runtime values.  Optional
`client_kwargs` are forwarded to :class:`pymongo.mongo_client.MongoClient` to
customise TLS, timeouts, or appName tagging.

When `manage_index` is `true`, the adapter inspects the backing collection and
provisions a vector-capable index on first write:

1. `index_type: vector_search` provisions a native
   `createIndexes` vector search definition via
   :func:`modules.Tools.providers.vector_store.mongodb.ensure_mongodb_vector_index`.
2. `index_type: atlas` provisions an Atlas Search `knnVector` index via
   :func:`modules.Tools.providers.vector_store.mongodb.ensure_atlas_search_index`.
3. `index_type: auto` (default) first attempts native vector search and falls
   back to Atlas Search if the command is unsupported, allowing the same config
   to operate against Atlas clusters and Community builds.

You can override index settings with `index_overrides` to supply custom
similarity metrics, filter fields, or Atlas Search definitions.  Provide
`vector_dimension` when you need to pre-create the index before ingesting any
vectors.  Runtime queries accept metadata filters which the adapter flattens to
match nested fields within the configured `metadata_field` (defaults to
`metadata`).

## Providers

Vector store providers live in `modules/Tools/providers/vector_store/` and
register themselves via `register_vector_store_adapter`.  The repository ships
with adapters for in-memory development, Chroma, FAISS, and Pinecone.  When
adding a new adapter, read credentials and runtime options from
`ConfigManager` so they can be managed through `config.yaml` or environment
overrides.

## Tool Manifest

The shared `functions.json` manifest defines entries for each operation under
the `vector_store` capability.  Providers can be swapped by overriding the
`providers` configuration block when assembling a persona manifest or by
changing the default adapter in configuration.

Idempotency is achieved via namespace-aware keys:

* `upsert_vectors` – namespace plus sorted vector IDs.
* `query_vectors` – namespace plus the query vector hash (optional).
* `delete_namespace` – namespace identifier.

These keys help ToolManager safely retry operations without duplicating work.

