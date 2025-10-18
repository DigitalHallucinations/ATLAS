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
      faiss:
        index_path: /var/lib/atlas/faiss-index.bin
```

Each adapter can expose provider-specific settings such as API keys, index
names, or file paths.  When no adapter information is supplied with a tool
call, the defaults from this section are used.

## Providers

Vector store providers live in `modules/Tools/providers/vector_store/` and
register themselves via `register_vector_store_adapter`.  The repository ships
with an `in_memory` adapter that is suitable for development and testing.  The
pluggable design enables downstream deployments to register additional
providers—such as Pinecone or FAISS—without modifying the core tool.  When
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

