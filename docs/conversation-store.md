# Conversation store data model

The conversation store persists chat transcripts, episodic memories, persona graphs, and local account data in PostgreSQL via SQLAlchemy models that are orchestrated by `ConversationStoreRepository`. The repository coordinates schema bootstrap, tenant-aware CRUD helpers, and retention policies for both messages and accounts.

## Relational schema

### Core chat tables
- **`conversations`** – conversation header keyed by UUID with optional session linkage, tenant ownership, JSON metadata, and archive timestamp. Messages cascade on delete and the tenant/timestamp index powers recent conversation lookups.【F:modules/conversation_store/models.py†L161-L179】
- **`messages`** – per-turn payload tied to a conversation and tenant, including author, type, status, structured content, metadata, attachments, TSV full-text search column, and relationships to assets, vectors, and events.【F:modules/conversation_store/models.py†L183-L247】
- **`message_assets`**, **`message_vectors`**, and **`message_events`** – auxiliary tables that store attachments, embeddings (pgvector or float array), and lifecycle events with tenant scoping and conversation/time indexes for efficient fan-out.【F:modules/conversation_store/models.py†L285-L405】

### Memory graph and episodic recall
- **`episodic_memories`** – tenant-scoped memory entries referencing conversations, messages, or users with rich metadata, tag arrays, and expiry timestamps for retention-aware recall tooling.【F:modules/conversation_store/models.py†L251-L282】
- **`memory_graph_nodes`** and **`memory_graph_edges`** – property graph projection with tenant-aware unique constraints, optional labels/types, metadata, and bidirectional relationships used by the memory graph toolchain.【F:modules/conversation_store/models.py†L408-L504】

### User accounts and sessions
- **`users`** and **`sessions`** – application users and active chat sessions with metadata blobs, timestamps, and cascading relationships into credentials and conversations.【F:modules/conversation_store/models.py†L56-L158】
- **`user_credentials`**, **`user_login_attempts`**, and **`password_reset_tokens`** – credential store backing the local account service, capturing hashed passwords, login attempts, lockout metadata, and reset tokens for administrative flows.【F:modules/conversation_store/models.py†L77-L142】【F:modules/conversation_store/models.py†L109-L140】

## Repository helpers and normalization

`ConversationStoreRepository` wraps the schema with context-managed sessions, cross-table helpers, and normalization utilities:

- **Tenant scoping** – `_normalize_tenant_id` enforces non-empty tenant identifiers and is applied on every query/mutation that accepts a tenant parameter, ensuring multi-tenant isolation in helpers like `ensure_conversation`, `add_message`, and the vector deletion APIs.【F:modules/conversation_store/repository.py†L108-L112】【F:modules/conversation_store/repository.py†L1303-L1337】【F:modules/conversation_store/repository.py†L1362-L1509】【F:modules/conversation_store/repository.py†L2600-L2618】
- **Timestamp coercion** – `_coerce_dt` and `_dt_to_iso` normalize strings and naive datetimes to UTC ISO-8601, which keeps persisted timestamps consistent across message creation, login auditing, and retention checks.【F:modules/conversation_store/repository.py†L115-L141】【F:modules/conversation_store/repository.py†L865-L907】【F:modules/conversation_store/repository.py†L1484-L1486】
- **Message lifecycle helpers** – `add_message` creates conversations on demand, hydrates or provisions the associated user/session, deduplicates `client_message_id`, populates PostgreSQL full-text vectors, and records creation events alongside assets, embeddings, or custom events.【F:modules/conversation_store/repository.py†L1362-L1509】 Edits and soft deletes reuse `_store_events` to capture audit trails while updating status/metadata snapshots.【F:modules/conversation_store/repository.py†L1511-L1599】
- **Profile management** – `ensure_user`, `get_user_profile`, `list_user_profiles`, and `upsert_user_profile` resolve user identifiers across credentials and external IDs, merge structured profile/documents data, and synchronise display names for downstream personalization.【F:modules/conversation_store/repository.py†L1040-L1188】
- **Account security hooks** – login attempt recording, lockout state management, and password reset storage provide per-user audit history and are reused by `UserAccountService` for policy enforcement.【F:modules/conversation_store/repository.py†L808-L1001】【F:modules/user_accounts/user_account_service.py†L49-L195】
- **Retention entry points** – `prune_expired_messages`, `prune_archived_conversations`, and `run_retention` consume repository-level retention settings to soft delete, archive, or purge rows while returning summary statistics for monitoring.【F:modules/conversation_store/repository.py†L2620-L2770】 See the [conversation retention guide](conversation_retention.md) for policy configuration and worker orchestration details.【F:docs/conversation_retention.md†L1-L34】

## Integration with other subsystems

- **Schema bootstrap and shared stores** – `create_schema` creates the conversation tables and ensures the task/job stores are migrated so a single PostgreSQL database can service conversations, scheduling, and automation primitives.【F:modules/conversation_store/repository.py†L573-L619】
- **Chat and automation APIs** – `AtlasServer` builds a repository-backed instance for REST routes; `ConversationRoutes` applies request-level tenant scopes and delegates CRUD/search operations to the repository for live chat sessions and polling endpoints.【F:modules/Server/routes.py†L99-L163】【F:modules/Server/conversation_routes.py†L155-L200】
- **User account services** – the `ConversationCredentialStore` adapts repository helpers for password hashing, verification, login auditing, and account queries, keeping GTK/CLI flows decoupled from SQLAlchemy internals.【F:modules/user_accounts/user_account_service.py†L49-L195】
- **Vector analytics and retrieval** – the conversation vector catalog hydrates embeddings from `message_vectors`, upserts new payloads, and honours metadata filters when surfacing semantic search results for analytics dashboards and retrieval-augmented workflows.【F:modules/conversation_store/vector_pipeline.py†L59-L190】
- **Retention background jobs** – `RetentionWorker` runs in-process loops against `run_retention`, emitting stats for operators and aligning with the retention workflow documented separately.【F:modules/background_tasks/retention.py†L1-L70】【F:docs/conversation_retention.md†L21-L34】

## Related documentation

- [Conversation retention](conversation_retention.md) – retention keys, worker scheduling, and administrative triggers.
- [User accounts](user-accounts.md) – operator workflows layered on top of the repository-backed credential store.

