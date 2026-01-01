---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-01-01
source_of_truth: docs/_audit/style-guide.md
---

# Architecture Alignment Report

> Navigation: See the [audit workspace README](./README.md) for cadence, quick-start steps, and recording guidance.

This report compares key architectural claims in the documentation against the current implementation. Each section lists notable claims, whether they match code behavior, and recommended follow-ups.

## StorageManager refactor

- ✅ **Sole storage mechanism** – StorageManager (`modules/storage/manager.py`) is now the only storage mechanism in ATLAS. All legacy fallback paths have been removed.
- ✅ **ATLAS integration** – `ATLAS/ATLAS.py` initializes StorageManager during `initialize()` and obtains repositories via `storage.conversations`, `storage.tasks`, `storage.jobs`.
- ✅ **ConfigManager delegation** – `ATLAS/config/persistence.py` methods now delegate to StorageManager; legacy repository builders removed.
- ✅ **Server routes** – `modules/Server/routes.py` methods (`_build_conversation_repository`, `_build_task_service`, `_build_job_service`) require StorageManager with no fallback.
- ✅ **Documentation** – Added `docs/storage-manager.md` covering configuration, API, health checks, and config converters.
- ✅ **Legacy removal** – Removed `ConfigManagerStorageBridge` from `modules/storage/compat.py`; only config converters remain.

## Owner registry alignment

- ✅ Added `docs/contributing/agent-owners.md` with owner and cadence mappings sourced from `_audit` inventory entries to clarify who to engage for audited subsystems.
- ✅ Added `docs/contributing/audit-rollout.md` to capture the standard onboarding flow for new subsystem audits, including template copies, owner/cadence setup, first-pass execution, and reminder scheduling.

## Messaging system migration (NCB/AgentBus)

- ✅ **Architecture replacement** – The legacy `modules/orchestration/message_bus.py` has been fully replaced by the Neural Cognitive Bus (NCB) and AgentBus architecture under `ATLAS/messaging/`.
- ✅ **Channel architecture** – Migrated from generic topics to 36+ domain-specific semantic channels (e.g., `user.input`, `llm.request`, `tool.invoke`, `task.created`, `job.complete`).
- ✅ **Message types** – Replaced `BusMessage` with `AgentMessage` dataclass carrying ATLAS context fields (agent_id, conversation_id, request_id, user_id, trace_id).
- ✅ **API surface** – High-level `AgentBus` API provides `publish()`, `subscribe()`, `publish_from_sync()`, and channel configuration with priority queues, idempotency, and dead-letter handling.
- ✅ **Documentation updates** – Updated `docs/ops/messaging.md`, `docs/architecture-overview.md`, `docs/configuration.md`, `docs/_audit/glossary.md`, and `docs/_audit/inventory.md` to reflect the new architecture.
- ✅ **Legacy removal** – Deprecated bridge files (`bridge_redis_to_kafka.py`, `kafka_sink.py`) and removed legacy test files (`test_message_bus_backends.py`, `test_redis_to_kafka_bridge.py`).

## Front matter and link spot-checks

- ✅ `docs/Personas.md`, `docs/architecture-overview.md`, `docs/conversation-store.md`, `docs/user-accounts.md`, `docs/configuration.md`, `docs/tasks/overview.md`, and `docs/tool-manifest.md` now include the standard front matter block. Quick previews confirmed heading rendering and intra-doc links remain intact after the retrofit.
- ✅ Updated `docs/configuration.md` to remove the legacy MCP `server_config` fallback and clarify that `servers` entries are required when enabling MCP tooling.

## Visual asset workflow

- ✅ Established `docs/assets/` with section folders (for example, `ui/`, `server/`) and added `docs/contributing/visual-assets.md` to standardize naming, versioning, and Markdown embed patterns for diagrams.
- ✅ Updated visual asset guidance to prioritize Mermaid fenced blocks for sequence/flow/state diagrams, prefer `.svg` exports for complex visuals, and reserve `.png` as a fallback when vector export is unavailable, including inline and static embed examples. Added accessibility reminders (alt text, color contrast, legible fonts), a quick checklist, and sizing/alignment conventions to keep pages consistent. Latest revision also captures source storage under `docs/assets/.../src/`, a reviewer checklist (sources, alt text, relative links, feature parity), and optional validation tips (Mermaid linting, local HTTP spot-checks).
- ✅ Added server component and request-flow Mermaid diagrams (sources and SVG exports in `docs/assets/server/`) embedded in `docs/architecture-overview.md` and `docs/server/api.md` to anchor deployment/runtime narratives.
- ✅ Added task lifecycle and job retry/timeout sequence diagrams (sources under `docs/assets/tasks/` and `docs/assets/jobs/`) embedded in `docs/tasks/overview.md` and `docs/jobs/lifecycle.md` to illustrate submission, scheduling, and recovery flows.
- ✅ Added persona schema and tool manifest relationship diagrams (sources under `docs/assets/personas/src/` and `docs/assets/tools/src/` with SVG exports) and embedded them in `docs/Personas.md` and `docs/tool-manifest.md` to highlight required fields, allowlists, and validation constraints.
- 🟡 Added placeholder UI asset slots under `docs/assets/ui/` (sources in `docs/assets/ui/src/`) and embedded temporary captures plus a Mermaid navigation flow stub in `docs/ui/gtk-overview.md`; replace with finalized exports once screenshots are available.

## docs/architecture-overview.md

- ✅ **Entry-point flow** – `main.py` instantiates `AtlasProvider`, gates startup on `is_setup_complete`, and defers `ATLAS.initialize()` until after setup succeeds via `FirstRunCoordinator`.  
- ✅ **Runtime construction** – `ATLAS/ATLAS.py` builds `ConfigManager`, configures the message bus, initializes speech, instantiates `AtlasServer`, and binds the conversation repository/service before exposing provider/persona/chat wiring during `initialize()`.  
- ✅ **Conversation store verification** – Startup calls `get_conversation_store_session_factory()` and raises if `is_conversation_store_verified()` is false, blocking the app when the store is missing required tables.  
- ✅ **Message bus adapters** – The doc now notes Redis/in-memory wiring and explicitly states the legacy `modules/Tools/tool_event_system` adapters are not auto-bridged, so callers must connect them manually when required.  
- ✅ **Conversation store scope** – The doc now limits the conversation store to conversations, accounts, and vector data and points task/job storage references to `modules/task_store` and `modules/job_store`.  

## docs/setup-wizard.md

- ✅ **Setup completion** – The GTK wizard registers the staged administrator and writes the setup marker only after the final step succeeds.  
- ✅ **Branching & ordering** – The flow now lists `Introduction → Setup Type → Preflight → (Company/Policies for enterprise) → Users roster → Admin identity → Storage architecture presets → Database intro → Database config → Job scheduling → Message bus → KV store → Providers → Speech`, matching the GTK wizard’s ordering.  
- ✅ **Step sequence detail** – The step list calls out the storage-architecture preset page, separates the database intro/configuration bullets, and notes where setup-type defaults and preflight performance scores seed storage presets.  

## docs/server/api.md

- ✅ **HTTP gateway lifecycle** – `server/http_gateway.py` creates a shared `ATLAS` instance, awaits `initialize()`, and wires a fresh `AtlasServer` to the configured message bus and services; shutdown closes ATLAS and the bus.  
- ✅ **Context enforcement and streaming** – Route helpers enforce tenant-scoped `RequestContext`, and streaming helpers fall back to polling when no message bus is configured.

## docs/tasks/overview.md

- ✅ **Task metadata plumbing** – Task manifests live under `modules/Tasks/` (with persona overrides), are loaded by `manifest_loader`, and surface through `CapabilityRegistry.summary()`.  
- ✅ **Lifecycle orchestration** – `TaskService` delegates to `TaskStoreRepository`, emits lifecycle analytics, and enforces transition rules that match the documented state machine.  
- ✅ **Dashboard payloads** – Capability registry summaries combine tool/skill/task/job catalogs with lifecycle metrics for dashboards, as described.

## docs/conversation-store.md

- ✅ **Tenant context enforcement** – Credential and conversation lookups now reject missing tenant context when strict mode is enabled, and legacy tenantless auto-upgrades have been removed. A one-time helper (`scripts/migrations/tenantless_account_backfill.py`) migrates tenantless `users` and `user_credentials` rows before enabling strict tenancy.

## docs/user-accounts.md

- ✅ **SQLite uplift guidance** – Documented the supported `migrate_sqlite_accounts` helper (`modules/user_accounts/sqlite_to_postgres_migration.py`) for moving credentials, lockouts, reset tokens, and login attempts from SQLite into the PostgreSQL conversation store used by current deployments.
