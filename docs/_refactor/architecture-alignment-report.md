---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2025-12-21
source_of_truth: docs/_refactor/style-guide.md
---

# Architecture Alignment Report

This report compares key architectural claims in the documentation against the current implementation. Each section lists notable claims, whether they match code behavior, and recommended follow-ups.

## docs/architecture-overview.md

- ✅ **Entry-point flow** – `main.py` instantiates `AtlasProvider`, gates startup on `is_setup_complete`, and defers `ATLAS.initialize()` until after setup succeeds via `FirstRunCoordinator`.  
- ✅ **Runtime construction** – `ATLAS/ATLAS.py` builds `ConfigManager`, configures the message bus, initializes speech, instantiates `AtlasServer`, and binds the conversation repository/service before exposing provider/persona/chat wiring during `initialize()`.  
- ✅ **Conversation store verification** – Startup calls `get_conversation_store_session_factory()` and raises if `is_conversation_store_verified()` is false, blocking the app when the store is missing required tables.  
- ⚠️ **Message bus adapters** – The doc notes Redis/in-memory wiring plus legacy `modules/Tools/tool_event_system` adapters. `ConfigManager.configure_message_bus()` only builds the bus and does not attach legacy adapters, so tool events are isolated unless callers wire them manually.  
  - *Recommended doc fix*: Clarify that legacy tool event adapters are not automatically bridged and must be integrated explicitly if needed.  
  - *Optional code fix*: Add an opt-in bridge that subscribes the message bus to `tool_event_system` topics.
- ⚠️ **Conversation store scope** – The doc claims the conversation store backs conversations, tasks, and job metadata. `ConversationStoreRepository` wraps conversations, vectors, accounts, and graph helpers; task/job persistence lives under `modules/task_store` and `modules/job_store`.  
  - *Recommended doc fix*: Update the scope to conversations/accounts/vector data and point task/job storage references to their dedicated repositories.  

## docs/setup-wizard.md

- ✅ **Setup completion** – The GTK wizard registers the staged administrator and writes the setup marker only after the final step succeeds.  
- ⚠️ **Branching & ordering** – The doc says Personal jumps straight to the Admin step and Enterprise routes through a combined company page first. In reality, both modes run `Introduction → Setup Type → Preflight`, then (for Personal/Enterprise) a `Users` roster page, the admin identity page, database intro/config, storage architecture, job scheduling, message bus, KV store, providers, and speech. Enterprise adds Company and Policies pages before the Users roster.  
  - *Recommended doc fix*: Rephrase the flow to include Preflight, Users roster before admin, and the storage architecture/database split, noting that company/policy pages are enterprise-only.
- ⚠️ **Step sequence detail** – The documented step list omits the storage-architecture preset page and splits the database configuration into a single bullet. The wizard handles database intro and configuration as separate steps and interleaves storage architecture before DB details.  
  - *Recommended doc fix*: Expand the step list to match the current sequence and call out where presets are applied (setup type for global defaults; storage architecture for performance presets).

## docs/server/api.md

- ✅ **HTTP gateway lifecycle** – `server/http_gateway.py` creates a shared `ATLAS` instance, awaits `initialize()`, and wires a fresh `AtlasServer` to the configured message bus and services; shutdown closes ATLAS and the bus.  
- ✅ **Context enforcement and streaming** – Route helpers enforce tenant-scoped `RequestContext`, and streaming helpers fall back to polling when no message bus is configured.

## docs/tasks/overview.md

- ✅ **Task metadata plumbing** – Task manifests live under `modules/Tasks/` (with persona overrides), are loaded by `manifest_loader`, and surface through `CapabilityRegistry.summary()`.  
- ✅ **Lifecycle orchestration** – `TaskService` delegates to `TaskStoreRepository`, emits lifecycle analytics, and enforces transition rules that match the documented state machine.  
- ✅ **Dashboard payloads** – Capability registry summaries combine tool/skill/task/job catalogs with lifecycle metrics for dashboards, as described.
