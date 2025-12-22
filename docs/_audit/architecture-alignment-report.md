---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-02-27
source_of_truth: docs/_audit/style-guide.md
---

# Architecture Alignment Report

> Navigation: See the [audit workspace README](./README.md) for cadence, quick-start steps, and recording guidance.

This report compares key architectural claims in the documentation against the current implementation. Each section lists notable claims, whether they match code behavior, and recommended follow-ups.

## Owner registry alignment

- ✅ Added `docs/contributing/agent-owners.md` with owner and cadence mappings sourced from `_audit` inventory entries to clarify who to engage for audited subsystems.
- ✅ Added `docs/contributing/audit-rollout.md` to capture the standard onboarding flow for new subsystem audits, including template copies, owner/cadence setup, first-pass execution, and reminder scheduling.

## Front matter and link spot-checks

- ✅ `docs/Personas.md`, `docs/architecture-overview.md`, `docs/conversation-store.md`, `docs/user-accounts.md`, `docs/configuration.md`, `docs/tasks/overview.md`, and `docs/tool-manifest.md` now include the standard front matter block. Quick previews confirmed heading rendering and intra-doc links remain intact after the retrofit.

## Visual asset workflow

- ✅ Established `docs/assets/` with section folders (for example, `ui/`, `server/`) and added `docs/contributing/visual-assets.md` to standardize naming, versioning, and Markdown embed patterns for diagrams.
- ✅ Updated visual asset guidance to prioritize Mermaid fenced blocks for sequence/flow/state diagrams, prefer `.svg` exports for complex visuals, and reserve `.png` as a fallback when vector export is unavailable, including inline and static embed examples.

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
