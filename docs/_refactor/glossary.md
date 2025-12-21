---
audience: All readers
status: draft
last_verified: 2025-12-21
source_of_truth: docs/_refactor/style-guide.md
---

# ATLAS Glossary

Standardized definitions for recurring ATLAS terms. Each entry points to the primary code location and the audience doc that should own deeper guidance.

## Persona
- **Definition**: A configurable behavioral profile composed of one or more `persona` entries with a name, optional meaning, and locked/editable prompt sections. Personas can declare personal assistant toggles (calendar access and terminal permissions) plus allowlists for tools, skills, and collaboration participants. Personas are defined by JSON payloads validated against the persona schema. 【F:modules/Personas/schema.json†L1-L161】
- **Audience pointers**: Persona authors and backend developers. See [Personas](../Personas.md) for authoring workflows and schema examples.

## Tool
- **Definition**: A callable capability exposed to personas via normalized manifest entries that capture name, description, version, capability tags, auth requirements, safety/consent flags, idempotency, timeouts, cost metadata, persona allowlists, feature flags, and provider definitions. Shared tools live in `modules/Tools/tool_maps/functions.json`, with persona overrides under `modules/Personas/<Persona>/Toolbox/`. 【F:modules/Tools/manifest_loader.py†L34-L136】
- **Audience pointers**: Tool authors and reviewers. See [Tool Manifest](../tool-manifest.md) for field-level guidance.

## Skill
- **Definition**: A reusable instruction set with required tools and capabilities, versioning, safety notes, and optional collaboration configuration, sourced from skill manifests. Skills validate required fields (instruction prompt, required tools/capabilities, safety notes) before being loaded. 【F:modules/Skills/manifest_loader.py†L26-L113】
- **Audience pointers**: Persona and skill authors. See [Skill docs](../tools) (to be reorganized for authors) once the audience-specific skill guide lands.

## Capability Registry
- **Definition**: The orchestration catalog that ingests tool, skill, task, and job manifests, normalizes metadata, enforces persona filters, and tracks rolling health/latency metrics for routing and UI summaries. It exposes compatibility checks and summary views consumed by planners and dashboards. 【F:modules/orchestration/capability_registry.py†L1-L112】
- **Audience pointers**: Backend developers and operators. See the forthcoming refactored architecture notes (linked from [Architecture Overview](../architecture-overview.md)) for registry-driven routing flows.

## Conversation Store
- **Definition**: The persistent SQLAlchemy model set for conversations, users, sessions, messages, embeddings, and related artifacts, with tenant-scoped identifiers and cross-dialect portability (JSON/JSONB, pgvector). It provides the database backbone for conversation history, user accounts, and search vectors. 【F:modules/conversation_store/models.py†L1-L122】
- **Audience pointers**: Data/DB engineers and backend developers. See [Conversation Store](../conversation-store.md) for schema diagrams and repository usage.

## Message Bus
- **Definition**: The asynchronous pub/sub layer offering topic routing, priority ordering, correlation IDs, tracing metadata, retries, and pluggable backends (in-memory asyncio queues by default, Redis Streams optionally). Configured at runtime through `ConfigManager.configure_message_bus()`, which selects the backend via messaging settings. 【F:modules/orchestration/message_bus.py†L1-L115】【F:ATLAS/config/config_manager.py†L503-L515】
- **Audience pointers**: Operators and backend developers. See [Messaging Runbook](../ops/messaging.md) for deployment patterns and backend selection.

## Job
- **Definition**: A higher-level automation definition loaded from job manifests with names, summaries, persona eligibility, required skills/tools, task graphs, recurrence, acceptance criteria, and escalation policies. Shared jobs live in `modules/Jobs/jobs.json` with persona overrides merged during load. 【F:modules/Jobs/manifest_loader.py†L34-L117】
- **Audience pointers**: Backend developers and persona authors. See [Jobs in ATLAS](../jobs/index.md) and [Job manifests](../jobs/manifest.md) for authoring guidance.

## Task
- **Definition**: A routable unit of work described in task manifests with summaries, required skills/tools, acceptance criteria, escalation policies, tags, and priority. Tasks support persona overrides and validation before surfacing to the capability registry and task services. 【F:modules/Tasks/manifest_loader.py†L34-L108】
- **Audience pointers**: Backend developers and UI engineers. See [Task overview](../tasks/overview.md) for lifecycle and analytics details.

## Tenant
- **Definition**: A logical isolation boundary carried through identifiers (e.g., `tenant_id` on users, credentials, and session records) to separate user data and enforce scoping within the conversation store and related services. 【F:modules/conversation_store/models.py†L63-L110】
- **Audience pointers**: Operators and compliance reviewers. Tenant-specific policies will be detailed in the forthcoming multi-tenant operations guide linked from [Configuration](../configuration.md).

## Setup Wizard Presets
- **Definition**: YAML-driven profiles applied by the setup controller to prefill retention, auditing, provider defaults, and optional tenant IDs. Presets are loaded from `ATLAS/config/setup_presets/` and merged into the wizard state before database and storage configuration. 【F:ATLAS/setup/controller.py†L880-L960】
- **Audience pointers**: Administrators using the setup wizard. See [Setup Wizard](../setup-wizard.md) for end-to-end flow and preset behaviors.
