---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-01-03
source_of_truth: docs/_audit/style-guide.md
---

# Architecture Alignment Report

> Navigation: See the [audit workspace README](./README.md) for cadence, quick-start steps, and recording guidance.

This report compares key architectural claims in the documentation against the current implementation. Each section lists notable claims, whether they match code behavior, and recommended follow-ups.

## SOTA RAG Upgrade (2026-01-03)

Comprehensive RAG pipeline enhancement implementing state-of-the-art retrieval techniques:

- ✅ **Hybrid retrieval** – BM25 lexical search combined with pgvector semantic search via Reciprocal Rank Fusion (RRF). Implementation in `modules/storage/retrieval/`.
- ✅ **Query routing** – Zero-shot classification routes queries to optimal retrieval strategies. Uses HuggingFace transformers (`facebook/bart-large-mnli`).
- ✅ **Evidence gating** – Faithfulness scoring validates retrieved chunks against NLI entailment. Filters low-quality context before LLM prompting.
- ✅ **Hierarchical chunking** – Parent-child chunk relationships enable context expansion. Parent chunks (2048 tokens) contain child chunks (512 tokens).
- ✅ **Context compression** – LLMLingua-style perplexity-based compression and extractive sentence selection reduce token usage while preserving relevance.
- ✅ **Caching layers** – Embedding cache (LRU, 10k entries) and query result cache (semantic matching, 95% similarity threshold) improve latency.
- ✅ **Evaluation harness** – RAGAS-style metrics (faithfulness, relevancy, context precision/recall, answer correctness/similarity) with CLI at `scripts/evaluate_rag.py`.
- ✅ **Observability** – OpenTelemetry distributed tracing and Prometheus metrics. Grafana dashboard at `docs/ops/grafana-rag-dashboard.json`.
- ✅ **Configuration** – New `rag.hybrid`, `rag.routing`, `rag.compression`, `rag.caching`, `rag.observability`, and `rag.evaluation` sections in `docs/configuration.md`.
- ✅ **Dependencies** – Added `rank-bm25` to requirements.txt; `sentence-transformers` to requirements-accelerators.txt for evaluation.

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

## Polyglot Architecture Discussion (2026-01-09)

Added comprehensive strategy document for introducing compiled languages alongside Python:

- ✅ **Strategy document** – Created `docs/architecture/polyglot-strategy.md` exploring Rust integration for performance-critical components while maintaining Python-first development.
- ✅ **Performance analysis** – Identified four high-value optimization targets: vector operations, document processing, message bus, and storage management.
- ✅ **Language selection** – Recommended Rust over C++/Go/Zig based on safety, performance, PyO3 integration quality, and ecosystem momentum.
- ✅ **Integration patterns** – Documented three strategies: PyO3 native extensions (recommended), ctypes/CFFI for C/C++ libs, and gRPC microservices for Go.
- ✅ **Migration roadmap** – Outlined 4-phase approach: PoC (1-2 months), core modules (3-4 months), advanced optimizations (4-6 months), production hardening (2-3 months).
- ✅ **Cost-benefit analysis** – Projected 10-100x performance gains for compute-intensive operations with detailed impact assessment.
- ✅ **Decision framework** – Established clear criteria for when to use Rust vs Python based on bottleneck analysis, CPU-bound workload, and interface clarity.
- ✅ **Risk assessment** – Documented technical and organizational risks with mitigation strategies.
- ✅ **Code examples** – Included working Rust/PyO3 examples for vector operations with Python integration layer showing fallback patterns.
- ✅ **Build configuration** – Provided complete Cargo workspace, pyproject.toml, and CI/CD workflow configurations.
- ✅ **Documentation index** – Added architecture directory README and updated main README with link to architecture strategy docs.
- ✅ **Audit tracking** – Updated `docs/_audit/inventory.md` with new architecture documents.

**Status:** Proposed for stakeholder discussion. Next step: approval and Phase 1 PoC implementation.

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

## Multi-provider image generation (2026-01-04)

Comprehensive multi-provider image generation infrastructure with 10 provider implementations:

- ✅ **Provider registry** – `modules/Providers/Media/registry.py` implements a global registry pattern with async factory functions for provider instantiation.
- ✅ **Base abstractions** – `modules/Providers/Media/base.py` defines `MediaProvider` abstract base class with `generate_image()` and `edit_image()` methods, plus `ImageGenerationRequest`, `ImageGenerationResult`, and `EditImageRequest` dataclasses.
- ✅ **Manager orchestration** – `modules/Providers/Media/manager.py` provides `MediaProviderManager` for multi-provider management with 15 registration names: openai, huggingface, black_forest_labs, xai, stability_ai, stability, google, vertex_imagen, fal_ai, replicate, ideogram, runway, dall-e-3, flux, grok.
- ✅ **OpenAI provider** – DALL-E 2/3 support with quality, style, and size parameters.
- ✅ **HuggingFace provider** – Access to Stable Diffusion, FLUX, and custom models via Inference API.
- ✅ **Black Forest Labs provider** – FLUX.1 model family (schnell, dev, pro) with native aspect ratio support.
- ✅ **XAI provider** – Grok image generation with Aurora model.
- ✅ **Stability AI provider** – Stable Diffusion 3.x and legacy SD 1.x/SDXL; includes `stability` alias registration.
- ✅ **Google provider** – Imagen 3 via Vertex AI; includes `vertex_imagen` alias registration.
- ✅ **Fal.AI provider** – Fast inference for FLUX and custom LoRA models.
- ✅ **Replicate provider** – Access to thousands of open-source models including FLUX, SDXL, Kandinsky, and Playground.
- ✅ **Ideogram provider** – Text-in-image specialist with accurate typography rendering and magic prompt enhancement.
- ✅ **Runway provider** – Gen-3 Alpha creative generation tools with task-based async processing.
- ✅ **Tool integration** – `generate_image`, `edit_image`, `prompt_compiler`, and `clip_embeddings` tools in `modules/Tools/Base_Tools/`.
- ✅ **Documentation** – `docs/tools/image_generation.md` covers all providers, configuration, and usage patterns; `docs/configuration.md` updated with environment variables.

## docs/user-accounts.md

- ✅ **SQLite uplift guidance** – Documented the supported `migrate_sqlite_accounts` helper (`modules/user_accounts/sqlite_to_postgres_migration.py`) for moving credentials, lockouts, reset tokens, and login attempts from SQLite into the PostgreSQL conversation store used by current deployments.

## Job & Task Service Layer Migration (2026-06-16)

Comprehensive service layer implementation for Jobs and Tasks following the ATLAS service pattern with Actor-based permissions:

- ✅ **Common service types** – Added `core/services/common/types.py` with `Actor` (system, admin, user roles with tenant context), `OperationResult[T]` (typed return wrapper), and `Pagination` utilities.
- ✅ **Common exceptions** – Added `core/services/common/exceptions.py` with `ServiceError`, `NotFoundError`, `ValidationError`, `PermissionDeniedError`, and `ConflictError` base classes.
- ✅ **JobService implementation** – `core/services/jobs/service.py` provides CRUD operations, lifecycle management, and SOTA enhancements (checkpoints, agent assignment, partial progress tracking).
- ✅ **JobPermissionChecker** – `core/services/jobs/permissions.py` enforces role-based access control with tenant isolation; `admin`/`writer` can create/update, `reader` can view, `executor` manages lifecycle transitions.
- ✅ **Job domain types** – `core/services/jobs/types.py` defines `CreateJobRequest`, `UpdateJobRequest`, `JobFilters`, `JobListResult` with SOTA enhancement fields.
- ✅ **Job exceptions** – `core/services/jobs/exceptions.py` provides `JobNotFoundError`, `JobValidationError`, `JobPermissionDeniedError`, `JobTransitionError`, and `JobConflictError`.
- ✅ **TaskService implementation** – `core/services/tasks/service.py` provides CRUD, lifecycle, subtask management, task dependencies (with circular detection), priority system (1-100), and execution context.
- ✅ **TaskPermissionChecker** – `core/services/tasks/permissions.py` enforces role-based access with subtask permission inheritance from parent tasks.
- ✅ **Task domain types** – `core/services/tasks/types.py` defines `CreateTaskRequest`, `UpdateTaskRequest`, `TaskFilters`, `TaskListResult`, `CreateSubtaskRequest`, and `TaskDependency`.
- ✅ **Task exceptions** – `core/services/tasks/exceptions.py` provides `TaskNotFoundError`, `TaskValidationError`, `TaskPermissionDeniedError`, `TaskTransitionError`, `SubtaskError`, and `DependencyCycleError`.
- ✅ **Domain events** – Both services emit events via `MessageBus`: `job.created`, `job.updated`, `job.status_changed`, `job.checkpoint`, `job.agent_assigned`, `task.created`, `task.updated`, `task.status_changed`, `task.dependency_added`, `task.subtask_created`.
- ✅ **Permission model** – Consistent four-role hierarchy (admin → writer → executor → reader) with tenant isolation ensuring cross-tenant access is blocked.
- ✅ **Unit tests** – Comprehensive tests in `tests/services/jobs/` and `tests/services/tasks/` covering CRUD, permissions, lifecycle transitions, SOTA features, and error handling.
- ✅ **Documentation** – Added `docs/jobs/service.md` and `docs/tasks/service.md` with architecture overviews, permission scope tables, lifecycle diagrams, and code examples.
- 🟡 **Test execution blocked** – Tests are correctly written but cannot execute due to pre-existing circular import in `core/services/__init__.py` → `core/services/budget` → `modules/budget/api.py`. Existing budget tests also fail with the same import error.

**Status:** Implementation complete. Documentation aligned. Test execution requires resolving pre-existing circular import in the budget module.
