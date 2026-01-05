---
audience: Backend and data service owners
status: draft
last_verified: 2026-01-03
source_of_truth: ./style-guide.md
---

# Subsystem audit inventory

> Navigation: Start with the [modules audit README](./README.md) for cadence guidance and cross-links to related findings.

| path | owner | last_audited | alignment_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| modules/storage/manager.py | @data-eng | 2026-01-01 | Aligned | New module; StorageManager is now the sole storage mechanism. | 2026-04-01 | Primary orchestrator for all persistence; see [`docs/storage-manager.md`](../../docs/storage-manager.md). |
| modules/storage/settings.py | @data-eng | 2026-01-01 | Aligned | Configuration dataclasses for SQL, Mongo, Vector, and retention settings. | 2026-04-01 | Loaded from environment or config.yaml. |
| modules/storage/adapters.py | @data-eng | 2026-01-01 | Aligned | Domain store factory bridges StorageManager to conversation/task/job repositories. | 2026-04-01 | Repositories obtained via `storage.conversations`, `storage.tasks`, `storage.jobs`. |
| modules/storage/compat.py | @data-eng | 2026-01-01 | Aligned | Config converters for setup wizard and StorageArchitecture interop. | 2026-04-01 | Legacy ConfigManagerStorageBridge removed; only converters remain. |
| modules/storage/retrieval/ | @data-eng | 2026-01-03 | Aligned | SOTA RAG hybrid retrieval with BM25 lexical search and RRF fusion. | 2026-04-03 | Includes HybridRetriever, BM25Index, QueryRouter, and EvidenceGate. |
| modules/storage/chunking/ | @data-eng | 2026-01-03 | Aligned | Hierarchical chunking with parent-child relationships for better context. | 2026-04-03 | HierarchicalChunker with parent context retrieval. |
| modules/storage/ingestion/ | @data-eng | 2026-01-03 | Aligned | Context compression (extractive, LLMLingua) integrated into RAG pipeline. | 2026-04-03 | ContextCompressor with multiple strategy support. |
| modules/storage/evaluation/ | @data-eng | 2026-01-03 | Aligned | RAG quality evaluation harness with RAGAS-style metrics. | 2026-04-03 | RAGEvaluator, dataset loader; CLI at `scripts/evaluate_rag.py`. |
| modules/storage/observability/ | @data-eng | 2026-01-03 | Aligned | OpenTelemetry tracing and Prometheus metrics for RAG pipeline. | 2026-04-03 | RAGTracer, RAGMetrics; dashboards at `docs/ops/`. |
| modules/orchestration/job_scheduler.py | @backend-core | 2026-02-20 | Needs review | Retry defaults and backoff handling recently changed; confirm persisted schedules match docs. | 2026-04-15 | Track alongside scheduler drift in [`alignment-report.md`](./alignment-report.md#drift-findings). |
| modules/Personas/schema.json | @persona-maintainers | 2026-02-20 | Aligned | Schema validations reflect current persona tooling expectations. | 2026-05-20 | Re-run persona schema tests after manifest changes. |
| modules/conversation_store/repository.py | @data-eng | 2026-02-20 | Needs review | Retention hooks and vector pipeline calls may diverge from current configs. | 2026-03-30 | Pair with Data/DB alignment items in [`alignment-report.md`](./alignment-report.md#current-risks). |
| modules/job_store/service.py | @backend-core | 2026-02-20 | Aligned | Job lifecycle transitions match documented status machine. | 2026-05-01 | Reconfirm after task/job routing updates. |
| modules/Providers/Media/base.py | @backend-core | 2026-01-04 | Aligned | Abstract base class for media providers; defines generate_image() and edit_image() contracts. | 2026-04-04 | Core abstraction for all image generation providers. |
| modules/Providers/Media/manager.py | @backend-core | 2026-01-04 | Aligned | MediaProviderManager with 15 provider registrations; handles multi-provider orchestration. | 2026-04-04 | AVAILABLE_PROVIDERS list must stay aligned with provider implementations. |
| modules/Providers/Media/registry.py | @backend-core | 2026-01-04 | Aligned | Global registry pattern with async factory functions for provider instantiation. | 2026-04-04 | Central registration point for all media providers. |
| modules/Providers/Media/OpenAIImages/ | @backend-core | 2026-01-04 | Aligned | DALL-E 2/3 provider with quality, style, and size parameters. | 2026-04-04 | Registered as "openai" and "dall-e-3". |
| modules/Providers/Media/HuggingFace/ | @backend-core | 2026-01-04 | Aligned | HuggingFace Inference API provider for SD, FLUX, and custom models. | 2026-04-04 | Registered as "huggingface". |
| modules/Providers/Media/BlackForestLabs/ | @backend-core | 2026-01-04 | Aligned | FLUX.1 model family (schnell, dev, pro) with aspect ratio support. | 2026-04-04 | Registered as "black_forest_labs" and "flux". |
| modules/Providers/Media/XAI/ | @backend-core | 2026-01-04 | Aligned | Grok image generation with Aurora model. | 2026-04-04 | Registered as "xai" and "grok". |
| modules/Providers/Media/Stability/ | @backend-core | 2026-01-04 | Aligned | Stable Diffusion 3.x and legacy SD 1.x/SDXL support. | 2026-04-04 | Registered as "stability_ai" and "stability" (alias). |
| modules/Providers/Media/Google/ | @backend-core | 2026-01-04 | Aligned | Imagen 3 via Vertex AI with safety settings. | 2026-04-04 | Registered as "google" and "vertex_imagen" (alias). |
| modules/Providers/Media/FalAI/ | @backend-core | 2026-01-04 | Aligned | Fast inference for FLUX and custom LoRA models. | 2026-04-04 | Registered as "fal_ai". |
| modules/Providers/Media/Replicate/ | @backend-core | 2026-01-04 | Aligned | Access to thousands of open-source models via Replicate API. | 2026-04-04 | Registered as "replicate"; supports FLUX, SDXL, Kandinsky, Playground. |
| modules/Providers/Media/Ideogram/ | @backend-core | 2026-01-04 | Aligned | Text-in-image specialist with accurate typography and magic prompt. | 2026-04-04 | Registered as "ideogram"; V_2, V_2_TURBO models. |
| modules/Providers/Media/Runway/ | @backend-core | 2026-01-04 | Aligned | Gen-3 Alpha creative generation with task-based async processing. | 2026-04-04 | Registered as "runway"; gen3a_turbo, gen2 models. |

## Legend

- **path**: Source file under `modules/` being tracked for audit coverage.
- **owner**: Primary maintainer or reviewer group; prefer team aliases.
- **last_audited**: Date of the last substantive review; include short parenthetical notes if the review added context.
- **alignment_status**:
  - `Aligned`: Behavior or docs reflect current source of truth.
  - `Needs review`: Partial confidence; schedule near-term verification.
  - `Needs overhaul`: Significant drift or missing coverage; prioritize remediation.
- **gaps_found**: Brief description of drift, missing claims, or link debt observed during the last review.
- **next_review**: Target date based on the cadence described in [`README.md`](./README.md#audit-checklist).
- **notes**: Freeform cross-links to remediation issues, PRs, or [`alignment-report.md`](./alignment-report.md) entries.
