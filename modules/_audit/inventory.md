---
audience: Backend and data service owners
status: draft
last_verified: 2026-01-01
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
| modules/orchestration/job_scheduler.py | @backend-core | 2026-02-20 | Needs review | Retry defaults and backoff handling recently changed; confirm persisted schedules match docs. | 2026-04-15 | Track alongside scheduler drift in [`alignment-report.md`](./alignment-report.md#drift-findings). |
| modules/Personas/schema.json | @persona-maintainers | 2026-02-20 | Aligned | Schema validations reflect current persona tooling expectations. | 2026-05-20 | Re-run persona schema tests after manifest changes. |
| modules/conversation_store/repository.py | @data-eng | 2026-02-20 | Needs review | Retention hooks and vector pipeline calls may diverge from current configs. | 2026-03-30 | Pair with Data/DB alignment items in [`alignment-report.md`](./alignment-report.md#current-risks). |
| modules/job_store/service.py | @backend-core | 2026-02-20 | Aligned | Job lifecycle transitions match documented status machine. | 2026-05-01 | Reconfirm after task/job routing updates. |

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
