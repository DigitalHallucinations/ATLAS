---
audience: Backend and data service owners
status: draft
last_verified: 2026-01-03
source_of_truth: ./style-guide.md
---

# Alignment and risk report

> See the [modules audit README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.

This report captures backend and data-service risks, observed drift, and remediation plans. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| StorageManager migration | All storage operations now route through StorageManager; legacy fallback paths removed. | @data-eng | 2026-02-01 | Resolved | modules/storage/manager.py |
| SOTA RAG upgrade | Implemented hybrid retrieval, query routing, evidence gating, hierarchical chunking, context compression, caching, evaluation harness, and observability. | @data-eng | 2026-01-03 | Resolved | modules/storage/retrieval/, modules/storage/evaluation/, modules/storage/observability/ |
| Scheduler retry semantics | Recent retry/backoff changes may diverge between scheduler defaults and documented behavior. | @backend-core | 2026-03-10 | Tracking | modules/orchestration/job_scheduler.py |
| Persona schema/tool maps | Tool manifest updates may not fully reflect schema enforcement paths. | @persona-maintainers | 2026-03-05 | Tracking | modules/Personas/schema.json |
| Conversation retention hooks | Repository retention callbacks may not align with configured TTLs and vector pipelines. | @data-eng | 2026-03-08 | Tracking | modules/conversation_store/repository.py |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| modules/orchestration/job_scheduler.py | Backoff defaults differ from current job configuration docs; jitter handling unclear. | `ATLAS/config/persistence.py`; scheduler config loaders | @backend-core | link-to-issue-or-PR | 2026-03-22 |
| modules/conversation_store/repository.py | Retention worker expectations for vector cleanup differ from repository hooks. | `modules/background_tasks/retention.py`; `ATLAS/config/persistence.py` | @data-eng | link-to-issue-or-PR | 2026-03-25 |
| modules/Personas/schema.json | New optional fields for skill metadata are missing from published schema notes. | `modules/Tools/tool_maps/functions.json`; persona manifest loaders | @persona-maintainers | link-to-issue-or-PR | 2026-03-12 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update documentation and schema references as fixes land; link the commits/issues here.
- [ ] Re-run persona schema and scheduler checks after applying fixes.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
