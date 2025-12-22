---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-02-20
source_of_truth: docs/_audit/style-guide.md
---

# Alignment and risk report

> See the [audit workspace README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.

This report captures documentation drift risks and remediation plans for high-traffic pages. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| Architecture overview diagrams | Diagrams and runtime paths trail orchestration updates for background workers and gateways. | @docs-core | 2026-03-15 | Tracking | docs/architecture-overview.md |
| Server API parity | REST route listings may omit websocket and retention updates from the latest server changes. | @docs-core | 2026-03-25 | Tracking | docs/server/api.md |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| docs/architecture-overview.md | Component diagram and flow descriptions omit new retention/vector pipelines. | `modules/orchestration/`; `modules/conversation_store/`; `ATLAS/config/persistence.py` | @docs-core | link-to-issue-or-PR | 2026-03-22 |
| docs/server/api.md | Route table misses websocket reconnect options and backup endpoints. | `server/http_gateway.py`; `modules/Server/routes.py` | @docs-core | link-to-issue-or-PR | 2026-03-25 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update documentation and diagrams as fixes land; link the commits/issues here.
- [ ] Re-run link checks after applying fixes and refresh inventory statuses.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
