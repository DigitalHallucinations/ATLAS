---
audience: Infra/Config owners and API reviewers
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Alignment and risk report

> See the [server audit README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.

This report captures server-layer risks, observed drift, and remediation plans. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| Gateway websocket stability | Reconnect and timeout handling may not reflect current streaming defaults. | @infra-core | 2026-03-05 | Tracking | server/http_gateway.py |
| Auth + config integration | Startup wiring assumes env defaults that may diverge from `config.yaml` overrides. | @infra-core | 2026-03-12 | Tracking | server/__init__.py |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| server/http_gateway.py | Streaming timeout and reconnect policies differ from documented server route expectations. | `modules/Server/routes.py`; `config.yaml` | @infra-core | link-to-issue-or-PR | 2026-03-22 |
| server/__init__.py | Gateway initialization does not surface new CORS and auth toggles by default. | `config.yaml`; `ATLAS/config/config_manager.py` | @infra-core | link-to-issue-or-PR | 2026-03-18 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update gateway wiring after validating timeout and auth defaults; link the commits/issues here.
- [ ] Re-run route-level tests after applying fixes.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
