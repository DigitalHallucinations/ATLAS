---
audience: Documentation maintainers and subsystem reviewers
status: template
last_verified: 2024-05-21
source_of_truth: docs/_audit_template/style-guide.md
---

# Alignment and risk report

> See the [template README](./README.md) for cadence guidance and the instantiation checklist, and use [`inventory.md`](./inventory.md) to keep row-level status in sync with these findings.

This report captures subsystem-level risks, observed drift, and remediation plans. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| Example: “Task scheduler retries” | Potential drift between docs and `modules/orchestration/job_scheduler.py` retry defaults. | @owner_handle | 2024-06-15 | Tracking | [`docs/<subsystem>/<page>.md`](./inventory.md#subsystem-audit-inventory) |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| docs/<subsystem>/<page>.md | Documented limits do not match current config defaults. | `ATLAS/config/<module>.py` | @owner_handle | link-to-issue-or-PR | 2024-06-30 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update docs and close gaps found during the audit; link the commits/issues here.
- [ ] Re-run link checks (see [`linking-and-sources.md`](./linking-and-sources.md#running-link-checks)) after applying fixes.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
