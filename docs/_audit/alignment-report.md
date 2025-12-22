---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-02-22
source_of_truth: docs/_audit/style-guide.md
---

# Alignment and risk report

> See the [audit workspace README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.
> For security-related reviews, drive findings from the [`security-audit-checklist.md`](./security-audit-checklist.md) and log outcomes in this report.

This report captures documentation drift risks and remediation plans for high-traffic pages. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Review cadence

| area | owner | frequency | next_review | notes |
| --- | --- | --- | --- | --- |
| Server API and gateway docs | @docs-core | Monthly (fast-moving) | 2026-03-25 | Keep parity with websocket reconnect and retention endpoints. |
| Architecture overview and diagrams | @docs-core | Quarterly | 2026-04-20 | Align diagrams with orchestration and storage pipeline updates. |
| UI/setup flow narratives | @docs-core | Quarterly | 2026-05-31 | Ensure GTK setup/UI screens match controller responses and assets. |

## API / schema / UI contract risks

| claim_or_area | risk_summary | severity | owner | due_date | status | related_inventory_row | issue_link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Architecture overview diagrams | Diagrams and runtime paths trail orchestration updates for background workers and gateways. | High | @docs-core | 2026-03-15 | Tracking | docs/architecture-overview.md | link-to-issue-or-PR |
| Server API parity (REST + websocket) | Route listings may omit websocket reconnect, backup, and retention updates from the latest server changes. | High | @docs-core | 2026-03-25 | Tracking | docs/server/api.md | link-to-issue-or-PR |
| UI/setup flow contracts | GTK setup pages and controller responses may drift from documented field order and validation rules. | Medium | @docs-core | 2026-05-31 | Monitoring | docs/setup-wizard.md | link-to-issue-or-PR |

## Drift findings

| path | drift_observed | severity | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- | --- |
| docs/architecture-overview.md | Component diagram and flow descriptions omit new retention/vector pipelines. | High | `modules/orchestration/`; `modules/conversation_store/`; `ATLAS/config/persistence.py` | @docs-core | link-to-issue-or-PR | 2026-03-22 |
| docs/server/api.md | Route table misses websocket reconnect options and backup endpoints. | High | `server/http_gateway.py`; `modules/Server/routes.py` | @docs-core | link-to-issue-or-PR | 2026-03-25 |
| docs/setup-wizard.md | Validation order and preset behaviors may not match the latest controller responses. | Medium | `ATLAS/setup/controller.py`; `GTKUI/Setup/`; `main.py` | @docs-core | link-to-issue-or-PR | 2026-05-31 |

## Reporting and tracking expectations

- Capture every checklist item (especially DLP, access controls, and transport security) with a status (`Aligned`, `Needs update`, or `Blocked`) in the tables above.
- Link findings to the relevant source of truth (code/config) and reference issues or pull requests for remediation; ensure owners and due dates are populated.
- Reflect status updates back into [`inventory.md`](./inventory.md) so the ledger mirrors the latest alignment outcomes and next review dates.
- When closing a finding, note the verification evidence (test commands, config diffs, or doc updates) directly in the affected table row.

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update documentation and diagrams as fixes land; link the commits/issues here.
- [ ] Re-run link checks after applying fixes and refresh inventory statuses.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
