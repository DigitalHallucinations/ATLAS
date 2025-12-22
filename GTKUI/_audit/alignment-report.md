---
audience: GTK shell owners and release reviewers
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Alignment and risk report

> See the [GTKUI audit README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.

This report captures GTK shell risks, observed drift, and remediation plans. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| Sidebar badge counts | Recent task/job payload shape changes may desync badge totals from server pagination responses. | @ui-core | 2026-03-15 | Tracking | GTKUI/sidebar.py |
| Backup settings UI | Settings panel may still reflect legacy retention defaults instead of server-configured values. | @ui-core | 2026-03-10 | Tracking | GTKUI/Settings/backup_settings.py |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| GTKUI/sidebar.py | Badge updates assume synchronous refresh even when websocket reconnects mid-session. | `modules/Server/routes.py`; `server/http_gateway.py` | @ui-core | link-to-issue-or-PR | 2026-03-20 |
| GTKUI/Settings/backup_settings.py | Backup destination list does not display server validation errors from new retention endpoints. | `server/http_gateway.py`; `modules/Server/routes.py` | @ui-core | link-to-issue-or-PR | 2026-03-18 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update UI bindings after backend API confirmations; link the commits/issues here.
- [ ] Re-run UI smoke checks after applying fixes.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
