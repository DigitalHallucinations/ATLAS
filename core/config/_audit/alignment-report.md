---
audience: Config and runtime owners
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Alignment and risk report

> See the [configuration audit README](./README.md) for cadence guidance, and keep [`inventory.md`](./inventory.md) aligned with the findings below.

This report captures configuration risks, observed drift, and remediation plans. Cross-link entries to inventory rows and source code references per [`linking-and-sources.md`](./linking-and-sources.md).

## Current risks

| claim_or_area | risk_summary | owner | due_date | status | related_inventory_row |
| --- | --- | --- | --- | --- | --- |
| Retention and vector defaults | Persistence TTLs and vector store toggles may diverge between defaults and deployed settings. | @config-owners | 2026-03-08 | Tracking | ATLAS/config/persistence.py |
| Env override precedence | Config manager precedence may not match setup controller expectations for CLI vs env overrides. | @config-owners | 2026-03-15 | Tracking | ATLAS/config/config_manager.py |

## Drift findings

| path | drift_observed | source_of_truth | remediation_owner | target_pr | due_date |
| --- | --- | --- | --- | --- | --- |
| ATLAS/config/persistence.py | Retention TTL defaults differ from values shipped with server deployments. | `config.yaml`; `modules/conversation_store/repository.py` | @config-owners | link-to-issue-or-PR | 2026-03-25 |
| ATLAS/config/tooling.py | Sandbox default resource limits may not match current tool runtime guardrails. | `modules/Tools/Base_Tools/*`; deployment settings | @config-owners | link-to-issue-or-PR | 2026-03-22 |
| ATLAS/config/config_manager.py | Ordering for env vs file overrides not fully documented in setup flows. | `ATLAS/setup/controller.py`; `scripts/setup_atlas.py` | @config-owners | link-to-issue-or-PR | 2026-03-20 |

## Remediation plan

- [ ] Confirm owners and due dates for each row above.
- [ ] Update configuration defaults and associated docs as fixes land; link the commits/issues here.
- [ ] Re-run configuration and setup checks after applying fixes.
- [ ] Mark `alignment_status` in [`inventory.md`](./inventory.md) as `Aligned` once verification is complete.
