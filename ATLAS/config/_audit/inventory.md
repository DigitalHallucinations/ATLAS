---
audience: Config and runtime owners
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Subsystem audit inventory

> Navigation: Start with the [configuration audit README](./README.md) for cadence guidance and cross-links to related findings.

| path | owner | last_audited | alignment_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| config.yaml | @config-owners | 2026-02-20 | Needs review | Ensure prod-like defaults and credentials remain externalized; confirm documented toggle coverage. | 2026-03-20 | Treat as deployment-facing defaults; avoid embedding secrets. |
| ATLAS/config/persistence.py | @config-owners | 2026-02-20 | Needs review | Retention TTL defaults and vector store toggles may differ from deployed values. | 2026-03-25 | Coordinate with Data/DB store alignment in modules audit. |
| ATLAS/config/tooling.py | @config-owners | 2026-02-20 | Aligned | Tool sandbox defaults match current runtime expectations. | 2026-05-20 | Re-validate after new tool providers land. |
| ATLAS/config/config_manager.py | @config-owners | 2026-02-20 | Needs review | Env override precedence should be rechecked against setup controller behavior. | 2026-04-10 | Track alongside gateway wiring updates in server audit. |
| ATLAS/config/atlas_config.yaml | @config-owners | 2026-02-20 | Needs review | Validate presets avoid prod-like defaults and that toggle documentation is current. | 2026-03-20 | Keep aligned with config.yaml guidance and setup flows. |
| ATLAS/config/logging_config.yaml | @config-owners | 2026-02-20 | Aligned | Confirm log sinks/levels stay non-prod by default and remain documented. | 2026-05-20 | Revisit after logging pipeline updates or new sinks. |

## Legend

- **path**: Source file under `ATLAS/config/` being tracked for audit coverage.
- **owner**: Primary maintainer or reviewer group; prefer team aliases.
- **last_audited**: Date of the last substantive review; include short parenthetical notes if the review added context.
- **alignment_status**:
  - `Aligned`: Behavior or docs reflect current source of truth.
  - `Needs review`: Partial confidence; schedule near-term verification.
  - `Needs overhaul`: Significant drift or missing coverage; prioritize remediation.
- **gaps_found**: Brief description of drift, missing claims, or link debt observed during the last review.
- **next_review**: Target date based on the cadence described in [`README.md`](./README.md#audit-checklist).
- **notes**: Freeform cross-links to remediation issues, PRs, or [`alignment-report.md`](./alignment-report.md) entries.
