---
audience: Infra/Config owners and API reviewers
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Subsystem audit inventory

> Navigation: Start with the [server audit README](./README.md) for cadence guidance and cross-links to related findings.

| path | owner | last_audited | alignment_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| server/http_gateway.py | @infra-core | 2026-02-20 | Needs review | Websocket reconnect handling and streaming timeouts need validation against recent gateway changes. | 2026-04-10 | Paired with gateway drift tracking in [`alignment-report.md`](./alignment-report.md#drift-findings). |
| server/__init__.py | @infra-core | 2026-02-20 | Aligned | Startup wiring and config surface match current deployment defaults. | 2026-06-01 | Revisit after config defaults change in `config.yaml` or `ATLAS/config/`. |

## Legend

- **path**: Source file under `server/` being tracked for audit coverage.
- **owner**: Primary maintainer or reviewer group; prefer team aliases.
- **last_audited**: Date of the last substantive review; include short parenthetical notes if the review added context.
- **alignment_status**:
  - `Aligned`: Behavior or docs reflect current source of truth.
  - `Needs review`: Partial confidence; schedule near-term verification.
  - `Needs overhaul`: Significant drift or missing coverage; prioritize remediation.
- **gaps_found**: Brief description of drift, missing claims, or link debt observed during the last review.
- **next_review**: Target date based on the cadence described in [`README.md`](./README.md#audit-checklist).
- **notes**: Freeform cross-links to remediation issues, PRs, or [`alignment-report.md`](./alignment-report.md) entries.
