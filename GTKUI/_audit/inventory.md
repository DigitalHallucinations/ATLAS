---
audience: GTK shell owners and release reviewers
status: draft
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Subsystem audit inventory

> Navigation: Start with the [GTKUI audit README](./README.md) for cadence guidance and cross-links to related findings.

| path | owner | last_audited | alignment_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| GTKUI/sidebar.py | @ui-core | 2026-02-20 | Needs review | Navigation badges rely on task/job payload shape changes; confirm against latest API responses. | 2026-05-20 | Track alongside Sidebar routing drift in [`alignment-report.md`](./alignment-report.md#drift-findings). |
| GTKUI/Setup/first_run.py | @ui-core | 2026-02-20 | Aligned | Setup wizard steps match current setup controller responses. | 2026-06-20 | Re-verify after setup controller changes land. |
| GTKUI/Settings/backup_settings.py | @ui-core | 2026-02-20 | Needs review | Backup targets follow legacy retention defaults; confirm parity with server endpoints. | 2026-04-15 | Pair with server backup route review noted in [`alignment-report.md`](./alignment-report.md#current-risks). |

## Legend

- **path**: Source file under `GTKUI/` being tracked for audit coverage.
- **owner**: Primary maintainer or reviewer group; prefer team aliases.
- **last_audited**: Date of the last substantive review; include short parenthetical notes if the review added context.
- **alignment_status**:
  - `Aligned`: Behavior or docs reflect current source of truth.
  - `Needs review`: Partial confidence; schedule near-term verification.
  - `Needs overhaul`: Significant drift or missing coverage; prioritize remediation.
- **gaps_found**: Brief description of drift, missing claims, or link debt observed during the last review.
- **next_review**: Target date based on the GTKUI cadence described in [`README.md`](./README.md#audit-checklist).
- **notes**: Freeform cross-links to remediation issues, PRs, or [`alignment-report.md`](./alignment-report.md) entries.
