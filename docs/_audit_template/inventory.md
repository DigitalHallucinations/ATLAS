---
audience: Documentation maintainers and subsystem owners
status: template
last_verified: 2024-05-21
source_of_truth: docs/_audit_template/style-guide.md
---

# Subsystem audit inventory

> Navigation: Start with the [audit template README](./README.md) for instantiation steps, cadence guidance, and cross-links to related templates.

Use this table to record every page in the subsystem scope, along with ownership, audit status, and remediation notes. Align column values with the legend below, and keep code references current using the rules in [`linking-and-sources.md`](./linking-and-sources.md).

| path | owner | last_audited | alignment_status | gaps_found | next_review | notes |
| --- | --- | --- | --- | --- | --- | --- |
| docs/<subsystem>/<page>.md | @owner_handle | 2024-05-21 | Needs review | Pending code-path confirmation for new features. | 2024-08-15 | Link to related row in [`alignment-report.md`](./alignment-report.md#drift-findings). |
| docs/<subsystem>/<other-page>.md | @secondary_owner | 2024-05-21 | Aligned | No gaps noted. | 2024-11-15 | Confirmed against <module/function> per [`linking-and-sources.md`](./linking-and-sources.md#citing-source-of-truth). |

## Legend

- **path**: Markdown page under the subsystem’s `docs/` subtree (include relative links).
- **owner**: Primary maintainer or reviewer group; prefer handles that map to issue trackers or chat groups.
- **last_audited**: Date of the last substantive review; include short parenthetical notes if the review added context.
- **alignment_status**:
  - `Aligned`: Content reflects current behavior and sources of truth.
  - `Needs review`: Partial confidence; schedule near-term verification.
  - `Needs overhaul`: Significant drift or missing coverage; prioritize remediation.
- **gaps_found**: Brief description of drift, missing claims, or link debt observed during the last review.
- **next_review**: Target date based on the subsystem cadence set in the [template README](./README.md#how-to-instantiate-checklist).
- **notes**: Freeform cross-links to remediation issues, PRs, or [`alignment-report.md`](./alignment-report.md) entries.
