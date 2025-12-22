---
audience: Documentation maintainers and scoped agents
status: in_review
last_verified: 2026-02-26
source_of_truth: docs/contributing/agent-owners.md
---

# Subsystem owner registry

This registry lists primary owners for audited documentation areas, the `_audit` folders where their inventories live, expected cadences, and who to contact if escalations are required. Keep this table aligned with the `owner` and `next_review` fields in the `_audit` inventories.

| Subsystem | `_audit` folder | Owner | Next review | Cadence | Escalation contact |
| --- | --- | --- | --- | --- | --- |
| Security checklist coverage | `docs/_audit/` (see `security-audit-checklist.md`) | @security | 2026-05-31 | Quarterly | @security |
| Architecture overview docs | `docs/_audit/` (see inventory entry for `architecture-overview.md`) | @docs-core | 2026-04-20 | Quarterly | @docs-core |
| Server API reference | `docs/_audit/` (see inventory entry for `server/api.md`) | @docs-core | 2026-03-25 | Monthly | @docs-core |
| Ops runbook index | `docs/_audit/` (see inventory entry for `ops/README.md`) | @docs-core | 2026-07-31 | Quarterly | @docs-core |

## Update process

- When `_audit` inventory entries change owners or `next_review` dates, update this table in the same commit.
- Escalation contacts should match the owning group unless a rotation dictates otherwise; record the paging contact that can act on missed cadences.
