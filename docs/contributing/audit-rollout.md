---
audience: Documentation maintainers and subsystem owners
status: draft
last_verified: 2026-02-27
source_of_truth: docs/_audit_template; docs/_audit/inventory.md
---

# Audit Rollout Plan

Use this checklist when onboarding a subsystem into the documentation audit program. It keeps owner assignments, cadence, and traceability aligned with the shared `_audit_template` and audit workspace.

## Onboard a subsystem
1. **Copy the template** – Duplicate `_audit_template` into a new folder under `docs/_audit/<subsystem>` or a nearby staging path. Keep the directory name consistent with the subsystem name to simplify cataloging.
2. **Set owners and cadence** – Fill the front matter with the subsystem owner and review cadence pulled from `docs/contributing/agent-owners.md`. Add the audit to `docs/_audit/inventory.md` and the alignment-report queues as soon as the file is created.
3. **Run the first pass** – Complete the template’s initial checklist (front matter, link checks, code anchors) and record findings in the new audit page plus `docs/_audit/architecture-alignment-report.md` when they touch architecture claims.
4. **Schedule reminders** – Open calendar reminders or tracker tasks for the next review date and any follow-ups. Include escalation contacts from the owner registry for overdue items.

## Pre-close checklist
- [ ] **AGENTS alignment** – Confirm scope and required checks from repository `AGENTS.md` files before and after copying templates. Note any scope nuances in the new audit page.
- [ ] **CI coverage** – Ensure the subsystem’s standard CI gates are green (persona schema + `pytest` for backend changes, relevant route suites for server work, doc previews/link checks when available). Record executed commands in the audit notes.
- [ ] **Inventory linkage** – Add or update rows in `docs/_audit/inventory.md` and carry drift items into `docs/_audit/architecture-alignment-report.md` so the new audit shows up in the dashboards.

## Communication and timeline
- **Announcement channels** – Post the rollout note in `#docs-maint` (or the subsystem’s Slack channel) and create an issue in the tracker with the subsystem label and owner tag.
- **Timeline expectations** – Day 0: announce + create the tracker issue. Day 2: owner confirmed and cadence set. Day 5: first audit pass completed and recorded. Day 7: reminders scheduled and follow-up tasks filed. Day 14: validate follow-up closures or extend cadence with a new reminder.
