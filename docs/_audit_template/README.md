---
audience: Documentation maintainers, subsystem owners, and reviewers
status: template
last_verified: 2024-05-21
source_of_truth: docs/_audit_template/style-guide.md
---

# Audit template workspace

This folder is a ready-to-copy audit workspace for any subsystem that needs consistent documentation hygiene. It centralizes the audit inventory, alignment report, glossary linkage, and process references so teams can track drift and remediation in one place.

## Purpose and scope

- **Purpose**: Provide a consistent starting point for subsystem-level documentation audits, including inventory tracking, alignment validation, and citation rules.
- **Scope**: Use this template when creating or refreshing a subsystem audit under `docs/<subsystem>/_audit`. Keep edits scoped to the copied workspace to avoid cross-subsystem coupling.
- **Cadence guidance**: Start with a quarterly comprehensive review plus lightweight monthly link and metadata checks. Increase cadence for high-change areas or when alignment gaps emerge.

## How to instantiate (checklist)

1. Copy the template: `cp -R docs/_audit_template docs/<subsystem>/_audit`.
2. Update ownership: set the primary owner in [`inventory.md`](./inventory.md) and [`alignment-report.md`](./alignment-report.md); confirm reviewers in front matter.
3. Set cadence: choose dates for quarterly audits and monthly spot checks; record them in the `next_review` column of [`inventory.md`](./inventory.md).
4. Seed the inventory: add initial rows for the subsystem’s pages, linking to code touchpoints per [`linking-and-sources.md`](./linking-and-sources.md).
5. Align style: ensure every page uses the front matter and naming rules in [`style-guide.md`](./style-guide.md).

## Template contents and cross-links

- [`inventory.md`](./inventory.md): Audit table with owners, alignment status, and review cadence.
- [`alignment-report.md`](./alignment-report.md): Skeleton for risks, drift findings, and remediation owners.
- [`style-guide.md`](./style-guide.md): Required front matter, naming conventions, and recording expectations.
- [`linking-and-sources.md`](./linking-and-sources.md): Canonical source rules, citation tips, and link-check guidance.

## Notes on copying or renaming

- When you instantiate this template, rename the folder to `_audit` within the target subsystem (`docs/<subsystem>/_audit`) so paths resolve correctly.
- After copying, update internal links if you add nested folders; relative links here assume the files stay co-located.
- If you later move the audit workspace, re-run link checks and update any stored `source_of_truth` references as described in [`linking-and-sources.md`](./linking-and-sources.md).
