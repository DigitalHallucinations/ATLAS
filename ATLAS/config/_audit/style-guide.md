---
audience: Documentation maintainers and subsystem owners
status: template
last_verified: 2024-05-21
source_of_truth: docs/_audit/style-guide.md
---

# Audit recording style guide

This guide defines how to structure audit pages, apply front matter, and name files within an instantiated `_audit` workspace. Use it alongside [`linking-and-sources.md`](./linking-and-sources.md) to keep references accurate.

## Front matter expectations

Every Markdown file in the audit workspace should start with YAML front matter that captures ownership and verification metadata. Use this template and update the values after each review:

```yaml
---
audience: Documentation maintainers and subsystem owners
status: draft        # Use draft, in-progress, or current
last_verified: 2024-05-21
source_of_truth: docs/<subsystem>/_audit/style-guide.md
---
```

- **`audience`**: Identify the readers (e.g., maintainers, reviewers, subsystem teams).
- **`status`**: Track lifecycle; update to `current` after a full review.
- **`last_verified`**: Refresh this date whenever content is revalidated.
- **`source_of_truth`**: Point to the primary references for the page (code files, schemas, or this guide).

## Naming conventions

- Keep the folder name `_audit` within each subsystem to align with links in [`README.md`](./README.md) and [`inventory.md`](./inventory.md).
- Use lowercase, hyphen-separated filenames (e.g., `alignment-report.md`, `linking-and-sources.md`).
- Prefer descriptive headings that mirror the file name and the subsystem area under review.

## Recording audits

1. Start from the [instantiation checklist](./README.md#how-to-instantiate-checklist) and confirm the cadence for this subsystem.
2. Add or update rows in [`inventory.md`](./inventory.md) after each review, updating `last_audited`, `alignment_status`, and `next_review`.
3. Capture risks and drift in [`alignment-report.md`](./alignment-report.md), linking back to the relevant inventory rows.
4. Cite code, schemas, and configuration files per [`linking-and-sources.md`](./linking-and-sources.md#citing-source-of-truth) to keep audit notes traceable.
5. When adding new pages to the subsystem, seed corresponding inventory rows and note the owner.

## Cross-subsystem consistency

- If the organization uses a shared source-of-truth for audit metadata, reference it here and mirror any required fields.
- Keep terminology aligned with the main `docs/_audit/style-guide.md` to avoid divergent patterns across subsystems.
- Before copying or renaming the workspace, ensure internal links stay relative; re-run link checks after moves.
