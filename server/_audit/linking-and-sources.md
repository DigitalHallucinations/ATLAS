---
audience: Documentation maintainers and subsystem reviewers
status: template
last_verified: 2024-05-21
source_of_truth: docs/_audit_template/style-guide.md
---

# Linking and sources

Use this guide to choose authoritative references, cite code locations, and validate links for the subsystem audit workspace. Pair these steps with the cadence and ownership guidance in [`README.md`](./README.md).

## Citing source of truth

- Prefer deep links to stable code paths (modules, schemas, or configuration files) rather than top-level directories.
- When citing tests or scripts, include the specific function, class, or CLI entry point.
- Cross-link every drift or risk entry in [`alignment-report.md`](./alignment-report.md) to the corresponding rows in [`inventory.md`](./inventory.md).
- Note the authoritative module or spec in the `source_of_truth` front matter field for each page; update it when ownership changes.

## Cross-linking within the audit workspace

- Keep relative links between the template files (`README.md`, `inventory.md`, `alignment-report.md`, and `style-guide.md`) so the workspace remains portable when copied.
- If you add new supporting documents, link them from `README.md` and reference them in the relevant sections here.
- After copying the template to `_audit`, validate that the links still resolve within `docs/<subsystem>/_audit/`.

## Running link checks

- Run a Markdown link checker before publishing audit updates. If no standard tool exists, use your preferred CLI checker (e.g., `markdown-link-check`) against the `_audit` folder.
- Re-run link checks after moving or renaming the audit workspace to catch broken relative links.
- Document any known false positives or intentional external redirects in the `notes` column of [`inventory.md`](./inventory.md).

## Source hygiene tips

- Avoid duplicating explanations across subsystem docs; instead, link back to the primary page and record the relationship in [`inventory.md`](./inventory.md).
- When a subsystem relies on shared components, cite the shared module and the local integration point to keep future audits traceable.
- Include PR or issue links alongside code citations to help reviewers verify remediation progress.
