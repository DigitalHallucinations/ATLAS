---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2025-12-21
source_of_truth: docs/_audit/style-guide.md
---

# Linking and canonical sources

Establish consistent source-of-truth rules and cross-linking expectations for refactored documentation.

## Canonical sources

- **Developer docs and code** are the source of truth for technical behavior. When summarizing configuration or APIs, always point back to the owning modules or schemas.
- **Generated references** (for example, tool catalogs) should link to their generation inputs (schemas, manifests, or scripts) rather than duplicating data.
- Use the `source_of_truth` front matter to declare where the definitive information lives (module path, schema, or owning team).

## Linking expectations

- Prefer deep links to specific files or sections in the repository (e.g., module paths, schema files) instead of repeating long descriptions.
- Cross-link related refactored pages to reduce duplication. Landing pages should connect to detailed subpages rather than restating their content.
- When referencing configuration, include a pointer to the authoritative table or schema rather than copying values.

## No-duplication policy

- Avoid copying configuration tables, schema fields, or enumerations across multiple pages. Maintain one canonical table per topic and link to it elsewhere.
- If a temporary duplication is unavoidable, mark the derivative section with a clear note and open a follow-up task to consolidate it.
- For configuration examples, keep values minimal and refer readers to the canonical reference for exhaustive options.
