---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2025-12-21
source_of_truth: docs/_audit/linking-and-sources.md
---

# Refactored documentation style guide

This guide defines the required metadata and structure for refactored ATLAS documentation. Apply it to every refactored page as the first block in the file.

## Required front matter

Use YAML front matter at the top of each Markdown file to declare audiences and currency:

```yaml
---
audience: <comma-separated roles>
status: <draft|in_review|published|deprecated>
last_verified: <YYYY-MM-DD>
# Optional: identify the upstream owner for the content
source_of_truth: <team, doc, or system>
---
```

- **audience**: Primary readers (e.g., "operators, backend developers").
- **status**: Content maturity. Use `draft` for in-progress refactors, `in_review` when awaiting approval, `published` when stable, and `deprecated` when superseded.
- **last_verified**: Date the content was last checked against code/config behavior.
- **source_of_truth** *(optional)*: The canonical owner or system for the information (for example, a configuration module or API surface). Include it when the page summarizes another source.

If front matter is not supported (for example, by a downstream renderer), replace it with a Markdown header block immediately under the title using the same fields:

```
**Audience**: ...  
**Status**: ...  
**Last verified**: ...  
**Source of truth**: ... (optional)
```

## Landing page application

- Apply the front matter block to landing pages that orient readers (for example, inventories, glossaries, and alignment reports). Landing pages should list their audiences explicitly to clarify ownership expectations.
- Ensure the `last_verified` date reflects the current review and update it whenever content or code references change.

## Rollout plan for existing pages

1. **Seed templates**: Copy this front matter block into shared templates and content scaffolds in `docs/_audit` so new pages start compliant.
2. **Prioritize high-traffic docs**: Retrofit landing pages and README-linked docs first (those flagged as high-traffic in `inventory.md`).
3. **Batch updates by area**: Update related clusters together (e.g., all `ops/` runbooks), verifying `last_verified` while aligning with code references.
4. **Track completion**: Mark retrofits in `inventory.md` with updated `last_updated_hint` values and note outstanding sections in `architecture-alignment-report.md` where docs lag code.
5. **Spot-check**: After batches, run link validation or quick previews to confirm front matter renders correctly in the chosen publishing pipeline.
