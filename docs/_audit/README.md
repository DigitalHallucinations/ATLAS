---
audience: Documentation maintainers and contributors
status: draft
last_verified: 2026-02-13
source_of_truth: docs/_audit/style-guide.md
---

# Documentation audit workspace

Use this folder to coordinate documentation refactors, track alignment with the codebase, and capture audit outcomes. It contains the inventory, architecture alignment report, glossary, and process references that keep refactored docs consistent.

## Audit cadence and ownership

- **Quarterly full audit**: Review high-traffic docs, regenerate the inventory, and refresh the architecture alignment report before each quarterly release or milestone.
- **Lightweight monthly check**: Spot-check links, front matter currency, and any newly merged docs to keep drift low between full audits.
- **After major feature merges**: Revisit affected rows in `inventory.md` and corresponding sections in `architecture-alignment-report.md` to capture new behaviors and risks.

## Recording findings

1. **Inventory updates (`inventory.md`)**
   - Add or update rows when a page changes scope, audience, or code dependencies.
   - Refresh `last_updated_hint`, `rewrite_needed?`, and `alignment_risk` when refactors finish or new risks appear.
   - Keep code/module references current and prefer deep links per the linking guidance.
2. **Architecture checks (`architecture-alignment-report.md`)**
   - Log claim-by-claim confirmations or mismatches, citing the owning modules or schemas.
   - Note required follow-ups and align them with the inventory row for the same page when applicable.
3. **Metadata hygiene**
   - Update `last_verified` on any page you touch in this folder.
   - Use `source_of_truth` to point back to authoritative modules, schemas, or owning teams.

## Quick-start workflow

1. Read the process authorities: [`style-guide.md`](./style-guide.md) for required front matter and structure, and [`linking-and-sources.md`](./linking-and-sources.md) for canonical source rules.
2. Validate front matter on every edited page using the template in `style-guide.md`; ensure audiences, status, and dates reflect the latest review.
3. Run link checks (for example, your preferred Markdown link checker or the publishing pipeline’s link validation) and fix broken or redirected targets.
4. Record findings in `inventory.md` and `architecture-alignment-report.md`, cross-linking related sections so future reviewers can trace decisions.

## Process authority references

- [`style-guide.md`](./style-guide.md): Required metadata, tone, and landing-page expectations.
- [`linking-and-sources.md`](./linking-and-sources.md): Canonical source selection, cross-linking rules, and no-duplication policy.
