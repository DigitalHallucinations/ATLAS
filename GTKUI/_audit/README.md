---
audience: GTK shell owners and release reviewers
status: active
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# GTKUI audit workspace

This folder tracks UI-alignment work for the GTK desktop shell. It centralizes the inventory, alignment report, glossary linkage, and process references so UI contributors can keep settings panels, managers, and navigation in sync with current behaviors.

## Purpose and scope

- **Purpose**: Provide a consistent starting point for GTKUI audits, including inventory tracking, alignment validation, and citation rules for window controllers and GTK wiring.
- **Scope**: Covers code under `GTKUI/` that ships the desktop shell, including setup, sidebar navigation, and settings surfaces. Coordinate with backend owners before logging issues that originate from API contract changes.
- **Cadence guidance**: Run a quarterly comprehensive review of GTK navigation and dialogs, plus lightweight monthly checks for recently merged UI features or changed server contracts.

## Audit checklist

1. Confirm owners in [`inventory.md`](./inventory.md) and [`alignment-report.md`](./alignment-report.md); update when rotating maintainers or reviewers.
2. Capture new screens or flows in the inventory with code references per [`linking-and-sources.md`](./linking-and-sources.md).
3. Schedule the next review date in [`inventory.md`](./inventory.md) whenever a row is updated.
4. Keep GTK style rules aligned with [`style-guide.md`](./style-guide.md) to ensure front matter and terminology stay consistent.
5. When UI changes rely on new backend behaviors, log the dependency in [`alignment-report.md`](./alignment-report.md) and link to the owning issue or PR.
