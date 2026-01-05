---
audience: Config and runtime owners
status: active
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Configuration audit workspace

This folder tracks alignment for configuration modules under `ATLAS/config/`, including persistence defaults, tooling settings, and config management. It centralizes the inventory, alignment report, glossary linkage, and process references so configuration owners can keep runtime defaults synchronized with docs and infra expectations.

## Purpose and scope

- **Purpose**: Provide a consistent starting point for configuration audits, including inventory tracking, alignment validation, and citation rules for persistence/tooling defaults.
- **Scope**: Covers config modules under `ATLAS/config/`, with cross-links to server and backend behavior that consume these settings. Coordinate with Infra/Config and Backend owners when detecting drift in downstream consumers.
- **Cadence guidance**: Run quarterly reviews for config defaults, with monthly checks for high-churn areas like persistence and tooling integrations.

## Audit checklist

1. Confirm owners in [`inventory.md`](./inventory.md) and [`alignment-report.md`](./alignment-report.md); update when config reviewers rotate.
2. Capture new configuration toggles or defaults in the inventory with references per [`linking-and-sources.md`](./linking-and-sources.md).
3. Set next review dates in [`inventory.md`](./inventory.md) after each update, prioritizing modules feeding persistence or gateway behavior.
4. Keep terminology and metadata aligned with [`style-guide.md`](./style-guide.md).
5. Cross-link downstream impacts to server or module audits when logging drift.
