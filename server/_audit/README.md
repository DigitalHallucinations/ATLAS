---
audience: Infra/Config owners and API reviewers
status: active
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Server audit workspace

This folder tracks alignment for the HTTP gateway and server wiring. It centralizes the inventory, alignment report, glossary linkage, and process references so infra reviewers can keep routing, auth, and streaming behaviors consistent with configuration defaults.

## Purpose and scope

- **Purpose**: Provide a consistent starting point for server audits, including inventory tracking, alignment validation, and citation rules for gateway and route wiring.
- **Scope**: Covers code under `server/`, including the HTTP gateway layer and shared server initialization hooks. Coordinate with Backend and Data/DB owners for route contract or persistence-related drift.
- **Cadence guidance**: Run quarterly reviews for gateway behavior, with monthly checks during periods of API churn or configuration changes.

## Audit checklist

1. Confirm owners in [`inventory.md`](./inventory.md) and [`alignment-report.md`](./alignment-report.md); update when route reviewers rotate.
2. Capture new endpoints or transport behaviors in the inventory with code references per [`linking-and-sources.md`](./linking-and-sources.md).
3. Set next review dates in [`inventory.md`](./inventory.md) after each update, prioritizing areas with recent auth or streaming changes.
4. Keep terminology and metadata aligned with [`style-guide.md`](./style-guide.md).
5. Cross-link persistence or backend dependencies to the relevant audits when logging drift.
