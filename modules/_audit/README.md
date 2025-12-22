---
audience: Backend and data service owners
status: active
last_verified: 2026-02-20
source_of_truth: ./style-guide.md
---

# Modules audit workspace

This folder tracks alignment for backend services, orchestration, and persistence modules under `modules/`. It centralizes the inventory, alignment report, glossary linkage, and process references to keep high-churn services and data stores in sync with runtime behaviors.

## Purpose and scope

- **Purpose**: Provide a consistent starting point for backend and data audits, including inventory tracking, alignment validation, and citation rules across orchestration, persona schemas, and stores.
- **Scope**: Covers backend modules under `modules/`, including orchestration flows, persona schema enforcement, and persistence services such as conversation, task, and job stores. Coordinate with Infra/Config when route or configuration drift is identified.
- **Cadence guidance**: Run quarterly reviews for orchestration and schema changes, and monthly reviews for persistence surfaces with higher drift risk.

## Audit checklist

1. Confirm owners in [`inventory.md`](./inventory.md) and [`alignment-report.md`](./alignment-report.md); update when service owners rotate.
2. Capture newly added services or schema changes in the inventory with code references per [`linking-and-sources.md`](./linking-and-sources.md).
3. Record next review dates whenever rows change, prioritizing stores and schedulers with recent PRs.
4. Keep terminology and metadata aligned with [`style-guide.md`](./style-guide.md).
5. Cross-link API or configuration dependencies to the relevant Infra/Config audits when logging drift.
