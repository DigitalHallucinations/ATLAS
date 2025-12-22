# Docs Agent Guidelines

## Scope
- Documentation changes in this subtree are owned by the Docs Agent. Do not modify application code, tests, or configuration from here.
- Refactors should prefer `docs/_audit/` when restructuring reference material.

## Required checks before merge
- Ensure `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md` are updated when doc refactors affect canonical references.
- Perform a spell/format pass if tooling is available; no automated tests are required unless specified by deeper nested instructions.

## Coordination
- Honor the **non-overlap principle**: keep doc updates separate from code changes. For cross-cutting efforts (e.g., adding docs for a backend change), coordinate commits with the relevant agent and ensure the PR describes which scopes were updated.
