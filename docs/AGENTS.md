# Docs Agent Guidelines

## Scope
- Documentation changes in this subtree are owned by the Docs Agent. Do not modify application code, tests, or configuration from here.
- Refactors should prefer `docs/_audit/` when restructuring reference material and follow the audit cadence documented there (at least weekly and after significant reorganizations), including link validation and front-matter checks.

## Required checks before merge
- Treat `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md` as the traceability source for changes. Refresh both files after any documentation update to keep mappings current.
- Ensure `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md` are updated when doc refactors affect canonical references.
- Perform a spell/format pass if tooling is available; no automated tests are required unless specified by deeper nested instructions.

## Coordination
- Honor the **non-overlap principle**: keep doc updates separate from code changes. For cross-cutting efforts (e.g., adding docs for a backend change), coordinate commits with the relevant agent and ensure the PR describes which scopes were updated.
- Use the shared workflow in [`docs/contributing/agent-workflow.md`](./contributing/agent-workflow.md) to capture intent, check guardrails, align design, confirm constraints, validate changes, and provide traceable handoffs.
