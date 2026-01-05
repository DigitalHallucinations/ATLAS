# ATLAS Agent Guidelines (Backend)

## Scope

- Backend Agent changes in this subtree cover application orchestration and runtime logic under `core/`.
- Avoid direct persistence-layer edits (handled by Data/DB Agent) and infrastructure configuration (handled by Infra/Config Agent).

## Required checks before merge

- Run persona validation: `pytest tests/test_persona_schema.py`.
- Run backend sanity tests most relevant to ATLAS wiring, e.g., `pytest tests/test_setup_controller.py tests/test_setup_cli.py` when touching setup flows.
- If changes affect runtime entry points, run the full suite: `pytest`.

## Coordination

- Apply the **non-overlap principle**: keep backend logic changes distinct from Data/DB or Infra/Config scopes. For cross-cutting work, coordinate and run the union of required tests.
- Follow the shared workflow in [`../docs/contributing/agent-workflow.md`](../docs/contributing/agent-workflow.md) to capture intent, confirm guardrails, align design, apply execution constraints, run required validations, and provide traceable handoffs.
