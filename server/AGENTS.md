# Server Agent Guidelines (Infra/Config)

## Scope
- Infra/Config Agent changes under `server/` cover deployment wiring, configuration defaults, and server-specific runtime hooks.
- Avoid altering core application logic (Backend Agent) or database schemas (Data/DB Agent).

## Required checks before merge
- Run route-level tests relevant to server behaviors: `pytest tests/server/test_conversation_server_routes.py tests/server/test_task_routes.py tests/server/test_job_routes.py`.
- When modifying server startup/config integration, run `pytest tests/test_setup_controller.py` to validate configuration flows.

## Change management for API or schema updates
- Include consumer impact notes in PRs and commits (e.g., affected clients, route behavior changes, deprecations, and compatibility windows).
- Provide migration and rollback steps for any schema or contract changes, and gate risky changes behind feature flags or config switches to enable staged rollouts.
- Surface pagination defaults, response limits, and any new index requirements when modifying server-facing query patterns so downstream consumers and persistence owners can prepare.

## Performance and data safety
- Preserve pagination, maximum result limits, and bounded payload sizes on route handlers; highlight indexing needs or query-shape changes to the Data/DB Agent when relevant.
- Do not introduce secrets or allow production configuration drift in committed defaults; coordinate with Infra/Config for environment-specific overrides.

## Traceability
- Record API and schema changes against the subsystem ledger at [`../docs/_audit/inventory.md`](../docs/_audit/inventory.md) and reference the alignment posture in [`../docs/_audit/alignment-report.md`](../docs/_audit/alignment-report.md).

## Coordination
- Apply the **non-overlap principle**: keep infrastructure/configuration updates separate from backend or persistence changes. Coordinate on cross-cutting work and run the combined required tests.
- Use the shared workflow in [`../docs/contributing/agent-workflow.md`](../docs/contributing/agent-workflow.md) to capture intent, verify guardrails, align designs, respect execution constraints, and document validations for handoff.
