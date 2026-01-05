# Modules Agent Guidelines (Backend)

## Scope

- Backend Agent work in `modules/` covers application logic, service orchestration, and integrations routed through module helpers.
- Defer to more specific AGENTS files (e.g., `modules/conversation_store/AGENTS.md`) for persistence or domain-specialized guidance.

## Required checks before merge

- Run targeted module tests or `pytest` when changing shared module behavior or contracts.

## Change management for API or schema updates

- Capture consumer impact notes for any API or schema change (affected callers, breaking/compatibility details, rollout expectations) in commit/PR descriptions.
- Provide migration and rollback steps, and use feature flags or configuration switches to stage risky behavior changes.
- When query or payload shapes change, document pagination defaults, response limits, and any new index requirements to keep downstream consumers aligned.

## Performance and data safety

- Enforce pagination, bounded limits, and index-aware access patterns on module entry points; call out expected indexes when adjusting query shapes and coordinate with Data/DB owners.
- Do not introduce secrets or embed production configuration values in defaults; keep environment-specific overrides out of source control and align with Infra/Config guidance.

## Traceability

- Log API/schema changes in the subsystem ledger at [`../docs/_audit/inventory.md`](../docs/_audit/inventory.md) and reference alignment details in [`../docs/_audit/alignment-report.md`](../docs/_audit/alignment-report.md).
