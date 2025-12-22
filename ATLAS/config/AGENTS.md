# ATLAS Config Agent Guidelines

## Scope
- Applies to configuration sources under `ATLAS/config/` and runtime defaults surfaced via `config.yaml`.
- Owned by the **Infra/Config Agent**. Coordinate with Backend or Docs owners when wiring changes affect code paths or reference material.

## Guardrails
- Do **not** change production-like defaults, secrets, or credentials in `config.yaml`, `atlas_config.yaml`, `logging_config.yaml`, or related presets. Keep sensitive or environment-specific values externalized.
- Every new or updated toggle must be documented in the relevant config docs or audit inventory at the time of change.
- Default behaviors must retain or add test coverage (e.g., targeted `pytest` cases) when toggles shift expected runtime flows.
- Record audit updates for major config files (including review cadence and owner) in `ATLAS/config/_audit/inventory.md` whenever defaults or wiring change.

## Validation
- Prefer running `pytest tests/test_persona_schema.py` and focused config-related tests when defaults move.
- Keep audit metadata (`inventory.md`, `alignment-report.md`) in sync if runtime defaults or toggle surfaces change.
