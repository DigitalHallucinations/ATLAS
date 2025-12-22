# Testing Agent Guidelines

## Scope and boundaries
- Testing Agent changes stay within the `tests/` tree. Do **not** modify application logic, configs, or assets outside this directory.
- Prefer adding or updating test fixtures, parametrizations, and assertions over changing production code to make tests pass.

## Fixtures and stability
- Reuse shared fixtures in `tests/conftest.py` and module-level `conftest.py` files; add new fixtures only when reuse is impractical.
- Keep fixtures side-effect free and lightweight; avoid network calls and long sleeps. Use fakes/mocks to isolate external dependencies.
- Seed randomness in tests and helpers (for example, `random.seed(0)` or `np.random.seed(0)`) to minimize flakes.

## Flake reduction and coverage goals
- Aim for deterministic assertions and bounded timeouts; prefer polling helpers over fixed `sleep` calls.
- Target coverage to exercise critical contracts such as persona schema validation and setup flows; see `docs/_audit/inventory.md` rows for `docs/Personas.md` and `docs/setup-wizard.md`, and the related risks in `docs/_audit/alignment-report.md`.
- When tightening coverage, focus on high-churn areas (orchestration, persistence boundaries, provider adapters) and keep UI tests resilient to widget timing.

## Required validation commands before merge
- Run the persona schema guardrail: `pytest tests/test_persona_schema.py`.
- Run the full suite (parallelization optional): `pytest`.
- For coverage work, prefer `pytest --cov=ATLAS --cov=modules --cov-report=term-missing` and address meaningful gaps before merging.
