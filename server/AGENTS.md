# Server Agent Guidelines (Infra/Config)

## Scope
- Infra/Config Agent changes under `server/` cover deployment wiring, configuration defaults, and server-specific runtime hooks.
- Avoid altering core application logic (Backend Agent) or database schemas (Data/DB Agent).

## Required checks before merge
- Run route-level tests relevant to server behaviors: `pytest tests/server/test_conversation_server_routes.py tests/server/test_task_routes.py tests/server/test_job_routes.py`.
- When modifying server startup/config integration, run `pytest tests/test_setup_controller.py` to validate configuration flows.

## Coordination
- Apply the **non-overlap principle**: keep infrastructure/configuration updates separate from backend or persistence changes. Coordinate on cross-cutting work and run the combined required tests.
