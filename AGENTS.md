# ATLAS Agent Guidelines

## Roles and writable scopes

- **UI Agent**: Works in `GTKUI/`, visual assets under `Icons/`, and UI entry points that wire GTK shell behaviors. Avoid backend or storage logic changes.
- **Backend Agent**: Owns application logic in `core/`, `atlas_provider.py`, `modules/` (excluding data/DB submodules when a Data/DB Agent is assigned), and orchestration codepaths wired through `main.py` or CLI adapters.
- **Data/DB Agent**: Handles persistence layers in `modules/conversation_store/`, `modules/task_store/`, `modules/job_store/`, associated migrations under `scripts/migrations/`, and storage-related configuration in `core/config/persistence.py`.
- **Infra/Config Agent**: Manages configuration defaults, deployment templates, and runtime wiring in `server/`, `config.yaml`, `core/config/`, and scripts under `scripts/` that change operational behaviors.
- **Docs Agent**: Focuses solely on documentation tasks within `docs/`. When refactoring docs, ensure all changes are within the scope of `docs/_audit`. Follow the audit cadence documented in `docs/_audit/` (at least weekly and after material reorganizations), including link validation and front-matter checks, and treat `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md` as the traceability sources. After any docs change, refresh `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md` to reflect the updated state.
- **Testing Agent**: Adds or updates tests in `tests/` without modifying application logic elsewhere.
- **Security Agent**: Reviews configurations for security issues while avoiding changes to secrets or database credentials.

## Tech stack & entry points

- **Runtime**: Python 3.10+.
- **UI shell**: Run `python3 main.py` to start the GTK shell.
- **Persona validation**: Execute `pytest tests/test_persona_schema.py`.
- **Full test suite**: Run `pytest`.
- **Optional environment setup**: `python3 scripts/install_environment.py --with-accelerators`.

## File-scope boundaries

- Agents must stay within their declared writable scopes. When multiple roles are active, apply the **non-overlap principle**: each change should belong to exactly one role’s scope to prevent cross-contamination of responsibilities.
- Docs Agent edits are limited to `docs/`.
- Testing Agent writes tests under `tests/` only and must not alter source logic.
- Security Agent confines changes to configuration reviews, never touching secrets or database credentials.

## Coordination & cross-cutting changes

- When work spans multiple scopes (e.g., Backend + Data/DB), sequence commits by scope or collaborate with the relevant role to ensure ownership and reviews remain clear.
- Use nested `AGENTS.md` files to refine constraints and required checks for a subtree. Defer to the most specific file in the path of the files you touch.
- For cross-cutting refactors, document affected scopes in the PR description and run the union of required tests from each scope.
- Use the [agent owner registry](docs/contributing/agent-owners.md) to confirm who to involve for each audited subsystem and its escalation contact.

## Style references

- Persona JSON schema lives at `modules/Personas/schema.json`; reference it for schema-aligned changes.
- Tool/skill manifests should follow existing patterns in the repository and keep manifest expectations consistent.
- Documentation should use clear Markdown headings, ordered/unordered lists where appropriate, and concise code fences for commands.

## Safety & validation rules

- Security Agents focus on configuration, policy, and deployment-safety reviews (for example, transport security defaults, access controls, and data-loss-prevention policies) and must not alter secrets, credentials, or other sensitive values.
- Do not introduce secrets or modify production configuration values.
- Run persona schema tests and the full test suite (`pytest tests/test_persona_schema.py` and `pytest`) before submitting PRs.
- Ensure commands and changes align with ATLAS conventions before final review.
- Follow the shared workflow in [`docs/contributing/agent-workflow.md`](docs/contributing/agent-workflow.md) for intent capture, guardrail discovery, design alignment, execution constraints, validation, traceability, and handoff.
