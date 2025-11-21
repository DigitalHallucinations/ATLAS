# ATLAS Agent Guidelines

## Roles
- **Docs Agent**: Focuses solely on documentation tasks within `docs/`.
- **Testing Agent**: Adds or updates tests in `tests/` without modifying application logic elsewhere.
- **Security Agent**: Reviews configurations for security issues while avoiding changes to secrets or database credentials.

## Tech stack & entry points
- **Runtime**: Python 3.10+.
- **UI shell**: Run `python3 main.py` to start the GTK shell.
- **Persona validation**: Execute `pytest tests/test_persona_schema.py`.
- **Full test suite**: Run `pytest`.
- **Optional environment setup**: `python3 scripts/install_environment.py --with-accelerators`.

## File-scope boundaries
- Docs agent edits are limited to `docs/`.
- Test agent writes tests under `tests/` only and must not alter source logic.
- Security agent confines changes to configuration reviews, never touching secrets or database credentials.

## Style references
- Persona JSON schema lives at `modules/Personas/schema.json`; reference it for schema-aligned changes.
- Tool/skill manifests should follow existing patterns in the repository and keep manifest expectations consistent.
- Documentation should use clear Markdown headings, ordered/unordered lists where appropriate, and concise code fences for commands.

## Safety & validation rules
- Do not introduce secrets or modify production configuration values.
- Run persona schema tests and the full test suite (`pytest tests/test_persona_schema.py` and `pytest`) before submitting PRs.
- Ensure commands and changes align with ATLAS conventions before final review.
