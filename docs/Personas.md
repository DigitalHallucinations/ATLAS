# Persona definitions

The personas shipped with ATLAS live under `modules/Personas/<Persona Name>/Persona/<Persona Name>.json`.
Each file is validated against [`modules/Personas/schema.json`](../modules/Personas/schema.json), a
[JSON Schema](https://json-schema.org/) contract that guards the most important parts of every
persona definition.

## Schema overview

The schema enforces that each persona document:

- wraps persona entries in a top-level `persona` array (only the first entry is used today),
- provides a `name` and the `content.start_locked` / `content.editable_content` /
  `content.end_locked` fields, and
- lists any `allowed_tools` as canonical tool identifiers that exist in
  `modules/Tools/tool_maps/functions.json`.

When the persona loader runs it automatically injects the set of known tool identifiers into the
schema so the validator can confirm that `allowed_tools` only contains supported tools. Any
additional persona fields remain opt-in and are still passed through unchanged.

## Validating personas locally

Persona validation happens automatically whenever `modules.Personas.load_persona_definition` reads a
persona file, so most developer entry points (UI loading, tests, etc.) will fail fast if a persona is
invalid. To run a dedicated validation sweep locally, execute the schema test suite:

```bash
pytest tests/test_persona_schema.py
```

The positive test loads every persona in the repository and the negative test confirms that
misconfigured personas raise a descriptive `PersonaValidationError`.

## Updating personas

When you add or modify a persona:

1. Update the JSON definition under `modules/Personas/<Persona Name>/Persona/`.
2. Ensure any referenced tools exist in `modules/Tools/tool_maps/functions.json` (or add them there).
3. Run `pytest tests/test_persona_schema.py` (or the full `pytest` suite) before opening a PR.
4. If the schema needs to change, edit `modules/Personas/schema.json` and keep these docs in sync.
