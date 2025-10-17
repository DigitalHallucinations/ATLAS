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

## Exporting and importing personas

Persona definitions can be exchanged as signed bundles so that the `allowed_tools`
list remains trustworthy. Two workflows are available:

### Command line tools

The `scripts/persona_tools.py` helper exposes `export` and `import` commands. The
commands require a signing key (either directly via `--signing-key` or from a
file with `--signing-key-file`). The examples below assume the repository root as
the working directory; pass `--app-root` when working in a temporary test tree.

```bash
# Export the "Atlas" persona to bundle.json using the key stored in signing.key
python scripts/persona_tools.py export Atlas bundle.json \
    --signing-key-file signing.key

# Import the bundle, persisting the persona definition and recording an audit rationale
python scripts/persona_tools.py import bundle.json \
    --signing-key-file signing.key \
    --rationale "Imported from staging"
```

The importer validates the payload against the persona schema and compares the
referenced tools against the current catalog. Any missing tools are removed and
reported as warnings.

### GTK UI

The persona manager window now includes buttons to export the persona currently
being edited and to import new personas. Both actions prompt for a signing key
and use the same validation logic as the CLI. The import option is available on
the persona list window and opens a file chooser dialog to select the bundle
before verification and persistence.
4. If the schema needs to change, edit `modules/Personas/schema.json` and keep these docs in sync.
