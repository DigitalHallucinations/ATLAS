# Tool Manifest Metadata

Tool declarations in ATLAS now include structured metadata so that personas and
shared tools can advertise execution constraints to downstream consumers.

## Discovering tools via the API

The `/tools` discovery endpoint returns the combined set of shared and persona
manifests. Query parameters can scope the result set; for example,
`/tools?persona=Atlas` limits persona-specific tools to the Atlas persona. Tools
marked as shared are now always included in persona-scoped queries so that
callers see the complete toolbox a persona can access. To omit shared tools from
persona-filtered results, add `-shared` to the persona filter
(`persona=Atlas&persona=-shared`). Additional filters allow callers to narrow the
result set by provider name (`provider=openai`), semantic version constraints
(`version=>=1.2`), and observed reliability (`min_success_rate=0.8`). Every tool
response now includes normalized capability tags, declared authentication scopes,
and a rolling health summary derived from recent executions.

## Manifest keys

Each entry inside `modules/Tools/tool_maps/functions.json` and the persona-level
`modules/Personas/<Persona>/Toolbox/functions.json` files can declare the
following fields:

| Field | Type | Description |
| --- | --- | --- |
| `version` | string | Human readable semantic version for the tool. |
| `side_effects` | string | One of `"none"`, `"write"`, `"network"`, `"read_external_service"`, `"filesystem"`, `"compute"`, or `"system"`, signalling the type of external interaction the tool performs. |
| `default_timeout` | integer | Preferred execution timeout in seconds for the tool. |
| `auth` | object | Authentication requirements, e.g. `{ "required": true, "type": "api_key", "env": "GOOGLE_API_KEY" }`. |
| `allow_parallel` | boolean | Indicates whether the tool is safe to invoke in parallel with itself. |

Additional fields (such as `description` and `parameters`) continue to follow
the OpenAI function-calling schema.

## Schema validation

All tool manifests are validated against the JSON Schema stored at
`modules/Tools/tool_maps/schema.json` when they are loaded. Invalid manifests
cause `ToolManifestValidationError` to be raised during startup, so make sure
your persona and shared manifests pass validation before committing changes.

You can validate a manifest locally by running:

```bash
python -m jsonschema -i modules/Personas/<Persona>/Toolbox/functions.json \
    modules/Tools/tool_maps/schema.json
```

The automated test suite also checks for validation failures, so any schema
regressions will be caught in CI.

## Example

```json
{
  "name": "google_search",
  "version": "1.0.0",
  "side_effects": "none",
  "default_timeout": 30,
  "auth": {
    "required": true,
    "type": "api_key",
    "env": "GOOGLE_API_KEY"
  },
  "allow_parallel": true,
  "description": "A Google search result API...",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search content."
      }
    },
    "required": ["query"]
  }
}
```

## Accessing metadata at runtime

`ATLAS.ToolManager.load_function_map_from_current_persona` and
`ATLAS.ToolManager.load_default_function_map` now return entries shaped as:

```python
{
    "google_search": {
        "callable": <callable>,
        "metadata": {
            "version": "1.0.0",
            "side_effects": "none",
            "default_timeout": 30,
            "auth": {"required": True, "type": "api_key", "env": "GOOGLE_API_KEY"},
            "allow_parallel": True,
        },
    },
    ...
}
```

The callable can still be executed directly, while the immutable `metadata`
mapping provides the manifest hints described above.

## Capability registry

Tool and skill metadata is cached by the `CapabilityRegistry`, which ingests the
shared manifests and persona overrides and exposes the merged view to
`ToolManager`, `SkillManager`, and the `/tools` and `/skills` APIs. The registry
automatically detects manifest changes, so registering a new tool only requires
adding it to the appropriate manifest file. Health metrics—success rate, latency
averages, and provider backoff state—are continuously recorded by
`ToolManager` and surfaced through the registry. If you need to eagerly reload
metadata after editing a manifest, call
`CapabilityRegistry.refresh(force=True)` or restart the service.
