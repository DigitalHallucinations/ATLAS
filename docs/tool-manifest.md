# Tool Manifest Metadata

Tool declarations in ATLAS now include structured metadata so that personas and
shared tools can advertise execution constraints to downstream consumers.

## Manifest keys

Each entry inside `modules/Tools/tool_maps/functions.json` and the persona-level
`modules/Personas/<Persona>/Toolbox/functions.json` files can declare the
following fields:

| Field | Type | Description |
| --- | --- | --- |
| `version` | string | Human readable semantic version for the tool. |
| `side_effects` | string | Either `"none"` or `"write"`, signalling whether a tool mutates external state. |
| `default_timeout` | integer | Preferred execution timeout in seconds for the tool. |
| `auth` | object | Authentication requirements, e.g. `{ "required": true, "type": "api_key", "env": "GOOGLE_API_KEY" }`. |
| `allow_parallel` | boolean | Indicates whether the tool is safe to invoke in parallel with itself. |

Additional fields (such as `description` and `parameters`) continue to follow
the OpenAI function-calling schema.

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
