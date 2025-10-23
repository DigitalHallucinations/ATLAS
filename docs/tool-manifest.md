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
| `side_effects` | string | One of `"none"`, `"write"`, `"network"`, `"read_external_service"`, `"filesystem"`, `"compute"`, `"system"`, or `"database"`, signalling the type of external interaction the tool performs. |
| `default_timeout` | integer | Preferred execution timeout in seconds for the tool. |
| `auth` | object | Authentication requirements, e.g. `{ "required": true, "type": "api_key", "env": "GOOGLE_API_KEY", "docs": "Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback." }`. |
| `allow_parallel` | boolean | Indicates whether the tool is safe to invoke in parallel with itself. |

Additional fields (such as `description` and `parameters`) continue to follow
the OpenAI function-calling schema.

### Side-effect categories

Use the `side_effects` field to describe the strongest external interaction a
tool performs. The allowed values are:

* `none` – Purely computational helpers that do not observe or mutate external
  state.
* `read_external_service` – Makes outbound requests to APIs or services without
  modifying data.
* `network` – Performs generalized network I/O beyond a single API call (for
  example, long-lived sockets or streaming transfers).
* `filesystem` – Reads or writes the local filesystem.
* `write` – Mutates remote state (HTTP `POST`, `PUT`, etc.) or otherwise
  persists data outside the agent process.
* `database` – Interacts with a database or durable data store through query or
  mutation operations.
* `compute` – Triggers heavyweight computational workloads (e.g., job runners or
  ML pipelines).
* `system` – Performs privileged or system-level operations such as process
  control or shell execution.

When in doubt, choose the most restrictive label that applies so downstream
callers can reason about the potential impact of invoking the tool.

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
    "env": "GOOGLE_API_KEY",
    "docs": "Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback."
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
            "auth": {"required": True, "type": "api_key", "env": "GOOGLE_API_KEY", "docs": "Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback."},
            "allow_parallel": True,
        },
    },
    ...
}
```

The callable can still be executed directly, while the immutable `metadata`
mapping provides the manifest hints described above.

### Provider selection rules

When a manifest entry lists multiple providers, the dispatcher evaluates them in
ascending `priority` order and skips any option that fails its health check or
is currently backing off after previous errors. The Google search tool now
declares an official Google Programmable Search provider (`google_cse`) ahead of
the legacy SerpAPI integration. The CSE provider only reports healthy when both
`GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are configured (either in the application
config or environment). If either credential is missing, the router
automatically falls back to the SerpAPI provider, which still honours
`GOOGLE_API_KEY`/`SERPAPI_KEY` for compatibility.

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

The registry snapshot returned by `CapabilityRegistry.summary` (and surfaced by
the discovery APIs) now also includes a normalized `jobs` collection. Each entry
reports the job summary and description, the personas allowed to schedule it,
the required skills and tools, and the derived capability requirements inferred
from those dependencies. Rolling execution metrics are attached under
`health.job`, mirroring the tool health payload, so dashboards can filter on
availability or success rate. When querying by persona, the registry applies
persona filters to both the manifest owner (shared vs persona-specific) and the
job-level persona allowlist to ensure UI clients only see eligible jobs.

For task-specific manifests, lifecycle states, and API guidance see
[Task metadata and lifecycle](tasks/overview.md).
