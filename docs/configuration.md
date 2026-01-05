---
audience: Operators and platform engineers
status: in_review
last_verified: 2026-06-01
source_of_truth: ATLAS/config/tooling.py; ATLAS/config/persistence.py; modules/Tools/Base_Tools/kv_store.py; modules/Tools/Base_Tools/vector_store.py; modules/Tools/providers/mcp.py; modules/background_tasks/conversation_summary.py
---

# Configuration reference

ATLAS centralises runtime configuration in `ConfigManager`, which merges `.env` values with `atlas_config.yaml` and normalises well-known blocks before the GTK shell or automation services start.【F:ATLAS/config/core.py†L30-L73】【F:ATLAS/config/config_manager.py†L78-L193】 Use this guide to understand each configuration block, the expected data types, defaults, environment variable overrides, and the runtime components that consume them. For subsystem-specific deep dives, see the [documentation map](../README.md#documentation-map) and the focused articles linked below.【F:README.md†L29-L37】

## Tool execution defaults

### `tool_defaults`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `timeout_seconds` | Number of seconds; defaults to `30` when unset.【F:ATLAS/config/tooling.py†L31-L39】 | None (YAML-only). | Tool timeout resolution in `ToolManager` when dispatching tools.【F:ATLAS/ToolManager.py†L1334-L1350】 |
| `max_cost_per_session` | Floating-point currency budget; defaults to `null` to disable budget enforcement.【F:ATLAS/config/tooling.py†L31-L39】 | None (YAML-only). | Session budget enforcement in `AgentRouter`.【F:ATLAS/AgentRouter.py†L253-L276】 |

### `conversation`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `max_tool_duration_ms` | Number of milliseconds tools may run within a conversation; defaults to `120000` ms.【F:ATLAS/config/tooling.py†L41-L48】 | None (YAML-only). | Conversation runtime budgeting utilities used by the orchestration layer.【F:modules/orchestration/budget_tracker.py†L26-L50】 |

### `tool_logging`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `log_full_payloads` | Boolean; defaults to `false` to avoid storing full request/response bodies.【F:ATLAS/config/tooling.py†L50-L58】 | None. | Public tool log projections in `ToolManager` honour this flag when redacting payloads.【F:ATLAS/ToolManager.py†L1706-L1772】 |
| `payload_summary_length` | Integer; defaults to `256` characters for redacted payload summaries.【F:ATLAS/config/tooling.py†L50-L58】 | None. | Public tool log projections in `ToolManager`.【F:ATLAS/ToolManager.py†L1706-L1772】 |

### `tool_safety`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `network_allowlist` | Sequence of hostnames; normalised to a lowercase list or `null` to disable outbound HTTP(s).【F:ATLAS/config/tooling.py†L94-L105】 | None. | Browser-lite and webpage fetcher tools enforce this allowlist when issuing requests.【F:modules/Tools/Base_Tools/browser_lite.py†L279-L296】【F:modules/Tools/Base_Tools/webpage_fetch.py†L144-L198】 |

## Tool adapters (`tools.*`)

### `tools.javascript_executor`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `executable` | Path to the JavaScript runtime; optional. Picks up `.env` `JAVASCRIPT_EXECUTOR_BIN` when unset in YAML.【F:ATLAS/config/tooling.py†L73-L75】【F:ATLAS/config/core.py†L55-L70】 | `JAVASCRIPT_EXECUTOR_BIN`. | JavaScript executor factory used by tool maps.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `args` | Command-line arguments as list; string values are split automatically and can be provided through `JAVASCRIPT_EXECUTOR_ARGS`.【F:ATLAS/config/tooling.py†L77-L82】 | `JAVASCRIPT_EXECUTOR_ARGS`. | JavaScript executor factory used by tool maps.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `default_timeout` | Float seconds; defaults to `5.0`.【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor runtime budget. Same consumer as above.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `cpu_time_limit` | Float seconds; defaults to `2.0`.【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `memory_limit_bytes` | Integer bytes; defaults to `268435456` (256 MiB).【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_output_bytes` | Integer bytes; defaults to `65536`.【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_file_bytes` | Integer bytes; defaults to `131072`.【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_files` | Integer count; defaults to `32`.【F:ATLAS/config/tooling.py†L84-L89】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |

### `tools.mcp`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false` to keep MCP tooling opt-in.【F:ATLAS/config/tooling.py†L150-L183】 | `ATLAS_MCP_ENABLED`.【F:ATLAS/config/core.py†L167-L177】 | Downstream tool routing can use this flag to decide whether to register MCP tool providers. |
| `default_server` / `server` | String server name; defaults to empty when no servers are defined (falls back to the first defined server).【F:ATLAS/config/tooling.py†L161-L245】【F:modules/Tools/providers/mcp.py†L38-L50】 | `ATLAS_MCP_DEFAULT_SERVER`.【F:ATLAS/config/core.py†L167-L177】 | Used by `McpToolProvider` when selecting the target server for a call.【F:modules/Tools/providers/mcp.py†L38-L142】 |
| `timeout_seconds` | Float seconds; defaults to `30.0`.【F:ATLAS/config/tooling.py†L166-L169】 | `ATLAS_MCP_TIMEOUT_SECONDS`.【F:ATLAS/config/core.py†L167-L177】 | Applied as the call timeout inside `McpToolProvider.call`.【F:modules/Tools/providers/mcp.py†L52-L95】 |
| `health_check_interval` | Float seconds; defaults to `300.0`.【F:ATLAS/config/tooling.py†L171-L174】 | `ATLAS_MCP_HEALTH_CHECK_INTERVAL`.【F:ATLAS/config/core.py†L167-L177】 | Controls how frequently MCP provider health checks may run when routed by tool orchestrators.【F:ATLAS/config/tooling.py†L171-L246】 |
| `allow_tools` / `deny_tools` | Optional string or list of tool names to allow/deny; defaults to `null` (no filter).【F:ATLAS/config/tooling.py†L176-L183】 | `ATLAS_MCP_ALLOW_TOOLS`, `ATLAS_MCP_DENY_TOOLS` (comma-separated).【F:ATLAS/config/core.py†L167-L177】 | Exposed to tool routing/policy layers for filtering MCP tools. |
| `servers.*` | Mapping of per-server settings (transport defaults to `stdio`, `command`, `args`, `env`, `cwd`, `url`, `allow_tools`, `deny_tools`, `timeout_seconds`, `health_check_interval`).【F:ATLAS/config/tooling.py†L184-L246】 | Single-server overrides via `ATLAS_MCP_SERVER_TRANSPORT`, `ATLAS_MCP_SERVER_COMMAND`, `ATLAS_MCP_SERVER_ARGS` (split with `shlex`), `ATLAS_MCP_SERVER_URL`, `ATLAS_MCP_SERVER_CWD`, plus the allow/deny env vars above.【F:ATLAS/config/core.py†L167-L177】【F:ATLAS/config/tooling.py†L190-L225】 | Passed to `McpToolProvider` to build transports and connect to MCP servers.【F:modules/Tools/providers/mcp.py†L171-L195】 |
| `tool` | Default MCP tool name to invoke when none is provided at call time; defaults to empty string.【F:ATLAS/config/tooling.py†L243-L246】【F:modules/Tools/providers/mcp.py†L50-L142】 | None. | Used by `McpToolProvider` to determine the tool target when callers omit one.【F:modules/Tools/providers/mcp.py†L50-L142】 |

When enabled, MCP servers can be declared by name and transport. A minimal
stdio-backed server configuration looks like (servers are mandatory; legacy `server_config` fallback has been removed):

```yaml
tools:
  mcp:
    enabled: true
    default_server: scratchpad
    servers:
      scratchpad:
        transport: stdio
        command: mcp-scratchpad
        args: ["--project-root", "/srv/atlas"]
        allow_tools: ["read_file", "write_file"]
        persona: Atlas
```

The router honours `allow_tools`/`deny_tools` filters at both the global MCP
level and per-server level before registering providers, so unsafe or irrelevant
tools are skipped entirely. Side-effect levels, consent, and parallelism are
pulled from the server defaults during manifest translation
(`side_effects="network"`, `requires_consent=true`, `allow_parallel=false`,
`default_timeout=30` seconds unless overridden). If a server entry declares a
`persona`, discovered tools are scoped to that persona; otherwise they remain
shared. Persona allowlists present on manifest data still apply when
`CapabilityRegistry` calculates compatibility for discovery payloads.

Multiple MCP servers can be registered concurrently. Each server becomes its own
provider entry on the translated manifest, and the router performs per-provider
health checks, failure backoff, and priority-based selection. Providers under
backoff are skipped, and the router will fail over to the next healthy provider
before invoking any defined fallback callable.

### `tools.kv_store`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_adapter` | String adapter name; defaults to `postgres`.【F:ATLAS/config/persistence.py†L614-L619】 | None. | Key-value service builder picks the adapter by name.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.url` | PostgreSQL DSN string. Falls back to `.env` `ATLAS_KV_STORE_URL`.【F:ATLAS/config/persistence.py†L626-L635】 | `ATLAS_KV_STORE_URL`. | PostgreSQL adapter factory opens the engine for this DSN.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.namespace_quota_bytes` | Integer ≥0; defaults to `1_048_576` bytes, or an integer from `.env` `ATLAS_KV_NAMESPACE_QUOTA_BYTES`.【F:ATLAS/config/persistence.py†L636-L658】 | `ATLAS_KV_NAMESPACE_QUOTA_BYTES`. | Enforced in the PostgreSQL adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L327-L360】 |
| `adapters.postgres.global_quota_bytes` | Optional integer; can be set in YAML or via `.env` `ATLAS_KV_GLOBAL_QUOTA_BYTES` (values ≤0 are ignored).【F:ATLAS/config/persistence.py†L660-L683】 | `ATLAS_KV_GLOBAL_QUOTA_BYTES`. | Enforced in the PostgreSQL adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L327-L360】 |
| `adapters.postgres.reuse_conversation_store` | Boolean; defaults to `true` unless `.env` `ATLAS_KV_REUSE_CONVERSATION` provides an override.【F:ATLAS/config/persistence.py†L684-L706】 | `ATLAS_KV_REUSE_CONVERSATION`. | Determines whether the KV store reuses the conversation database connection pool when bootstrapping the adapter.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.pool.size` | Integer; optionally sourced from `.env` `ATLAS_KV_POOL_SIZE`.【F:ATLAS/config/persistence.py†L708-L732】 | `ATLAS_KV_POOL_SIZE`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config/persistence.py†L840-L899】 |
| `adapters.postgres.pool.max_overflow` | Integer; optionally from `.env` `ATLAS_KV_MAX_OVERFLOW`.【F:ATLAS/config/persistence.py†L708-L732】 | `ATLAS_KV_MAX_OVERFLOW`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config/persistence.py†L840-L899】 |
| `adapters.postgres.pool.timeout` | Float seconds; optionally from `.env` `ATLAS_KV_POOL_TIMEOUT`.【F:ATLAS/config/persistence.py†L708-L732】 | `ATLAS_KV_POOL_TIMEOUT`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config/persistence.py†L840-L899】 |
| `adapters.sqlite.url` | SQLite DSN string; defaults to `sqlite:///atlas_kv.sqlite`.【F:ATLAS/config/persistence.py†L746-L766】 | None. | SQLite adapter factory opens the engine for this path.【F:modules/Tools/Base_Tools/kv_store.py†L792-L844】 |
| `adapters.sqlite.namespace_quota_bytes` | Integer ≥0; defaults to the PostgreSQL namespace quota fallback when unset.【F:ATLAS/config/persistence.py†L752-L766】 | None. | Enforced by the SQLite adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L792-L844】 |
| `adapters.sqlite.global_quota_bytes` | Optional integer; ignored when blank.【F:ATLAS/config/persistence.py†L768-L779】 | None. | Enforced by the SQLite adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L792-L844】 |
| `adapters.sqlite.reuse_conversation_store` | Boolean; defaults to `false`.【F:ATLAS/config/persistence.py†L781-L788】 | None. | Controls whether the SQLite adapter reuses the conversation engine when available.【F:modules/Tools/Base_Tools/kv_store.py†L700-L739】 |

### `tools.vector_store`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_adapter` | String adapter name; defaults to `in_memory` and normalised to lowercase.【F:ATLAS/config/tooling.py†L78-L105】【F:ATLAS/config/config_manager.py†L210-L246】 | `ATLAS_VECTOR_STORE_ADAPTER`. | `build_vector_store_service` resolves this adapter when constructing the service.【F:modules/Tools/Base_Tools/vector_store.py†L269-L311】 |
| `adapters.*` | Mapping of adapter configuration dictionaries; defaults to an empty mapping with an `in_memory` entry.【F:ATLAS/config/tooling.py†L78-L105】 | None. | Passed to vector-store adapter factories registered in `modules.Tools.providers.vector_store.*`.【F:modules/Tools/Base_Tools/vector_store.py†L269-L311】 |

## Data services

### `conversation_database`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `backend` | String backend identifier; defaults to `postgresql` and includes additional options such as `sqlite`.【F:ATLAS/config/persistence.py†L1017-L1043】【F:ATLAS/config/persistence.py†L1147-L1200】 | `CONVERSATION_DATABASE_BACKEND`. | Persistence helpers and the setup controller select the appropriate SQLAlchemy dialect when verifying the store.【F:ATLAS/config/persistence.py†L1092-L1144】【F:ATLAS/setup/controller.py†L185-L214】 |
| `url` | PostgreSQL DSN string; defaults to `.env` `CONVERSATION_DATABASE_URL`, otherwise falls back to the built-in DSN during verification.【F:ATLAS/config/persistence.py†L1011-L1024】【F:ATLAS/config/persistence.py†L1072-L1090】 | `CONVERSATION_DATABASE_URL`. | Conversation-store engine verification and session factories.【F:ATLAS/config/persistence.py†L306-L346】【F:ATLAS/config/persistence.py†L1108-L1144】 |
| `pool.size` | Integer; optional override via `.env` `CONVERSATION_DATABASE_POOL_SIZE`.【F:ATLAS/config/persistence.py†L1022-L1043】 | `CONVERSATION_DATABASE_POOL_SIZE`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config/persistence.py†L306-L346】 |
| `pool.max_overflow` | Integer; optional override via `.env` `CONVERSATION_DATABASE_MAX_OVERFLOW`.【F:ATLAS/config/persistence.py†L1022-L1043】 | `CONVERSATION_DATABASE_MAX_OVERFLOW`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config/persistence.py†L306-L346】 |
| `pool.timeout` | Integer seconds; optional override via `.env` `CONVERSATION_DATABASE_POOL_TIMEOUT`.【F:ATLAS/config/persistence.py†L1022-L1043】 | `CONVERSATION_DATABASE_POOL_TIMEOUT`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config/persistence.py†L306-L346】 |
| `retention.days` | Optional integer retention window. `.env` `CONVERSATION_DATABASE_RETENTION_DAYS` overrides YAML when present.【F:ATLAS/config/persistence.py†L1045-L1062】 | `CONVERSATION_DATABASE_RETENTION_DAYS`. | Background workers and APIs honour retention policies; see [conversation retention](conversation_retention.md).【F:docs/conversation_retention.md†L1-L34】 |
| `retention.history_message_limit` | Integer; defaults to `500` messages when unset.【F:ATLAS/config/persistence.py†L1045-L1062】 | None. | Conversation trimming routines; see [conversation retention](conversation_retention.md).【F:docs/conversation_retention.md†L1-L34】 |

Review the [conversation store data model](conversation-store.md) for table-level context and repository helpers that consume these settings.【F:docs/conversation-store.md†L1-L88】

### `messaging`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `backend` | String; defaults to `ncb` (Neural Cognitive Bus with in-process async queues).【F:ATLAS/config/messaging.py†L23-L34】 | None. | `configure_agent_bus()` instantiates the AgentBus/NCB.【F:ATLAS/messaging/agent_bus.py†L1-L100】 |
| `redis_url` | Redis connection URI for optional bridging; defaults to `.env` `REDIS_URL`.【F:ATLAS/config/messaging.py†L27-L34】 | `REDIS_URL`. | Redis bridging for cross-process messaging. See [messaging bus operations](ops/messaging.md).【F:ATLAS/messaging/NCB.py†L1-L200】 |
| `kafka.enabled` | Boolean; defaults to `false`. Enables Kafka producer bridging.【F:ATLAS/config/messaging.py†L27-L34】 | None. | NCB Kafka bridging for external consumers.【F:ATLAS/messaging/NCB.py†L1-L200】 |
| `kafka.bootstrap_servers` | Kafka cluster connection string.【F:ATLAS/config/messaging.py†L27-L34】 | None. | Kafka producer configuration.【F:ATLAS/messaging/NCB.py†L1-L200】 |

### `conversation_summary`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false` so episodic summaries are opt-in.【F:ATLAS/config/conversation_summary.py†L9-L46】 | None. | Enables the background worker when true.【F:modules/background_tasks/conversation_summary.py†L61-L93】 |
| `cadence_seconds` | Positive float; defaults to `300`. Controls the minimum gap between summaries per conversation.【F:ATLAS/config/conversation_summary.py†L38-L46】【F:modules/background_tasks/conversation_summary.py†L238-L266】 | None. | Evaluated by `ConversationSummaryWorker` when deciding whether to flush a batch.【F:modules/background_tasks/conversation_summary.py†L238-L266】 |
| `window_seconds` | Positive float; defaults to `300`. Batches are flushed when the window elapses even if the batch size is not met.【F:ATLAS/config/conversation_summary.py†L38-L46】【F:modules/background_tasks/conversation_summary.py†L238-L266】 | None. | Same as above. |
| `batch_size` | Integer ≥1; defaults to `10`. Determines how many messages trigger an immediate summary.【F:ATLAS/config/conversation_summary.py†L38-L46】【F:modules/background_tasks/conversation_summary.py†L238-L266】 | None. | Same as above. |
| `tool` / `persona` | Optional strings naming the summarisation tool or persona. Defaults to `context_tracker`; persona is unset by default.【F:ATLAS/config/conversation_summary.py†L38-L46】 | None. | Currently the worker invokes the `context_tracker` tool but records the configured persona for analytics metadata.【F:modules/background_tasks/conversation_summary.py†L287-L314】 |
| `retention.default_days` | Optional integer TTL applied to episodic summaries when no tenant override exists.【F:ATLAS/config/conversation_summary.py†L48-L66】【F:modules/background_tasks/conversation_summary.py†L292-L314】 | None. | Converted to `expires_at` for `EpisodicMemoryTool.store`.【F:modules/background_tasks/conversation_summary.py†L292-L314】 |
| `followups` | Mapping with `defaults` and `personas` template lists; defaults to empty collections.【F:ATLAS/config/conversation_summary.py†L15-L177】 | None. | Templates drive actionable detection and emit `conversation.followups` events for orchestration.【F:modules/background_tasks/conversation_summary.py†L225-L356】【F:modules/orchestration/followups.py†L12-L185】 |
| `tenants.*` | Mapping of tenant-specific overrides for cadence, window, batch size, persona/tool, and retention days.【F:ATLAS/config/conversation_summary.py†L68-L96】 | None. | Overrides resolved per tenant during batch evaluation and TTL calculation.【F:modules/background_tasks/conversation_summary.py†L238-L314】 |

### `task_queue`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `jobstore_url` | PostgreSQL or MongoDB DSN validated on load; defaults to the derived job store URL resolved from `job_scheduling.job_store_url`, `task_queue.jobstore_url`, `TASK_QUEUE_JOBSTORE_URL`, `conversation_database.url`, or `_DEFAULT_CONVERSATION_STORE_DSN_BY_BACKEND` when unset.【F:ATLAS/config/config_manager.py†L202-L229】【F:ATLAS/config/persistence.py†L461-L532】【F:ATLAS/config/persistence.py†L633-L695】 | `TASK_QUEUE_JOBSTORE_URL`. | APScheduler job store connection used by the task queue service and normalised by the config helpers before the service enforces PostgreSQL/SQLite compatibility.【F:ATLAS/config/persistence.py†L633-L695】【F:modules/Tools/Base_Tools/task_queue.py†L397-L435】 |
| `max_workers` | Integer; defaults to `4` when omitted, and mirrored from `job_scheduling.max_workers` into the `task_queue` block when set.【F:ATLAS/config/persistence.py†L480-L490】【F:modules/Tools/Base_Tools/task_queue.py†L315-L395】 | `TASK_QUEUE_MAX_WORKERS` (checked at runtime). | Governs the APScheduler thread pool created for background execution.【F:modules/Tools/Base_Tools/task_queue.py†L315-L395】 |
| `timezone` | IANA timezone string or tzinfo; defaults to UTC when unspecified or invalid.【F:modules/Tools/Base_Tools/task_queue.py†L344-L360】 | None. | Determines the scheduler timezone when instantiating the task queue service.【F:modules/Tools/Base_Tools/task_queue.py†L344-L395】 |
| `queue_size` | Optional integer queue length carried over from job scheduling defaults when configured.【F:ATLAS/config/persistence.py†L480-L490】 | None. | Shared queue sizing hint for job scheduling and task queue helpers that read the merged block.【F:ATLAS/config/persistence.py†L480-L532】 |
| `retry_policy` | Mapping with `max_attempts`, `backoff_seconds`, `jitter_seconds`, `backoff_multiplier`; defaults to `{3, 30.0, 5.0, 2.0}` when omitted.【F:ATLAS/config/persistence.py†L515-L532】【F:modules/Tools/Base_Tools/task_queue.py†L361-L395】 | None. | Converted into a `RetryPolicy` for APScheduler jobs.【F:modules/Tools/Base_Tools/task_queue.py†L361-L395】 |

`ConfigManager` normalises the `task_queue` block during startup, automatically filling `jobstore_url` from the job scheduling settings, environment override, conversation database URL, or the built-in default DSN map when a value is missing.【F:ATLAS/config/config_manager.py†L202-L229】【F:ATLAS/config/persistence.py†L461-L532】【F:ATLAS/config/persistence.py†L633-L695】 At runtime the task queue service re-applies `TASK_QUEUE_JOBSTORE_URL`/`TASK_QUEUE_MAX_WORKERS` overrides and validates the effective job store URL before constructing the APScheduler-backed service in `modules/Tools/Base_Tools/task_queue.py`.【F:modules/Tools/Base_Tools/task_queue.py†L315-L435】

### `job_scheduling`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; coerced to `false` by default.【F:ATLAS/config/persistence.py†L348-L389】 | None. | Determines whether background job scheduling is initialised during startup.【F:ATLAS/config/persistence.py†L277-L305】 |
| `job_store_url` | PostgreSQL DSN; validated and falls back to task queue or conversation DSNs when absent.【F:ATLAS/config/persistence.py†L348-L387】【F:ATLAS/config/persistence.py†L520-L563】 | `TASK_QUEUE_JOBSTORE_URL`. | APScheduler-backed job manager setup and setup wizard defaults.【F:ATLAS/config/persistence.py†L277-L305】【F:ATLAS/setup/controller.py†L168-L182】 |
| `max_workers` | Integer; optional. | None. | Shared between job scheduler and task queue service.【F:ATLAS/config/persistence.py†L367-L417】【F:ATLAS/setup/controller.py†L168-L182】 |
| `timezone` | Optional timezone string. | None. | Scheduler timezone configuration.【F:ATLAS/config/persistence.py†L367-L417】 |
| `queue_size` | Optional integer queue length. | None. | Scheduler queue sizing.【F:ATLAS/config/persistence.py†L367-L417】 |
| `retry_policy.*` | `max_attempts`, `backoff_seconds`, `jitter_seconds`, `backoff_multiplier`; defaults to `{3, 30.0, 5.0, 2.0}` and validated on load.【F:ATLAS/config/persistence.py†L402-L417】 | None. | Scheduler retry policy shared with task queue consumers.【F:ATLAS/config/persistence.py†L402-L417】 |

## Provider credentials and defaults

### API key environment variables

`ConfigManager` maps provider names to environment variables so onboarding flows can surface missing credentials.【F:ATLAS/config/providers.py†L25-L37】【F:ATLAS/config/providers.py†L80-L91】 The setup controller preloads these keys when seeding wizard state.【F:ATLAS/setup/controller.py†L192-L209】

| Provider | Environment variable | Notes |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Required by `OpenAIGenerator` during client instantiation.【F:modules/Providers/OpenAI/OA_gen_response.py†L24-L50】 |
| Mistral | `MISTRAL_API_KEY` | Required by the Mistral generator client. |
| Google | `GOOGLE_API_KEY` and optional `GOOGLE_APPLICATION_CREDENTIALS` | Persisted via helpers.【F:ATLAS/config/providers.py†L698-L765】 |
| Anthropic | `ANTHROPIC_API_KEY` | Required by the Anthropic generator.【F:modules/Providers/Anthropic/Anthropic_gen_response.py†L246-L280】 |
| Grok | `GROK_API_KEY` | Used for Grok provider integrations.【F:ATLAS/config/providers.py†L25-L37】【F:ATLAS/config/providers.py†L2731-L2738】 |
| Hugging Face | `HUGGINGFACE_API_KEY` | Required for hosted inference APIs and cached downloads.【F:ATLAS/config/providers.py†L25-L37】【F:ATLAS/config/providers.py†L2075-L2140】 |
| ElevenLabs | `XI_API_KEY` | Speech synthesis key surfaced in setup flows.【F:ATLAS/config/providers.py†L25-L37】【F:ATLAS/setup/controller.py†L211-L220】 |

#### Media/Image generation providers

| Provider | Environment variable | Notes |
| --- | --- | --- |
| Stability AI | `STABILITY_API_KEY` | Stable Diffusion, SDXL, SD3.5 models.【F:modules/Providers/Media/Stability/provider.py】 |
| FalAI | `FAL_KEY` | Flux and other fast inference models.【F:modules/Providers/Media/FalAI/provider.py】 |
| Black Forest Labs | `BFL_API_KEY` | High-quality Flux 1.1 Pro/Ultra models.【F:modules/Providers/Media/BlackForestLabs/provider.py】 |
| XAI Aurora | `XAI_API_KEY` | Grok-based image generation.【F:modules/Providers/Media/XAI/provider.py】 |
| Google Imagen | `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT` | Vertex AI Imagen models; requires project ID.【F:modules/Providers/Media/Google/provider.py】 |
| Replicate | `REPLICATE_API_TOKEN` | Open model aggregator (FLUX, SDXL, Kandinsky, many more).【F:modules/Providers/Media/Replicate/provider.py】 |
| Ideogram | `IDEOGRAM_API_KEY` | Text-in-image specialist with accurate typography.【F:modules/Providers/Media/Ideogram/provider.py】 |
| Runway | `RUNWAY_API_KEY` | Gen-3 Alpha creative generation tools.【F:modules/Providers/Media/Runway/provider.py】 |
| HuggingFace | `HUGGINGFACE_API_KEY` | Inference API for diffusion models.【F:modules/Providers/Media/HuggingFace/provider.py】 |

See the [Image Generation Tools](tools/image_generation.md) guide for detailed usage and configuration.

### `DEFAULT_PROVIDER` and `DEFAULT_MODEL`

The default chat provider and model fall back to `OpenAI` / `gpt-4o` unless overridden via `.env`. Missing API keys trigger startup warnings so operators know which credentials still need to be supplied.【F:ATLAS/config/core.py†L55-L70】【F:ATLAS/config/providers.py†L80-L103】【F:ATLAS/config/providers.py†L1967-L2021】 The setup wizard reads these values to seed provider state.【F:ATLAS/setup/controller.py†L192-L209】

### OpenAI defaults (`OPENAI_LLM`)

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `model` | String; defaults to the configured `DEFAULT_MODEL` (defaulting to `gpt-4o`).【F:ATLAS/config/providers.py†L316-L333】 | `DEFAULT_MODEL`. | `OpenAIGenerator` keeps its model manager in sync with this value.【F:modules/Providers/OpenAI/OA_gen_response.py†L24-L118】 |
| `temperature` | Float; defaults to `0.0`.【F:ATLAS/config/providers.py†L320-L338】 | None. | OpenAI completion sampling parameters.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `top_p` | Float; defaults to `1.0`.【F:ATLAS/config/providers.py†L320-L338】 | None. | Same as above.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `frequency_penalty` | Float; defaults to `0.0`.【F:ATLAS/config/providers.py†L320-L338】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `presence_penalty` | Float; defaults to `0.0`.【F:ATLAS/config/providers.py†L320-L338】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `max_tokens` | Integer; defaults to `4000`.【F:ATLAS/config/providers.py†L320-L338】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `max_output_tokens` | Optional integer; defaults to `null`.【F:ATLAS/config/providers.py†L320-L338】 | None. | Governs streaming chunk size in the generator.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `stream` | Boolean; defaults to `true`.【F:ATLAS/config/providers.py†L320-L338】 | None. | Enables server-sent events in generator responses.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `function_calling`, `parallel_tool_calls` | Booleans; default to `true`.【F:ATLAS/config/providers.py†L320-L338】 | None. | Controls tool invocation in responses.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `tool_choice` | Optional mapping/string; defaults to `null`.【F:ATLAS/config/providers.py†L320-L338】 | None. | Controls OpenAI tool selection.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `reasoning_effort` | String; defaults to `'medium'`.【F:ATLAS/config/providers.py†L330-L338】 | None. | Passed through to OpenAI reasoning models.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `base_url`, `organization` | Strings populated from `.env` (`OPENAI_BASE_URL`, `OPENAI_ORGANIZATION`).【F:ATLAS/config/providers.py†L332-L334】【F:ATLAS/config/core.py†L55-L70】 | `OPENAI_BASE_URL`, `OPENAI_ORGANIZATION`. | Used when constructing the OpenAI SDK client.【F:modules/Providers/OpenAI/OA_gen_response.py†L36-L45】 |
| `json_mode`, `json_schema` | Optional boolean/mapping controls for JSON responses.【F:ATLAS/config/providers.py†L334-L366】 | None. | Influences structured response handling in the generator.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `enable_code_interpreter`, `enable_file_search` | Booleans defaulting to `false`.【F:ATLAS/config/providers.py†L334-L338】 | None. | Controls experimental OpenAI features surfaced by the provider manager.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `audio_enabled`, `audio_voice`, `audio_format` | Audio synthesis settings with defaults `false`, `'alloy'`, `'wav'`.【F:ATLAS/config/providers.py†L336-L340】 | None. | Shared with speech configuration UIs.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |

### Mistral defaults (`MISTRAL_LLM`)

The Mistral block accepts extensive sampling, retry, and prompting controls. Defaults include `model='mistral-large-latest'`, `temperature=0.0`, `top_p=1.0`, `max_tokens=null`, `safe_prompt=false`, `stream=true`, retry limits, JSON mode flags, and optional prompt mode validation.【F:ATLAS/config/providers.py†L1243-L1675】 Environment overrides include `.env` `MISTRAL_BASE_URL`, which seeds the base URL before YAML overrides.【F:ATLAS/config/core.py†L55-L70】【F:ATLAS/config/providers.py†L1243-L1675】 The Mistral generator reads this block on every request to configure retries, streaming, and tool behaviour.【F:modules/Providers/Mistral/Mistral_gen_response.py†L96-L155】

### Google defaults (`GOOGLE_LLM`)

Google-specific defaults cover streaming, function calling, response schemas, seeds, and allowed function lists, with validation handled via `GoogleSettingsResolver` and YAML persistence.【F:ATLAS/config/providers.py†L770-L1179】 These settings drive Gemini request construction and function tool declarations in the Google provider.【F:modules/Providers/Google/GG_gen_response.py†L64-L140】 Additional speech credentials can be persisted through helper methods for `GOOGLE_APPLICATION_CREDENTIALS` as needed.【F:ATLAS/config/providers.py†L698-L765】

### Anthropic defaults (`ANTHROPIC_LLM`)

Anthropic defaults include `model='claude-3-opus-20240229'`, streaming and function-calling flags, temperature/top-p controls, retry budgets, metadata, and optional Claude “thinking” fields.【F:ATLAS/config/providers.py†L2199-L2300】 The Anthropic generator hydrates its runtime settings from this block when initialising, ensuring safe coercion and bounds checking.【F:modules/Providers/Anthropic/Anthropic_gen_response.py†L246-L320】

### Hugging Face generation defaults (`HUGGINGFACE.generation_settings`)

Hugging Face generation defaults mirror `_DEFAULT_HUGGINGFACE_GENERATION_SETTINGS`, covering sampling controls such as `temperature`, `top_p`, `top_k`, `max_tokens`, repetition penalties, and booleans for `early_stopping`/`do_sample`. Values are validated before persistence.【F:ATLAS/config/providers.py†L1774-L1785】【F:ATLAS/config/providers.py†L2075-L2140】 These settings are consumed by Hugging Face provider helpers when issuing hosted inference requests.【F:ATLAS/config/providers.py†L2075-L2140】

## Account management configuration

Local account policies rely on environment variables (for example `ACCOUNT_PASSWORD_MIN_LENGTH`, `ACCOUNT_PASSWORD_REQUIRE_SYMBOL`, and related boolean flags) that `UserAccountService` reads through `ConfigManager`. Overrides are validated before updating the effective password requirements displayed in the UI.【F:modules/user_accounts/user_account_service.py†L457-L515】 Consult the [password policy guide](password-policy.md) for behavioural expectations and operator guidance.【F:docs/password-policy.md†L3-L23】

## RAG (Retrieval-Augmented Generation)

### `rag`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `true` when a knowledge store is configured. | `ATLAS_RAG_ENABLED`. | RAGService initialization and chat integration. |
| `default_store` | String; defaults to `postgres`. Selects the knowledge store backend. | `ATLAS_RAG_STORE`. | `build_knowledge_store()` factory. |

### `rag.embedding`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_provider` | String; defaults to `openai`. Embedding provider name. | `ATLAS_EMBEDDING_PROVIDER`. | EmbedManager provider selection. |
| `default_model` | String; defaults to `text-embedding-3-small`. Model identifier. | `ATLAS_EMBEDDING_MODEL`. | EmbedManager model configuration. |
| `batch_size` | Integer; defaults to `32`. Batch size for embedding operations. | None. | Batch embedding calls in DocumentIngester. |
| `cache_enabled` | Boolean; defaults to `true`. Enables embedding result caching. | None. | EmbedManager caching layer. |
| `cache_ttl_seconds` | Integer; defaults to `3600`. Cache entry time-to-live. | None. | EmbedManager cache eviction. |

### `rag.chunking`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_size` | Integer; defaults to `512`. Default chunk size in tokens. | None. | DocumentIngester chunking strategy. |
| `default_overlap` | Integer; defaults to `50`. Token overlap between chunks. | None. | DocumentIngester chunking strategy. |
| `max_size` | Integer; defaults to `2000`. Maximum allowed chunk size. | None. | Per-KB configuration validation. |
| `min_size` | Integer; defaults to `100`. Minimum allowed chunk size. | None. | Per-KB configuration validation. |

### `rag.search`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_top_k` | Integer; defaults to `5`. Default number of results. | None. | RAGService search defaults. |
| `max_top_k` | Integer; defaults to `20`. Maximum allowed results. | None. | Query validation in RAGService. |
| `min_score_threshold` | Float; defaults to `0.0`. Minimum similarity score filter. | None. | Search result filtering. |
| `max_context_tokens` | Integer; defaults to `4000`. Maximum tokens in RAG context. | None. | Context formatting for LLM prompts. |

### `rag.hybrid`

Hybrid search combines semantic vector search with lexical BM25 matching using Reciprocal Rank Fusion (RRF).

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false`. Enable hybrid search. | None. | `HybridRetriever` initialization. |
| `weight` | Float; defaults to `0.5`. Balance between vector (1.0) and lexical (0.0). | None. | RRF score combination. |
| `bm25_k1` | Float; defaults to `1.5`. BM25 term frequency saturation. | None. | BM25 index tuning. |
| `bm25_b` | Float; defaults to `0.75`. BM25 document length normalization. | None. | BM25 index tuning. |
| `rrf_k` | Integer; defaults to `60`. RRF ranking constant. | None. | Reciprocal Rank Fusion. |

### `rag.routing`

Query routing classifies incoming queries to determine optimal retrieval strategy.

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false`. Enable query routing. | None. | `QueryRouter` initialization. |
| `model` | String; defaults to `facebook/bart-large-mnli`. Zero-shot classifier model. | None. | HuggingFace pipeline. |
| `default_strategy` | String; defaults to `hybrid`. Fallback strategy. | None. | Route resolution. |
| `confidence_threshold` | Float; defaults to `0.6`. Minimum confidence for routing decisions. | None. | Strategy selection. |

### `rag.compression`

Context compression reduces token usage while preserving relevant information.

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false`. Enable context compression. | None. | `ContextCompressor` initialization. |
| `strategy` | String; defaults to `extractive`. Options: `none`, `extractive`, `llmlingua`, `hybrid`. | None. | Compression pipeline selection. |
| `target_ratio` | Float; defaults to `0.5`. Target compression ratio (0.0-1.0). | None. | Compression intensity. |
| `min_context_length` | Integer; defaults to `1000`. Only compress contexts longer than this. | None. | Compression trigger threshold. |
| `llmlingua_model` | String; defaults to `gpt2`. Model for perplexity scoring. | None. | LLMLingua compressor. |

### `rag.caching`

Caching improves performance by storing embeddings and query results.

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `embedding_cache_enabled` | Boolean; defaults to `true`. Enable embedding caching. | None. | `EmbeddingCache` initialization. |
| `embedding_cache_max_size` | Integer; defaults to `10000`. Maximum cached embeddings. | None. | LRU cache sizing. |
| `embedding_cache_ttl_seconds` | Integer; defaults to `3600`. Cache entry TTL. | None. | Cache eviction. |
| `query_cache_enabled` | Boolean; defaults to `true`. Enable query result caching. | None. | `QueryCache` initialization. |
| `query_cache_max_size` | Integer; defaults to `1000`. Maximum cached queries. | None. | LRU cache sizing. |
| `query_cache_ttl_seconds` | Integer; defaults to `300`. Query cache TTL. | None. | Cache eviction. |
| `query_cache_semantic_matching` | Boolean; defaults to `true`. Match semantically similar queries. | None. | Cache key generation. |
| `query_cache_similarity_threshold` | Float; defaults to `0.95`. Similarity threshold for cache hits. | None. | Semantic cache matching. |

### `rag.observability`

Observability features for monitoring RAG pipeline performance.

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `tracing_enabled` | Boolean; defaults to `false`. Enable OpenTelemetry tracing. | `ATLAS_RAG_TRACING`. | `RAGTracer` initialization. |
| `tracing_endpoint` | String; optional. OTLP endpoint for traces. | `OTEL_EXPORTER_OTLP_ENDPOINT`. | OpenTelemetry exporter. |
| `metrics_enabled` | Boolean; defaults to `true`. Enable Prometheus metrics. | None. | `RAGMetrics` initialization. |
| `metrics_port` | Integer; defaults to `9090`. Prometheus metrics port. | None. | Metrics HTTP server. |

### `rag.evaluation`

Configuration for RAG quality evaluation harness.

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; defaults to `false`. Enable evaluation metrics collection. | None. | `RAGEvaluator` initialization. |
| `metrics` | List of strings; defaults to all metrics. Which metrics to compute. | None. | Evaluator metric selection. |
| `output_dir` | String; defaults to `data/rag_eval`. Directory for evaluation results. | None. | Result persistence. |

### `rag.ingestion`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `max_file_size_mb` | Integer; defaults to `50`. Maximum file size for upload. | None. | Document upload validation. |
| `allowed_extensions` | List of strings; defaults to common document types. | None. | File type validation in upload dialog. |
| `url_timeout_seconds` | Integer; defaults to `30`. Timeout for URL ingestion. | None. | URL fetcher in DocumentIngester. |
| `duplicate_detection` | Boolean; defaults to `true`. Enable duplicate checking. | None. | Pre-ingestion duplicate check. |

### `rag.postgres`

| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `schema` | String; defaults to `atlas_rag`. PostgreSQL schema name. | `ATLAS_RAG_SCHEMA`. | PostgresKnowledgeStore table creation. |
| `vector_index_type` | String; defaults to `ivfflat`. Options: `ivfflat`, `hnsw`. | None. | pgvector index creation. |
| `index_lists` | Integer; defaults to `100`. IVFFlat lists parameter. | None. | Vector index tuning. |
| `hnsw_m` | Integer; defaults to `16`. HNSW M parameter. | None. | HNSW index tuning. |
| `hnsw_ef_construction` | Integer; defaults to `64`. HNSW build parameter. | None. | HNSW index tuning. |

Example RAG configuration:

```yaml
rag:
  enabled: true
  default_store: postgres
  
  embedding:
    default_provider: openai
    default_model: text-embedding-3-small
    batch_size: 32
    cache_enabled: true
    cache_ttl_seconds: 3600
    
  chunking:
    default_size: 512
    default_overlap: 50
    max_size: 2000
    min_size: 100
    
  search:
    default_top_k: 5
    max_top_k: 20
    min_score_threshold: 0.0
    max_context_tokens: 4000
    
  ingestion:
    max_file_size_mb: 50
    allowed_extensions:
      - .txt
      - .md
      - .pdf
      - .docx
      - .html
    url_timeout_seconds: 30
    duplicate_detection: true
    
  postgres:
    schema: atlas_rag
    vector_index_type: hnsw
    hnsw_m: 16
    hnsw_ef_construction: 64
```

For developer integration details, see the [RAG Integration Guide](developer/rag-integration.md). For end-user documentation, see the [RAG User Guide](user/rag-guide.md).

## Additional notes

- Messaging, conversation backend selection, vector store defaults, conversation retention, KV store, and task queue configuration all surface inside the setup wizard so operators can verify connectivity before completing onboarding.【F:ATLAS/setup/controller.py†L168-L246】【F:ATLAS/setup/cli.py†L520-L648】
- The Redis-backed message bus and retention policies have dedicated operational runbooks—see [messaging bus operations](ops/messaging.md) and [conversation retention](conversation_retention.md) for deployment guidance.【F:docs/ops/messaging.md†L3-L41】【F:docs/conversation_retention.md†L1-L34】
