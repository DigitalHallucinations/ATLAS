# Configuration reference

ATLAS centralises runtime configuration in `ConfigManager`, which merges `.env` values with `atlas_config.yaml` and normalises well-known blocks before the GTK shell or automation services start.【F:ATLAS/config.py†L94-L474】 Use this guide to understand each configuration block, the expected data types, defaults, environment variable overrides, and the runtime components that consume them. For subsystem-specific deep dives, see the [documentation map](../README.md#documentation-map) and the focused articles linked below.【F:README.md†L29-L37】

## Tool execution defaults

### `tool_defaults`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `timeout_seconds` | Number of seconds; defaults to `30` when unset.【F:ATLAS/config.py†L128-L136】 | None (YAML-only). | Tool timeout resolution in `ToolManager` when dispatching tools.【F:ATLAS/ToolManager.py†L1334-L1350】 |
| `max_cost_per_session` | Floating-point currency budget; defaults to `null` to disable budget enforcement.【F:ATLAS/config.py†L128-L136】 | None (YAML-only). | Session budget enforcement in `AgentRouter`.【F:ATLAS/AgentRouter.py†L253-L276】 |

### `conversation`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `max_tool_duration_ms` | Number of milliseconds tools may run within a conversation; defaults to `120000` ms.【F:ATLAS/config.py†L138-L144】 | None (YAML-only). | Conversation runtime budgeting utilities used by the orchestration layer.【F:modules/orchestration/budget_tracker.py†L26-L50】 |

### `tool_logging`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `log_full_payloads` | Boolean; defaults to `false` to avoid storing full request/response bodies.【F:ATLAS/config.py†L146-L153】 | None. | Public tool log projections in `ToolManager` honour this flag when redacting payloads.【F:ATLAS/ToolManager.py†L1706-L1772】 |
| `payload_summary_length` | Integer; defaults to `256` characters for redacted payload summaries.【F:ATLAS/config.py†L146-L153】 | None. | Public tool log projections in `ToolManager`.【F:ATLAS/ToolManager.py†L1706-L1772】 |

### `tool_safety`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `network_allowlist` | Sequence of hostnames; normalised to a lowercase list or `null` to disable outbound HTTP(s).【F:ATLAS/config.py†L345-L354】 | None. | Browser-lite and webpage fetcher tools enforce this allowlist when issuing requests.【F:modules/Tools/Base_Tools/browser_lite.py†L279-L296】【F:modules/Tools/Base_Tools/webpage_fetch.py†L144-L198】 |

## Tool adapters (`tools.*`)

### `tools.javascript_executor`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `executable` | Path to the JavaScript runtime; optional. Picks up `.env` `JAVASCRIPT_EXECUTOR_BIN` when unset in YAML.【F:ATLAS/config.py†L161-L170】【F:ATLAS/config.py†L1548-L1566】 | `JAVASCRIPT_EXECUTOR_BIN`. | JavaScript executor factory used by tool maps.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `args` | Command-line arguments as list; string values are split automatically and can be provided through `JAVASCRIPT_EXECUTOR_ARGS`.【F:ATLAS/config.py†L171-L183】【F:ATLAS/config.py†L2572-L2591】 | `JAVASCRIPT_EXECUTOR_ARGS`. | JavaScript executor factory used by tool maps.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `default_timeout` | Float seconds; defaults to `5.0`.【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor runtime budget. Same consumer as above.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `cpu_time_limit` | Float seconds; defaults to `2.0`.【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `memory_limit_bytes` | Integer bytes; defaults to `268435456` (256 MiB).【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_output_bytes` | Integer bytes; defaults to `65536`.【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_file_bytes` | Integer bytes; defaults to `131072`.【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |
| `max_files` | Integer count; defaults to `32`.【F:ATLAS/config.py†L178-L183】 | None. | JavaScript executor sandboxing.【F:modules/Tools/tool_maps/maps.py†L84-L86】 |

### `tools.kv_store`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `default_adapter` | String adapter name; defaults to `postgres`.【F:ATLAS/config.py†L187-L199】 | None. | Key-value service builder picks the adapter by name.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.url` | PostgreSQL DSN string. Falls back to `.env` `ATLAS_KV_STORE_URL`.【F:ATLAS/config.py†L205-L214】 | `ATLAS_KV_STORE_URL`. | PostgreSQL adapter factory opens the engine for this DSN.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.namespace_quota_bytes` | Integer ≥0; defaults to `1_048_576` bytes, or an integer from `.env` `ATLAS_KV_NAMESPACE_QUOTA_BYTES`.【F:ATLAS/config.py†L215-L238】 | `ATLAS_KV_NAMESPACE_QUOTA_BYTES`. | Enforced in the PostgreSQL adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L327-L360】 |
| `adapters.postgres.global_quota_bytes` | Optional integer; can be set in YAML or via `.env` `ATLAS_KV_GLOBAL_QUOTA_BYTES` (values ≤0 are ignored).【F:ATLAS/config.py†L239-L261】 | `ATLAS_KV_GLOBAL_QUOTA_BYTES`. | Enforced in the PostgreSQL adapter when writing entries.【F:modules/Tools/Base_Tools/kv_store.py†L327-L360】 |
| `adapters.postgres.reuse_conversation_store` | Boolean; defaults to `true` unless `.env` `ATLAS_KV_REUSE_CONVERSATION` provides an override.【F:ATLAS/config.py†L263-L282】 | `ATLAS_KV_REUSE_CONVERSATION`. | Determines whether the KV store reuses the conversation database connection pool when bootstrapping the adapter.【F:modules/Tools/Base_Tools/kv_store.py†L286-L314】 |
| `adapters.postgres.pool.size` | Integer; optionally sourced from `.env` `ATLAS_KV_POOL_SIZE`.【F:ATLAS/config.py†L284-L308】 | `ATLAS_KV_POOL_SIZE`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config.py†L1200-L1230】 |
| `adapters.postgres.pool.max_overflow` | Integer; optionally from `.env` `ATLAS_KV_MAX_OVERFLOW`.【F:ATLAS/config.py†L284-L308】 | `ATLAS_KV_MAX_OVERFLOW`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config.py†L1200-L1230】 |
| `adapters.postgres.pool.timeout` | Float seconds; optionally from `.env` `ATLAS_KV_POOL_TIMEOUT`.【F:ATLAS/config.py†L284-L308】 | `ATLAS_KV_POOL_TIMEOUT`. | Applied to the SQLAlchemy engine created for the KV store.【F:ATLAS/config.py†L1200-L1230】 |

## Data services

### `conversation_database`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `url` | PostgreSQL DSN string; defaults to `.env` `CONVERSATION_DATABASE_URL`, otherwise falls back to the built-in DSN during verification.【F:ATLAS/config.py†L359-L369】【F:ATLAS/config.py†L1410-L1445】 | `CONVERSATION_DATABASE_URL`. | Conversation-store engine verification and session factories.【F:ATLAS/config.py†L552-L619】 |
| `pool.size` | Integer; optional override via `.env` `CONVERSATION_DATABASE_POOL_SIZE`.【F:ATLAS/config.py†L369-L390】 | `CONVERSATION_DATABASE_POOL_SIZE`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config.py†L552-L619】 |
| `pool.max_overflow` | Integer; optional override via `.env` `CONVERSATION_DATABASE_MAX_OVERFLOW`.【F:ATLAS/config.py†L369-L390】 | `CONVERSATION_DATABASE_MAX_OVERFLOW`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config.py†L552-L619】 |
| `pool.timeout` | Integer seconds; optional override via `.env` `CONVERSATION_DATABASE_POOL_TIMEOUT`.【F:ATLAS/config.py†L369-L390】 | `CONVERSATION_DATABASE_POOL_TIMEOUT`. | Conversation-store SQLAlchemy engine creation.【F:ATLAS/config.py†L552-L619】 |
| `retention.days` | Optional integer retention window. `.env` `CONVERSATION_DATABASE_RETENTION_DAYS` overrides YAML when present.【F:ATLAS/config.py†L392-L408】 | `CONVERSATION_DATABASE_RETENTION_DAYS`. | Background workers and APIs honour retention policies; see [conversation retention](conversation_retention.md).【F:docs/conversation_retention.md†L1-L34】 |
| `retention.history_message_limit` | Integer; defaults to `500` messages when unset.【F:ATLAS/config.py†L392-L408】 | None. | Conversation trimming routines; see [conversation retention](conversation_retention.md).【F:docs/conversation_retention.md†L1-L34】 |

Review the [conversation store data model](conversation-store.md) for table-level context and repository helpers that consume these settings.【F:docs/conversation-store.md†L1-L88】

### `messaging`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `backend` | String; defaults to `in_memory` and normalised to lowercase.【F:ATLAS/config.py†L415-L426】 | None. | `ConfigManager.configure_message_bus` instantiates the configured backend.【F:ATLAS/config.py†L1458-L1481】 |
| `redis_url` | Redis connection URI; defaults to `.env` `REDIS_URL` when backend is `redis`, otherwise optional.【F:ATLAS/config.py†L420-L426】 | `REDIS_URL`. | Redis-backed message bus initialisation. See [messaging bus operations](ops/messaging.md).【F:ATLAS/config.py†L1458-L1481】【F:docs/ops/messaging.md†L3-L41】 |
| `stream_prefix` | String; defaults to `atlas_bus` for Redis backends.【F:ATLAS/config.py†L420-L426】 | None. | Namespaces Redis streams created by the bus.【F:ATLAS/config.py†L1458-L1481】 |

### `task_queue`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `jobstore_url` | PostgreSQL DSN; defaults to the derived job store URL when omitted and is normalised/validated on load.【F:ATLAS/config.py†L321-L343】【F:ATLAS/config.py†L1410-L1445】 | `TASK_QUEUE_JOBSTORE_URL`. | APScheduler SQL job store configuration inside the task queue service.【F:modules/Tools/Base_Tools/task_queue.py†L200-L352】 |
| `max_workers` | Integer; optional. Often mirrored from `job_scheduling.max_workers` and can be provided through environment overrides at runtime.【F:ATLAS/config.py†L1241-L1294】【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 | `TASK_QUEUE_MAX_WORKERS`. | Governs the APScheduler thread pool created for background execution.【F:modules/Tools/Base_Tools/task_queue.py†L200-L352】 |
| `timezone` | IANA timezone name or tzinfo; optional. | None. | Determines the scheduler timezone when instantiating the task queue service.【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 |
| `queue_size` | Optional integer queue length carried over from job scheduling defaults when configured.【F:ATLAS/config.py†L1241-L1294】 | None. | Used to pre-size scheduler queues in combined job/task orchestrations.【F:ATLAS/config.py†L1241-L1294】 |
| `retry_policy` | Mapping with `max_attempts`, `backoff_seconds`, `jitter_seconds`, `backoff_multiplier`; defaults to `{3, 30.0, 5.0, 2.0}` when omitted.【F:ATLAS/config.py†L1241-L1294】 | None. | Converted into a `RetryPolicy` for APScheduler jobs.【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 |

### `task_queue`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `jobstore_url` | PostgreSQL DSN; normalised and defaults to the configured job store (conversation DB or `_DEFAULT_CONVERSATION_STORE_DSN`) when not supplied.【F:ATLAS/config.py†L321-L343】【F:ATLAS/config.py†L1410-L1445】 | `TASK_QUEUE_JOBSTORE_URL`. | Task queue service uses this DSN to connect APScheduler’s SQL job store.【F:modules/Tools/Base_Tools/task_queue.py†L200-L352】 |
| `max_workers` | Integer; optional knob mirrored to job scheduling defaults. | `TASK_QUEUE_MAX_WORKERS` (read at runtime). | Task queue thread pool sizing.【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 |
| `timezone` | IANA timezone string or tzinfo instance; optional. | None. | Task queue scheduling timezone.【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 |
| `queue_size` | Integer; optional bounded queue length. | None. | Task queue scheduling defaults.【F:ATLAS/config.py†L1241-L1294】 |
| `retry_policy` | Mapping with `max_attempts`, `backoff_seconds`, `jitter_seconds`, `backoff_multiplier`; defaults to `{3, 30.0, 5.0, 2.0}` when unspecified.【F:ATLAS/config.py†L1241-L1294】 | None. | APScheduler retry guidance surfaced to job scheduling and task queue consumers.【F:ATLAS/config.py†L1241-L1294】【F:modules/Tools/Base_Tools/task_queue.py†L270-L352】 |

### `job_scheduling`
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `enabled` | Boolean; coerced to `false` by default.【F:ATLAS/config.py†L1241-L1294】 | None. | Determines whether background job scheduling is initialised during startup.【F:ATLAS/config.py†L1404-L1453】 |
| `job_store_url` | PostgreSQL DSN; validated and falls back to task queue or conversation DSNs when absent.【F:ATLAS/config.py†L1245-L1292】【F:ATLAS/config.py†L1410-L1445】 | `TASK_QUEUE_JOBSTORE_URL`. | APScheduler-backed job manager setup and setup wizard defaults.【F:ATLAS/config.py†L1404-L1453】【F:ATLAS/setup/controller.py†L168-L182】 |
| `max_workers` | Integer; optional. | None. | Shared between job scheduler and task queue service.【F:ATLAS/config.py†L1241-L1294】【F:ATLAS/setup/controller.py†L168-L182】 |
| `timezone` | Optional timezone string. | None. | Scheduler timezone configuration.【F:ATLAS/config.py†L1241-L1294】 |
| `queue_size` | Optional integer queue length. | None. | Scheduler queue sizing.【F:ATLAS/config.py†L1241-L1294】 |
| `retry_policy.*` | `max_attempts`, `backoff_seconds`, `jitter_seconds`, `backoff_multiplier`; defaults to `{3, 30.0, 5.0, 2.0}` and validated on load.【F:ATLAS/config.py†L1241-L1294】 | None. | Scheduler retry policy shared with task queue consumers.【F:ATLAS/config.py†L1241-L1294】 |

## Provider credentials and defaults

### API key environment variables
`ConfigManager` maps provider names to environment variables so onboarding flows can surface missing credentials.【F:ATLAS/config.py†L1518-L1535】【F:ATLAS/config.py†L428-L462】 The setup controller preloads these keys when seeding wizard state.【F:ATLAS/setup/controller.py†L192-L209】

| Provider | Environment variable | Notes |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Required by `OpenAIGenerator` during client instantiation.【F:modules/Providers/OpenAI/OA_gen_response.py†L24-L50】 |
| Mistral | `MISTRAL_API_KEY` | Required by the Mistral generator client. |
| Google | `GOOGLE_API_KEY` and optional `GOOGLE_APPLICATION_CREDENTIALS` (persisted via helpers).【F:ATLAS/config.py†L1732-L1759】 |
| Anthropic | `ANTHROPIC_API_KEY` | Required by the Anthropic generator.【F:modules/Providers/Anthropic/Anthropic_gen_response.py†L246-L280】 |
| Grok | `GROK_API_KEY` | Used for Grok provider integrations.【F:ATLAS/config.py†L1518-L1535】【F:ATLAS/config.py†L1548-L1566】 |
| Hugging Face | `HUGGINGFACE_API_KEY` | Required for hosted inference APIs and cached downloads.【F:ATLAS/config.py†L1518-L1535】【F:ATLAS/config.py†L3372-L3469】 |
| ElevenLabs | `XI_API_KEY` | Speech synthesis key surfaced in setup flows.【F:ATLAS/config.py†L1518-L1535】【F:ATLAS/setup/controller.py†L211-L220】 |

### `DEFAULT_PROVIDER` and `DEFAULT_MODEL`
The default chat provider and model fall back to `OpenAI` / `gpt-4o` unless overridden via `.env`. Missing API keys trigger startup warnings so operators know which credentials still need to be supplied.【F:ATLAS/config.py†L128-L136】【F:ATLAS/config.py†L454-L462】【F:ATLAS/config.py†L1548-L1566】 The setup wizard reads these values to seed provider state.【F:ATLAS/setup/controller.py†L192-L209】

### OpenAI defaults (`OPENAI_LLM`)
| Key | Type & default | Environment overrides | Consumed by |
| --- | --- | --- | --- |
| `model` | String; defaults to the configured `DEFAULT_MODEL` (defaulting to `gpt-4o`).【F:ATLAS/config.py†L2740-L2760】 | `DEFAULT_MODEL`. | `OpenAIGenerator` keeps its model manager in sync with this value.【F:modules/Providers/OpenAI/OA_gen_response.py†L24-L118】 |
| `temperature` | Float; defaults to `0.0`.【F:ATLAS/config.py†L2740-L2760】 | None. | OpenAI completion sampling parameters.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `top_p` | Float; defaults to `1.0`.【F:ATLAS/config.py†L2740-L2760】 | None. | Same as above.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `frequency_penalty` | Float; defaults to `0.0`.【F:ATLAS/config.py†L2740-L2760】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `presence_penalty` | Float; defaults to `0.0`.【F:ATLAS/config.py†L2740-L2760】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `max_tokens` | Integer; defaults to `4000`.【F:ATLAS/config.py†L2740-L2760】 | None. | OpenAI completion sampling.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `max_output_tokens` | Optional integer; defaults to `null`.【F:ATLAS/config.py†L2740-L2760】 | None. | Governs streaming chunk size in the generator.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `stream` | Boolean; defaults to `true`.【F:ATLAS/config.py†L2740-L2760】 | None. | Enables server-sent events in generator responses.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `function_calling`, `parallel_tool_calls` | Booleans; default to `true`.【F:ATLAS/config.py†L2740-L2760】 | None. | Controls tool invocation in responses.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `tool_choice` | Optional mapping/string; defaults to `null`.【F:ATLAS/config.py†L2740-L2760】 | None. | Controls OpenAI tool selection.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `reasoning_effort` | String; defaults to `'medium'`.【F:ATLAS/config.py†L2740-L2760】 | None. | Passed through to OpenAI reasoning models.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `base_url`, `organization` | Strings populated from `.env` (`OPENAI_BASE_URL`, `OPENAI_ORGANIZATION`).【F:ATLAS/config.py†L2740-L2760】【F:ATLAS/config.py†L2087-L2103】 | `OPENAI_BASE_URL`, `OPENAI_ORGANIZATION`. | Used when constructing the OpenAI SDK client.【F:modules/Providers/OpenAI/OA_gen_response.py†L36-L45】 |
| `json_mode`, `json_schema` | Optional boolean/mapping controls for JSON responses.【F:ATLAS/config.py†L2740-L2760】 | None. | Influences structured response handling in the generator.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `enable_code_interpreter`, `enable_file_search` | Booleans defaulting to `false`.【F:ATLAS/config.py†L2740-L2760】 | None. | Controls experimental OpenAI features surfaced by the provider manager.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |
| `audio_enabled`, `audio_voice`, `audio_format` | Audio synthesis settings with defaults `false`, `'alloy'`, `'wav'`.【F:ATLAS/config.py†L2740-L2760】 | None. | Shared with speech configuration UIs.【F:modules/Providers/OpenAI/OA_gen_response.py†L78-L120】 |

### Mistral defaults (`MISTRAL_LLM`)
The Mistral block accepts extensive sampling, retry, and prompting controls. Defaults include `model='mistral-large-latest'`, `temperature=0.0`, `top_p=1.0`, `max_tokens=null`, `safe_prompt=false`, `stream=true`, retry limits, JSON mode flags, and optional prompt mode validation.【F:ATLAS/config.py†L2774-L3359】 Environment overrides include `.env` `MISTRAL_BASE_URL`, which seeds the base URL before YAML overrides.【F:ATLAS/config.py†L1548-L1566】【F:ATLAS/config.py†L2774-L3359】 The Mistral generator reads this block on every request to configure retries, streaming, and tool behaviour.【F:modules/Providers/Mistral/Mistral_gen_response.py†L96-L155】

### Google defaults (`GOOGLE_LLM`)
Google-specific defaults cover streaming, function calling, response schemas, seeds, and allowed function lists, with validation handled via `GoogleSettingsResolver` and YAML persistence.【F:ATLAS/config.py†L2107-L2202】 These settings drive Gemini request construction and function tool declarations in the Google provider.【F:modules/Providers/Google/GG_gen_response.py†L64-L140】 Additional speech credentials can be persisted through helper methods for `GOOGLE_APPLICATION_CREDENTIALS` as needed.【F:ATLAS/config.py†L1732-L1759】

### Anthropic defaults (`ANTHROPIC_LLM`)
Anthropic defaults include `model='claude-3-opus-20240229'`, streaming and function-calling flags, temperature/top-p controls, retry budgets, metadata, and optional Claude “thinking” fields.【F:ATLAS/config.py†L3700-L3747】 The Anthropic generator hydrates its runtime settings from this block when initialising, ensuring safe coercion and bounds checking.【F:modules/Providers/Anthropic/Anthropic_gen_response.py†L246-L320】

### Hugging Face generation defaults (`HUGGINGFACE.generation_settings`)
Hugging Face generation defaults mirror `_DEFAULT_HUGGINGFACE_GENERATION_SETTINGS`, covering sampling controls such as `temperature`, `top_p`, `top_k`, `max_tokens`, repetition penalties, and booleans for `early_stopping`/`do_sample`. Values are validated before persistence.【F:ATLAS/config.py†L82-L108】【F:ATLAS/config.py†L3372-L3469】 These settings are consumed by Hugging Face provider helpers when issuing hosted inference requests.【F:ATLAS/config.py†L3372-L3469】

## Account management configuration

Local account policies rely on environment variables (for example `ACCOUNT_PASSWORD_MIN_LENGTH`, `ACCOUNT_PASSWORD_REQUIRE_SYMBOL`, and related boolean flags) that `UserAccountService` reads through `ConfigManager`. Overrides are validated before updating the effective password requirements displayed in the UI.【F:modules/user_accounts/user_account_service.py†L457-L515】 Consult the [password policy guide](password-policy.md) for behavioural expectations and operator guidance.【F:docs/password-policy.md†L3-L23】

## Additional notes
- Messaging, conversation retention, KV store, and task queue configuration all surface inside the setup wizard so operators can verify connectivity before completing onboarding.【F:ATLAS/setup/controller.py†L168-L209】
- The Redis-backed message bus and retention policies have dedicated operational runbooks—see [messaging bus operations](ops/messaging.md) and [conversation retention](conversation_retention.md) for deployment guidance.【F:docs/ops/messaging.md†L3-L41】【F:docs/conversation_retention.md†L1-L34】
