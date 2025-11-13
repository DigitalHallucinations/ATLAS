# AtlasServer API Reference

This reference summarizes the HTTP surface exposed by `AtlasServer` and its helper
route classes. Each endpoint description calls out the HTTP verb, resource path,
required payload fields or query parameters, and expected response structure.
Internally every handler translates requests into `ConversationRoutes`,
`TaskRoutes`, `JobRoutes`, or persona/blackboard helpers inside
`modules/Server/routes.py`.

## Running the standalone HTTP gateway

`server/http_gateway.py` packages these routes into a FastAPI application. To run
the gateway alongside other UI stacks:

1. Install the HTTP dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the FastAPI service with Uvicorn:

   ```bash
   uvicorn server.http_gateway:app --host 0.0.0.0 --port 8080
   ```

   Startup creates a shared `ATLAS` instance, calls `await atlas.initialize()`, and
   wires `AtlasServer` against the configured message bus, task service, and job
   orchestration components. Shutdown hooks dispose of the message bus, task queue
   executor, and provider resources via `atlas.close()`.

3. Forward `RequestContext` information through headers when invoking the API.
   The gateway inspects the following keys and falls back to the configured
   tenant identifier when they are absent:

   | Header | Purpose |
   | --- | --- |
   | `X-Atlas-Tenant` | Required tenant identifier used for authorization checks. |
   | `X-Atlas-User` | Optional user identifier propagated to audit logs. |
   | `X-Atlas-Session` | Optional session correlation identifier. |
   | `X-Atlas-Roles` | Comma-delimited roles (for example `admin,reviewer`). |
   | `X-Atlas-Metadata` | Optional JSON object that becomes `RequestContext.metadata`. |

Streaming endpoints emit Server-Sent Events (`text/event-stream`) with JSON
payloads so dashboards can subscribe to live job, task, and conversation updates.

## Authentication and context

Requests are evaluated against a `RequestContext` that carries the caller's
`tenant_id`, optional `user_id`, `session_id`, and role list. Most routes call
`_require_context` to enforce that a tenant-scoped context is provided and will
raise a 403-style `*AuthorizationError` otherwise.【F:modules/Server/conversation_routes.py†L46-L69】【F:modules/Server/task_routes.py†L342-L352】【F:modules/Server/job_routes.py†L301-L309】

- Conversation retention can only be invoked when the caller's roles include
  `admin` or `system`; other callers receive a 403 response.【F:modules/Server/routes.py†L1036-L1056】
- Task and job route helpers translate validation failures into HTTP-ish status
  codes: 422 (`*ValidationError`) for schema problems, 404 when the record cannot
  be found, 409 for concurrency or transition conflicts, and 403 when the context
  is missing or lacks tenant scoping.【F:modules/Server/task_routes.py†L33-L69】【F:modules/Server/job_routes.py†L26-L60】

When building REST wrappers, convert these exceptions into the framework's
response objects with the indicated status codes.

## Conversation endpoints

| Method & Path | Required fields | Notes |
| --- | --- | --- |
| `POST /messages` | Body must include `conversation_id`, `role`, and `content`. Optional fields cover `metadata`, `extra`, `assets`, `vectors`, `events`, `message_type`, and `status`. | Creates a message, ensures the conversation exists for the tenant, persists metadata, and emits any resulting events over the message bus.【F:modules/Server/conversation_routes.py†L130-L209】【F:modules/Server/conversation_routes.py†L291-L341】 |
| `PATCH /messages/{message_id}` | Body requires `conversation_id`; optional updates mirror creation plus an `events` array. | Records an edit and publishes delta events.【F:modules/Server/conversation_routes.py†L343-L373】 |
| `DELETE /messages/{message_id}` | Body requires `conversation_id`; optional `reason`, `metadata`, `message_type`, `status`. | Performs a soft delete and returns the tombstoned record, emitting delete events.【F:modules/Server/conversation_routes.py†L375-L413】 |
| `GET /conversations/{conversation_id}/messages` | Query supports `page_size`, `direction` (`forward` or `backward`), `cursor`, `include_deleted`, `metadata` filters, `message_types`, and `statuses`. | Returns up to `page_size` messages plus pagination cursors. Cursors encode `created_at|message_id` with URL-safe base64 trimming padding.【F:modules/Server/conversation_routes.py†L15-L89】【F:modules/Server/conversation_routes.py†L415-L457】 |
| `POST /conversations/search` | Body accepts `conversation_ids`, free-text `text`, vector similarity block (`vector.values`), `metadata` filters, `limit`, `offset`, `order`, and optional `top_k`. | Performs hybrid text/vector search across authorized conversations and returns scored hits.【F:modules/Server/conversation_routes.py†L459-L671】 |

**Example** – create a user message:

```json
{
  "conversation_id": "c42d3a0c-12ab-4c51-a920-09ed79d0f2a0",
  "role": "user",
  "content": {
    "text": "Summarize the latest deployment checklist"
  },
  "metadata": {"channel": "slack"},
  "message_type": "utterance"
}
```

### Streaming and polling

`AtlasServer.stream_conversation_events(conversation_id, after?)` yields message
create/edit/delete events. When a message bus is configured, events arrive in
real time; otherwise the helper polls `fetch_message_events` every
`poll_interval` seconds and deduplicates by timestamp and ID.【F:modules/Server/routes.py†L1119-L1150】【F:modules/Server/conversation_routes.py†L649-L677】

## Task endpoints

Task operations live in `TaskRoutes` and are typically exposed under `/tasks`.
All handlers require a tenant-scoped context.

| Method & Path | Required fields | Notes |
| --- | --- | --- |
| `POST /tasks` | Body must include `title` and `conversation_id`. Optional fields: `description`, `status`, `priority`, `owner_id`, `session_id`, `due_at`, and JSON `metadata`. | Creates a task record and publishes any emitted workflow events.【F:modules/Server/task_routes.py†L220-L265】 |
| `PATCH /tasks/{task_id}` | Body may include any of the fields above plus `expected_updated_at` for optimistic concurrency. At least one mutable field must be present. | Updates task metadata, raising 409 if `expected_updated_at` mismatches the stored value.【F:modules/Server/task_routes.py†L267-L309】 |
| `POST /tasks/{task_id}/transition` | Body (or query) must supply `target_status`; callers can also pass `expected_updated_at`. | Validates allowed transitions, dependency state, and concurrency before changing status.【F:modules/Server/task_routes.py†L311-L339】 |
| `GET /tasks/{task_id}` | Optional query `include_events=true` attaches recent workflow events. | Returns the task record or 404 when absent.【F:modules/Server/task_routes.py†L341-L370】 |
| `GET /tasks` | Query parameters: `status` (string or array), `owner_id`, `conversation_id`, `page_size` (default 20, capped at 100), and opaque `cursor`. | Provides keyset pagination. The cursor is a URL-safe base64 encoding of `created_at|task_id`; supply it to fetch the next page. Responses include `items` and a `page` block with `next_cursor`, `page_size`, and `count`.【F:modules/Server/task_routes.py†L45-L103】【F:modules/Server/task_routes.py†L372-L409】 |
| `POST /tasks/search` | Body can include `text` (matched against title/description), `status`, `owner_id`, `conversation_id`, `metadata` equality filters, `limit`, and `offset`. | Performs in-memory filtering on the tenant's tasks and returns `count` plus paged `items`. Useful for dashboards requiring ad-hoc filtering beyond cursor pagination.【F:modules/Server/task_routes.py†L207-L252】【F:modules/Server/task_routes.py†L411-L454】 |

**Example** – update a task with optimistic concurrency:

```json
{
  "title": "QA release checklist",
  "owner_id": "agent-17",
  "expected_updated_at": "2024-05-08T12:45:00Z"
}
```

### Streaming helpers

`AtlasServer.stream_task_events(task_id, after?)` provides an async iterator of
lifecycle events. With a configured message bus it subscribes to
`task.events.{task_id}`; otherwise it polls `get_task(..., with_events=True)` at
`poll_interval` until new events appear, deduplicating by timestamp and ID.【F:modules/Server/task_routes.py†L456-L531】

## Job endpoints

Job orchestration routes back `JobRoutes` and follow similar conventions.

| Method & Path | Required fields | Notes |
| --- | --- | --- |
| `POST /jobs` | Body must include `name`; optional fields mirror task creation plus `conversation_id` and `metadata`. | Persists the job and emits lifecycle analytics.【F:modules/Server/job_routes.py†L64-L118】 |
| `PATCH /jobs/{job_id}` | Body may include `name`, `description`, `owner_id`, `conversation_id`, `metadata`, `expected_updated_at`; at least one field is required. | Applies metadata changes, enforcing optimistic concurrency. | 
| `POST /jobs/{job_id}/transition` | Body or query must provide `target_status` and can pass `expected_updated_at`. | Validates transitions, dependencies, and concurrency before changing state.【F:modules/Server/job_routes.py†L120-L209】 |
| `POST /jobs/{job_id}/rerun` | Optional `expected_updated_at`; reruns completed jobs using the scheduler. | Requires scheduler access; 409 errors are raised when the job lacks schedule metadata or has been modified mid-flight.【F:modules/Server/job_routes.py†L300-L336】 |
| `POST /jobs/{job_id}/schedule/run-now` | Optional `expected_updated_at`. | Immediately enqueues a scheduled manifest run if schedule metadata exists.【F:modules/Server/job_routes.py†L338-L376】 |
| `POST /jobs/{job_id}/schedule/pause` and `/resume` | Optional `expected_updated_at`. | Toggle the manifest schedule via the shared scheduler, enforcing optimistic concurrency.【F:modules/Server/job_routes.py†L236-L298】【F:modules/Server/job_routes.py†L378-L418】 |
| `GET /jobs/{job_id}` | Query booleans `include_schedule`, `include_runs`, `include_events`. | Fetches job metadata with optional joins.【F:modules/Server/job_routes.py†L420-L447】 |
| `GET /jobs` | Query `status` (string or array), `owner_id`, `page_size` (default 20, cap 100), and `cursor`. | Returns job items plus pagination metadata. Cursors are simple `created_at|job_id` tokens; supply them verbatim to request the next slice.【F:modules/Server/job_routes.py†L449-L490】【F:modules/job_store/repository.py†L337-L381】 |
| `POST /jobs/{job_id}/tasks` | Body requires `task_id`; optional `relationship_type` and `metadata`. | Links a task to the job. | 
| `DELETE /jobs/{job_id}/tasks` | Body must include either `link_id` or `task_id`. | Removes the association. | 
| `GET /jobs/{job_id}/tasks` | None. | Lists current link metadata for dashboards.【F:modules/Server/job_routes.py†L492-L542】 |

**Example** – cursor-paginated job listing request:

```
GET /jobs?page_size=25&cursor=2024-05-01T18%3A22%3A11Z%7C5e0c1e2f-91ba-40cb-a0b8-6c1a8a7afc89
```

### Streaming helpers

`AtlasServer.stream_job_events(job_id, after?)` emits scheduler, lifecycle, and
run events. When the message bus exposes topics, the helper subscribes to each
job channel and pushes payloads into an async queue. Without a bus it polls
`get_job(..., with_events=True)` and deduplicates using timestamps and IDs.【F:modules/Server/job_routes.py†L544-L620】

## Persona endpoints

Persona routes are dispatched through `AtlasServer.handle_request` and operate on
persona manifests, analytics, and review state.【F:modules/Server/routes.py†L629-L765】

| Method & Path | Required fields | Notes |
| --- | --- | --- |
| `GET /personas/{persona}/analytics` | Query accepts ISO `start`, `end`, optional integer `limit`, and metric `type`. | Returns aggregated persona metrics for dashboards.【F:modules/Server/routes.py†L630-L655】【F:modules/Server/routes.py†L526-L575】 |
| `GET /personas/{persona}/review` | None. | Retrieves the current review status including expiry calculations.【F:modules/Server/routes.py†L656-L671】【F:modules/Server/routes.py†L575-L618】 |
| `POST /personas/{persona}/review` | Body may include `reviewer`, `expires_at` or `expires_in_days`, and optional `notes`. | Records a review attestation, updates the audit trail, and recomputes status.【F:modules/Server/routes.py†L712-L759】【F:modules/Server/routes.py†L600-L618】 |
| `POST /personas/{persona}/tools` | Body accepts `tools` (string/array/object) and optional `rationale`. | Normalizes tool identifiers, validates against known catalogs, persists the manifest update, and returns the sanitized list.【F:modules/Server/routes.py†L666-L723】【F:modules/Server/routes.py†L772-L841】 |
| `POST /personas/{persona}/skills` | Body accepts `skills` and optional `rationale`. | Mirrors the tool update path for allowed skills.【F:modules/Server/routes.py†L724-L732】【F:modules/Server/routes.py†L882-L950】 |
| `POST /personas/{persona}/export` | Optional `signing_key`. | Produces a signed bundle of the persona definition for distribution.【F:modules/Server/routes.py†L733-L748】【F:modules/Server/routes.py†L980-L1010】 |
| `POST /personas/import` | Body must include base64 `bundle`; optional `signing_key` and `rationale`. | Imports a persona bundle and persists it after validation.【F:modules/Server/routes.py†L749-L758】【F:modules/Server/routes.py†L1012-L1033】 |
| `GET /skills` | Optional `persona` filter. | Returns skill registry entries filtered by persona allowlists.【F:modules/Server/routes.py†L672-L687】【F:modules/Server/routes.py†L387-L427】 |
| `GET /tools` | Query filters `capability`, `safety_level`, `persona`, optional `include_provider_health`. | Lists tool manifests after applying capability, safety, and persona filters. When `include_provider_health` is truthy each entry's `health` block includes the most recent provider routing snapshot, per-provider `last_call` metrics, and a `last_invocation` summary for dashboards.【F:modules/Server/routes.py†L688-L712】【F:modules/Server/routes.py†L2523-L2550】 |

When provider health is requested the serialized response includes a `health.last_invocation`
object tracking the last provider selected, whether it succeeded, the measured
latency, and the sampling timestamp, as well as `health.providers[NAME].last_call`
records reflecting per-provider latency and success history to keep dashboards in
sync with router decisions.【F:modules/orchestration/capability_registry.py†L1008-L1038】【F:modules/Server/routes.py†L2523-L2550】

**Example** – update persona tooling:

```json
{
  "tools": ["summarize", "escalate"],
  "rationale": "Limit toolkit to escalation workflow"
}
```

Persona analytics payloads now expose `anomalies` (scoped to the requested metric
category) and `recent_anomalies` collections summarizing the latest threshold
breaches. When an anomaly is recorded the service publishes a
`persona_metrics.alert` event containing the persona identifier, metric name,
observed value, current baseline statistics, and suggested remediation steps so
dashboards can subscribe for proactive notifications.【F:modules/Server/routes.py†L630-L685】【F:modules/analytics/persona_metrics.py†L115-L225】

## Blackboard endpoints

The shared blackboard supports collaborative annotations keyed by scope type and
identifier (for example `conversation` and a conversation UUID).【F:modules/Server/routes.py†L640-L706】【F:modules/Server/routes.py†L460-L560】

| Method & Path | Required fields | Notes |
| --- | --- | --- |
| `GET /blackboard/{scope_type}/{scope_id}` | Optional query `summary=true` for aggregate counts or `category` to filter entries. | Returns entries for the scope or a summary when `summary` is set.【F:modules/Server/routes.py†L467-L487】 |
| `GET /blackboard/{scope_type}/{scope_id}/{entry_id}` | None. | Fetches a single entry; 404 (translated from `KeyError`) when missing.【F:modules/Server/routes.py†L648-L663】 |
| `POST /blackboard/{scope_type}/{scope_id}` | Body requires `category`, `title`, and `content`; optional `author`, `tags`, `metadata`. | Publishes a new entry and returns the stored record.【F:modules/Server/routes.py†L488-L520】 |
| `PATCH /blackboard/{scope_type}/{scope_id}/{entry_id}` | Body may include `title`, `content`, `tags`, `metadata`. | Updates an existing entry; raises `KeyError` when the entry does not exist.【F:modules/Server/routes.py†L522-L546】 |
| `DELETE /blackboard/{scope_type}/{scope_id}/{entry_id}` | None. | Deletes the entry and returns a boolean `success`.【F:modules/Server/routes.py†L548-L560】 |

### Streaming helpers

`AtlasServer.stream_blackboard_events(scope_type, scope_id)` delegates to the
`stream_blackboard` helper, which subscribes to the message bus topic
`blackboard.{scope_type}.{scope_id}` and yields each published payload. Use this
for Server-Sent Events (SSE) or WebSocket push surfaces.【F:modules/Server/routes.py†L561-L566】【F:modules/orchestration/blackboard.py†L402-L432】

## Pagination, cursors, and filtering quick reference

- **Conversation messages** – `page_size` (default 20, max configured limit),
  `cursor` encodes `created_at|message_id` via URL-safe base64; supports forward
  and backward directions and metadata/message-type/status filters.【F:modules/Server/conversation_routes.py†L58-L89】【F:modules/Server/conversation_routes.py†L415-L457】
- **Tasks** – `page_size` (default 20, max 100) and base64 `cursor` encoding
  `created_at|task_id`. Filter by `status` (string or array), `owner_id`, and
  `conversation_id`. Search endpoint adds `text` and `metadata` filters.【F:modules/Server/task_routes.py†L45-L103】【F:modules/Server/task_routes.py†L372-L454】
- **Jobs** – `page_size` (default 20, max 100) with plaintext `created_at|job_id`
  cursors. Filters include `status` (string or array) and `owner_id`. The backend
  enforces keyset pagination on `created_at` and `id` for stable ordering.【F:modules/Server/job_routes.py†L449-L490】【F:modules/job_store/repository.py†L337-L381】

## Error semantics

The route helpers raise structured exceptions that should be mapped to HTTP
responses:

| Exception | Suggested status | Description |
| --- | --- | --- |
| `ConversationValidationError`, `TaskValidationError`, `JobValidationError` | 422 | Request body or query violates schema or type constraints.【F:modules/Server/conversation_routes.py†L36-L43】【F:modules/Server/task_routes.py†L35-L37】【F:modules/Server/job_routes.py†L35-L37】 |
| `ConversationAuthorizationError`, `TaskAuthorizationError`, `JobAuthorizationError` | 403 | Missing tenant context or insufficient privilege (for example retention trigger).【F:modules/Server/conversation_routes.py†L30-L33】【F:modules/Server/task_routes.py†L39-L41】【F:modules/Server/job_routes.py†L39-L41】 |
| `ConversationNotFoundError`, `TaskNotFoundRouteError`, `JobNotFoundRouteError` | 404 | Target record is absent or inaccessible.【F:modules/Server/conversation_routes.py†L33-L35】【F:modules/Server/task_routes.py†L41-L43】【F:modules/Server/job_routes.py†L41-L43】 |
| `TaskConflictError`, `JobConflictError` | 409 | Optimistic concurrency mismatches, dependency violations, or invalid transitions.【F:modules/Server/task_routes.py†L43-L45】【F:modules/Server/job_routes.py†L45-L47】 |

Translate these into JSON API error payloads consistent with your gateway or web
framework.

