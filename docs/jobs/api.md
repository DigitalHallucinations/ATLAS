---
audience: integrators, contributors
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/job_store/service.py; modules/Server/job_routes.py
---

# Job APIs

The job service lives in `modules.job_store.service.JobService` and offers lifecycle, roster, and task-link management. Key entry points:

- `create_job` – persists a new record, emits `job.created`, and records a `jobs.metrics.lifecycle` event.
- `update_job` – applies metadata changes (owner, description) and records lifecycle analytics for significant updates.
- `transition_job` – validates allowed status transitions, linked task completion, and SLA metadata before recording analytics and emitting bus events.
- `link_task` / `unlink_task` – associate tasks with a job. Linked task status is enforced during transitions to `running` or `succeeded`.
- `list_linked_tasks` – returns the current linkage payload used by UI task rollups.

## REST route mappings

The AtlasServer routes in `modules/Server/job_routes.py` map directly onto the service methods above:

| Route | JobService mapping | Notes |
| --- | --- | --- |
| `POST /jobs` | `create_job` | Accepts `name`, optional `description`, `status`, `owner_id`, `conversation_id`, and `metadata`. Emits `jobs.created` and lifecycle analytics automatically. |
| `PATCH /jobs/{job_id}` | `update_job` | Accepts partial metadata updates plus `expected_updated_at` for optimistic concurrency. |
| `POST /jobs/{job_id}/transition` | `transition_job` | Validates roster, linked task completion, and allowed transitions before changing status. |
| `POST /jobs/{job_id}/tasks` / `DELETE /jobs/{job_id}/tasks` | `link_task` / `unlink_task` | Links or removes tasks. `relationship_type` and `metadata` are optional on link creation. |
| `GET /jobs/{job_id}/tasks` | `list_linked_tasks` | Returns the current linkage payload used by UI rollups. |
| `GET /jobs` | `list_jobs` | Supports filtering by `status`, `owner_id`, pagination `cursor`, and `page_size` (default 20, max 100). |
| `GET /jobs/{job_id}` | `get_job` | Optional `include_schedule`, `include_runs`, and `include_events` flags hydrate scheduler metadata and event history. |
| `POST /jobs/{job_id}/schedule/pause` | `pause_schedule` | Requires a persisted schedule; returns merged `metadata.schedule_state` with the new status. |
| `POST /jobs/{job_id}/schedule/resume` | `resume_schedule` | Resumes a paused manifest schedule and refreshes schedule metadata on the job record. |
| `POST /jobs/{job_id}/schedule/run-now` | `run_now` | Enqueues an immediate run for scheduled manifests; returns queue status in `metadata.schedule`. |
| `POST /jobs/{job_id}/rerun` | `rerun_job` | Replays a completed job via the scheduler, honoring `expected_updated_at` for concurrency. |

### Payload expectations

- **Metadata**: The routes coerce `metadata` to a mapping and persist it as-is; include manifest identifiers under `metadata.manifest` to align scheduler and UI payloads.
- **Tenant scoping**: All routes require a tenant-scoped context; multi-tenant filtering is enforced in `JobRoutes._require_context`.
- **Events**: Lifecycle analytics (`jobs.metrics.lifecycle`) and operational events (`job.status_changed`) are emitted automatically during transitions and reruns. UI consumers can subscribe via `stream_job_events` for SSE/WS delivery.

### Access patterns

When exposing APIs (REST or RPC), map requests to the JobService methods above. Persisting job transitions automatically produces analytics, so UI layers and downstream automations should not emit lifecycle events directly. Instead:

1. Call `transition_job` with the requested status (for example `running` → `succeeded`).
2. Subscribe to `jobs.metrics.lifecycle` (analytics) and `job.status_changed` (operational) topics for auditing.
3. Use `get_job_lifecycle_metrics` for persona-aware rollups displayed alongside [task metrics](../tasks/overview.md#analytics-and-rollups).

Additional metadata can be supplied in the `metadata` argument of service calls; these values are stored with lifecycle events and available in analytics payloads.
