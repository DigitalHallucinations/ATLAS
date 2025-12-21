---
audience: Backend developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules.job_store.service.JobService
---

# Job APIs

The job service lives in `modules.job_store.service.JobService` and offers lifecycle, roster, and task-link management. Key entry points:

- `create_job` – persists a new record, emits `job.created`, and records a `jobs.metrics.lifecycle` event.
- `update_job` – applies metadata changes (owner, description) and records lifecycle analytics for significant updates.
- `transition_job` – validates allowed status transitions, linked task completion, and SLA metadata before recording analytics and emitting bus events.
- `link_task` / `unlink_task` – associate tasks with a job. Linked task status is enforced during transitions to `running` or `succeeded`.
- `list_linked_tasks` – returns the current linkage payload used by UI task rollups.

### Access patterns

When exposing APIs (REST or RPC), map requests to the JobService methods above. Persisting job transitions automatically produces analytics, so UI layers and downstream automations should not emit lifecycle events directly. Instead:

1. Call `transition_job` with the requested status (for example `running` → `succeeded`).
2. Subscribe to `jobs.metrics.lifecycle` (analytics) and `job.status_changed` (operational) topics for auditing.
3. Use `get_job_lifecycle_metrics` for persona-aware rollups displayed alongside [task metrics](../tasks/overview.md#analytics-and-rollups).

Additional metadata can be supplied in the `metadata` argument of service calls; these values are stored with lifecycle events and available in analytics payloads.
