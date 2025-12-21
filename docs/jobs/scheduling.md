---
audience: integrators, contributors, operators/admins
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/orchestration/job_scheduler.py; modules/job_store/service.py
---

# Job scheduling

Job schedules coordinate recurring automation through `modules.orchestration.job_scheduler`. Schedules are defined in job manifests (`recurrence` block) and persisted on the job record. When `JobService.transition_job` moves a job into `scheduled` or `running`, validation ensures schedule metadata exists when required and that roster prerequisites are satisfied.

## Defining schedules

A recurrence block typically contains:

```json
{
  "cron": "0 * * * *",
  "timezone": "UTC",
  "start_date": "2024-05-01T12:00:00Z",
  "end_date": "2024-08-01T00:00:00Z"
}
```

Supported fields align with `modules/Jobs/schema.json` and are normalized into `metadata.recurrence` on the job record. The scheduler computes the cron expression (or interval cadence), registers it with the task queue, and persists schedule records in the job store with the following metadata:

- `metadata.task_queue_job_id`: Identifier returned by the task queue when the recurrence is registered.
- `metadata.state`: Queue status (for example `scheduled`, `paused`) surfaced in UI payloads.
- `metadata.retry_policy`: Retry policy passed to the task queue for the manifest.
- `metadata.recurrence`: Copy of the manifest recurrence for traceability.
- `next_run_at`: Persisted on the schedule record for dashboards and `run-now` previews.

Lifecycle events (`running` → `succeeded`/`failed`) should include scheduling metadata (for example the run identifier) in the `metadata` argument so analytics can group throughput and SLA adherence per schedule.

## Runtime controls

AtlasServer exposes scheduler controls via the job routes:

- `POST /jobs/{job_id}/schedule/pause` and `/resume` toggle the manifest schedule and update `metadata.schedule_state`.
- `POST /jobs/{job_id}/schedule/run-now` enqueues an immediate run using the registered manifest name/persona pair and returns queue status inside `metadata.schedule`.
- `POST /jobs/{job_id}/rerun` replays a completed job through the scheduler, maintaining optimistic concurrency with `expected_updated_at`.

## Monitoring scheduled jobs

- Subscribe to `jobs.metrics.lifecycle` to track terminal events and latency between scheduled runs.
- Use `get_job_lifecycle_metrics(tenant_id=...)` to evaluate throughput per tenant or persona.
- Combine job metrics with [task lifecycle analytics](../tasks/overview.md#analytics-and-rollups) to identify bottlenecks in scheduled workflows.

If a scheduled job repeatedly fails validation (missing roster, incomplete dependencies), the scheduler surfaces warnings through the job bus events while analytics capture the failed lifecycle transitions for dashboards.
