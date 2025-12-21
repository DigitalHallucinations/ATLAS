---
audience: Backend developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules.orchestration.job_scheduler; modules.job_store.service.JobService.transition_job
---

# Job scheduling

Job schedules coordinate recurring automation through `modules.orchestration.job_scheduler`. Schedules are defined in job manifests (`schedule` block) and persisted on the job record. When `JobService.transition_job` moves a job into `scheduled` or `running`, validation ensures the schedule is present if required.

## Defining schedules

A schedule block typically contains:

```json
{
  "cadence": "0 * * * *",
  "timezone": "UTC",
  "next_run_at": "2024-05-01T12:00:00Z",
  "policy": {
    "retry": {"max_attempts": 3, "backoff_seconds": 600}
  }
}
```

The scheduler reads the persisted job record, enqueues executions, and updates `next_run_at` as runs complete. Lifecycle events (`running` → `succeeded`/`failed`) should include scheduling metadata (for example the run identifier) in the `metadata` argument so analytics can group throughput and SLA adherence per schedule.

## Monitoring scheduled jobs

- Subscribe to `jobs.metrics.lifecycle` to track terminal events and latency between scheduled runs.
- Use `get_job_lifecycle_metrics(tenant_id=...)` to evaluate throughput per tenant or persona.
- Combine job metrics with [task lifecycle analytics](../tasks/overview.md#analytics-and-rollups) to identify bottlenecks in scheduled workflows.

If a scheduled job repeatedly fails validation (missing roster, incomplete dependencies), the scheduler surfaces warnings through the job bus events while analytics capture the failed lifecycle transitions for dashboards.
