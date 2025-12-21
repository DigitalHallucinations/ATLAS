---
audience: UI engineers and operators
status: in_review
last_verified: 2025-05-09
source_of_truth: CapabilityRegistry.summary; get_task_lifecycle_metrics; get_job_lifecycle_metrics
---

# Job dashboards and UI workflows

Dashboards render job and task analytics side-by-side so operators can validate throughput, SLA adherence, and downstream task health. The `CapabilityRegistry.summary` payload now includes an `analytics` section with:

- `tasks`: Output from `get_task_lifecycle_metrics`
- `jobs`: Output from `get_job_lifecycle_metrics`

UI payloads also include scheduler and manifest context:

- `metadata.manifest`: `{ "name": "<manifest_name>", "persona": "<owner>", "source": "<path>" }`
- `metadata.recurrence`: Copy of the manifest recurrence block for display alongside the schedule.
- `metadata.schedule` or `metadata.schedule_state`: Task queue status, retry policy, and the next scheduled run derived from the scheduler registration.
- `linked_tasks`: Result of `list_linked_tasks` (also available via `GET /jobs/{job_id}/tasks`) for inline dependency rollups.

UI components can:

1. Display persona-aware job queues with status totals and per-job success history.
2. Surface SLA gauges by combining `analytics["jobs"]["sla"]` with schedule metadata.
3. Overlay task completion rates (`analytics["tasks"]`) to highlight dependencies affecting job success.
4. Expose schedule controls (pause/resume/run-now) by calling the routes documented in [Job scheduling](scheduling.md#runtime-controls) and reflecting `metadata.schedule_state` changes in place.
5. Render event timelines by subscribing to `stream_job_events` for live updates or by fetching `include_events=true` from `GET /jobs/{job_id}` for historical context.

When users drill into a job, show the recent lifecycle events (with metadata describing schedule runs, SLAs, or linked task counts). Combine these events with task rollups to present a complete funnel from job activation through dependent tasks.

For guidance on rendering persona tasks, refer to the [task overview dashboard section](../tasks/overview.md#ui-and-dashboards). Job-specific UI should reuse those widgets where possible, adding columns for throughput per hour and SLA adherence derived from lifecycle metrics.
