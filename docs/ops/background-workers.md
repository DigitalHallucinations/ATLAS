---
audience: Operators and backend developers
status: in_review
last_verified: 2025-12-21
last_updated_hint: Added queue tuning examples and monitoring hooks for worker health.
source_of_truth: modules/Tools/Base_Tools/task_queue.py; worker configs under ATLAS/config/
---

# Background worker health and redundancy

ATLAS relies on background services for scheduled jobs, message dispatch, and
conversation maintenance. The job scheduler is backed by the task queue service
(`modules/Tools/Base_Tools/task_queue.py`) and is initialized by
`ConfigManager`. This document outlines recommended sizing defaults, redundancy
expectations, and monitoring hooks.

## Sizing defaults

* **Redis backends**: enterprise deployments default to **8 workers** and a
  **500 message queue** to absorb bursty automation workloads.
* **In-memory backends**: local-first configurations default to **4 workers**
  and a **100 message queue** to limit memory use.

During setup these defaults are applied automatically based on the selected
messaging backend. Enterprise wizard flows require explicit confirmation of the
worker and queue values to avoid undersized schedulers.

## Health checks

* `TaskQueueService.ensure_running()` performs a startup check to confirm the
  APScheduler instance is active. The ATLAS process logs a warning if the check
  fails.
* `JobScheduler.get_health_snapshot()` returns the current registrations,
  tenant context, and the associated task queue snapshot.
* The server exposes `GET /status/workers`, returning task queue and job
  scheduler health payloads suitable for dashboards.

## Redundancy and alerting

* **Redis with persistence** is recommended for production jobstores and
  message buses; configure `redis://` backends with AOF/RDB persistence and
  replica promotion where possible.
* **Database-backed jobstores** should reside on HA PostgreSQL instances to
  preserve scheduled work across restarts.
* **Alerting hooks**: monitor the `/status/workers` endpoint for `running: false`
  or unexpected drops in `scheduled_jobs`. Alerts can be forwarded to existing
  observability stacks by scraping the endpoint or tailing ATLAS logs for
  `Background worker health check failed` warnings.

## Monitoring and tuning examples

Monitor the worker surface with lightweight HTTP checks and log scraping:

```bash
curl -s http://<atlas-host>:<port>/status/workers | jq '.scheduler.running, .task_queue.depth'
journalctl -u atlas --grep "Background worker health check failed" --since "10 minutes ago"
```

If queue depth regularly exceeds the sizing defaults, increase worker counts and queue depth together to avoid starvation:

```yaml
messaging:
  backend: redis
  worker_pool:
    workers: 12
    queue_size: 1000
```

For bursty local runs, keep the in-memory backend small to avoid runaway memory usage:

```yaml
messaging:
  backend: in_memory
  worker_pool:
    workers: 4    # keep under 6 for laptops
    queue_size: 150
```

After changes, restart ATLAS and re-query `/status/workers` to confirm the scheduler and task queue report the new limits.
