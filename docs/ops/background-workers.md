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
