# Task Queue Tool

The task queue base tool provides a durable scheduling surface for ATLAS agents.
It is backed by [APScheduler](https://apscheduler.readthedocs.io/) and **requires
a PostgreSQL job store**. By default the queue shares the conversation-store
connection managed by `ConfigManager`, but it can be pointed at a dedicated
PostgreSQL database when needed.

## Features

* **Durable enqueue:** queue one-off tasks with optional execution delays or
  absolute run timestamps. Tasks are stored in the job store so they survive
  process restarts.
* **Cron scheduling:** register cron-style recurring jobs and update them
  idempotently via the manifest operations.
* **Retry policy:** one-off jobs automatically retry with exponential backoff.
  Defaults can be overridden via `ConfigManager` and the tool exposes the active
  policy through `TaskQueueService.get_retry_policy()`.
* **Monitoring hooks:** register callbacks with
  `TaskQueueService.add_monitor()` to observe state transitions, retry events,
  and backoff decisions.

## Configuration

The queue reads its configuration through `ConfigManager`. All job-store sources
must be PostgreSQL DSNs (for example,
`postgresql+psycopg://atlas:atlas@localhost:5432/atlas`). Configuration keys are
resolved in the following priority order:

| Key | Type | Description |
| --- | ---- | ----------- |
| `job_scheduling.job_store_url` | string | Primary knob for the scheduler job store. Must be a PostgreSQL DSN. |
| `task_queue.jobstore_url` | string | Legacy/dedicated task-queue override. Also must be PostgreSQL. |
| `TASK_QUEUE_JOBSTORE_URL` | string | Environment override for the job store URL. |

When no explicit URL is provided the queue falls back to the conversation-store
database (which defaults to
`postgresql+psycopg://atlas:atlas@localhost:5432/atlas`).

Additional keys continue to control runtime behaviour:

| Key | Type | Description |
| --- | ---- | ----------- |
| `task_queue.max_workers` / `TASK_QUEUE_MAX_WORKERS` | int | Number of executor threads used for dispatching tasks (default: `4`). |
| `task_queue.misfire_grace_time` | float | Seconds a job is allowed to run late before being considered missed (default: `60`). |
| `task_queue.coalesce` | bool | Whether to coalesce missed runs for recurring jobs (default: `false`). |
| `task_queue.max_instances` | int | Maximum concurrent executions per job (default: `1`). |
| `task_queue.retry_policy` | object | Mapping with `max_attempts`, `backoff_seconds`, `backoff_multiplier`, and `jitter_seconds` fields. |

## Manifest Operations

The manifest exposes four actions with the `task_queue` capability:

* `task_queue_enqueue` – enqueue one-off jobs. Requires an `idempotency_key`
  derived from the logical task identifier to safely retry requests.
* `task_queue_schedule` – register or replace cron schedules. Provide either a
  `cron_schedule` string (classic five-field cron expression) or structured
  `cron_fields` mapping.
* `task_queue_cancel` – cancel queued or scheduled jobs by `job_id`. The
  `idempotency_key` should match the `job_id` for safe retries.
* `task_queue_status` – inspect current state, retry attempts, and next
  execution time.

All write operations declare `side_effects: "write"` and surface an
`idempotency_key` requirement to guarantee predictable retries.

## Deployment Notes

* Ensure the APScheduler dependencies (`APScheduler`, `SQLAlchemy`, `psycopg`)
  are installed in the runtime environment.
* Provision a PostgreSQL database (or schema) that the queue can use. The
  scheduler validates the DSN and refuses to start without a PostgreSQL URL.
* APScheduler spawns worker threads. Shut down services cleanly via
  `TaskQueueService.shutdown()` during teardown to avoid lingering threads.
* The queue does not perform payload validation beyond requiring JSON-compatible
  values. Downstream task executors must validate inputs before running
  side-effecting operations.

## Limitations

* Recurring jobs share a single retry policy. Per-job retry configuration can be
  injected programmatically by overriding the `retry_policy` argument at
  enqueue/schedule time.
* The built-in executor is synchronous and runs tasks in a thread pool. If long
  running jobs are expected, consider integrating a dedicated worker process or
  asynchronous executor.
