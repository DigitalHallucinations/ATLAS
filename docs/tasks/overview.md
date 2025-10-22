# Task metadata and lifecycle

This guide summarizes how tasks are described, orchestrated, and surfaced in ATLAS. It covers manifest fields, lifecycle transitions, service APIs, and the dashboards that consume task analytics.

## Manifest fields

Task manifests live under `modules/Tasks/tasks.json` and persona overrides under `modules/Personas/<name>/Tasks/tasks.json`. Each entry is validated by `modules/Tasks/manifest_loader.py` and normalized into a `TaskMetadata` record. The following fields are accepted:

| Field | Description |
| --- | --- |
| `name` | Unique identifier used by orchestrators and UI clients. |
| `summary` | Short description shown in selectors and capability catalogs. |
| `description` | (Optional) Extended guidance for human operators. |
| `required_skills` | Skills the orchestrator must provision before running the task. |
| `required_tools` | Tools that should be available to personas executing the task. |
| `acceptance_criteria` | Checklist of conditions that define success. |
| `escalation_policy` | Structured contact information for escalation flows. |
| `tags` | Free-form labels used for filtering and routing. |
| `priority` | (Optional) Relative priority surfaced to routing heuristics. |
| `persona` | Set automatically for persona-specific overrides. |
| `source` | File path recorded by the loader for troubleshooting. |

Persona manifests can inherit shared tasks by setting `extends` and overriding a subset of the above fields. When a persona omits a field, the loader merges the value from the shared manifest.

## Lifecycle states

The task domain model defines the following lifecycle states (`modules/task_store/models.py`):

- `draft` – task created but not yet queued for execution.
- `ready` – task queued and waiting for assignment.
- `in_progress` – task actively being worked on.
- `review` – task awaiting validation or escalation.
- `done` – task successfully completed.
- `cancelled` – task aborted before completion.

`modules/task_store/service.py` enforces the transition graph (`_ALLOWED_TRANSITIONS`) and emits `task.created`, `task.updated`, and `task.status_changed` events via the message bus. Invalid transitions raise `TaskTransitionError`, and dependency checks prevent advancing tasks with incomplete prerequisites.

## Service APIs and metrics

`TaskService` exposes `create_task`, `update_task`, and `transition_task`, delegating persistence to `TaskStoreRepository`. Each operation records lifecycle analytics using `modules.analytics.persona_metrics.record_task_lifecycle_event`, which persists metrics in `TaskLifecycleEvent` records and publishes `task_metrics.lifecycle` messages. Aggregated metrics—including success rates, latency, and reassignment counts—are available through `get_task_lifecycle_metrics`.

## UI and dashboards

The `CapabilityRegistry.summary` helper bundles tool, skill, and task catalogs (including compatibility flags and tool health) for persona-aware dashboards. Task lifecycle metrics published on the message bus feed monitoring widgets alongside persona tool analytics, enabling UI workflows such as task completion funnels and reassignment alerts.
