---
audience: Persona authors and backend developers
status: in_review
last_verified: 2025-05-09
source_of_truth: modules.Jobs.manifest_loader.load_job_metadata; modules/Jobs/schema.json; CapabilityRegistry
---

# Job manifests

Job manifests live alongside persona tool and task manifests under `modules/Jobs/` and extend the scheduling layer with persona assignments, linked tasks, and required capabilities. Each manifest entry is loaded by `modules.Jobs.manifest_loader.load_job_metadata`, cached in the [`CapabilityRegistry`](../tasks/overview.md#ui-and-dashboards), and surfaced to orchestrators through the `CapabilityRegistry.summary` payload.

A manifest entry is validated against `modules/Jobs/schema.json` and normalized by `JobMetadata`. Required and optional fields include:

| Field | Requirement | Purpose |
| --- | --- | --- |
| `name` | Required | Unique identifier for the job. Used by the scheduler and manifests to correlate runs. |
| `summary` | Required | Short description shown in dashboards and pickers. |
| `description` | Optional | Extended explanation for UI detail panes. |
| `personas` | Required (non-empty array) | Persona roster that owns the job. When loaded from persona-specific manifests, the roster defaults to the persona directory if omitted. |
| `extends` | Optional | Name of a base manifest to merge via `merge_with_base`; useful for persona overrides. |
| `required_skills` / `required_tools` | Required (arrays, can be empty) | Capabilities that must exist in the registry before activation. |
| `task_graph` | Required (non-empty array) | Ordered graph of tasks with `task`, optional `depends_on`, `description`, and free-form `metadata` per node. |
| `recurrence` | Optional object | Schedule metadata (`frequency`, `interval`, `cron`, `timezone`, `start_date`, `end_date`); preserved as `metadata.recurrence` on the job. |
| `acceptance_criteria` | Required (non-empty array) | Completion checklist rendered alongside analytics. |
| `escalation_policy` | Required object | Includes `level`, `contact`, optional `timeframe`, `triggers`, `actions`, `notes`, and `auto_escalate`. |
| `accepts_external_requests` | Optional boolean | Declares whether the job can be triggered by external webhooks or RPC integrations. |
| `persona` | Derived | Populated during load for persona-scoped manifests and retained in the Capability Registry. |
| `source` | Derived | Relative manifest path for traceability. |

Job manifests may also embed `metadata` blocks used by downstream automation (for example SLA definitions or escalation contacts). These metadata values are preserved on lifecycle events and surface in [analytics payloads](lifecycle.md#metrics-and-message-topics).

To add or update a manifest:

1. Author the JSON manifest under the persona-specific directory in `modules/Personas/<Persona>/Jobs/` (matching existing naming conventions).
2. Include `recurrence` details if the job should be scheduled and ensure `task_graph` nodes list dependencies explicitly. Use `extends` to inherit shared defaults from `modules/Jobs/jobs.json` when applicable.
3. Run existing manifest validation tests (`pytest tests/test_job_service.py`) to confirm schema compatibility.
4. Refresh the Capability Registry in your environment or restart the service so dashboards pick up the latest metadata and scheduler registrations.
