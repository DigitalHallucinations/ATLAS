---
audience: Persona authors and backend developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules.Jobs.manifest_loader.load_job_metadata; modules/Jobs/; CapabilityRegistry
---

# Job manifests

Job manifests live alongside persona tool and task manifests under `modules/Jobs/` and extend the scheduling layer with persona assignments, linked tasks, and required capabilities. Each manifest entry is loaded by `modules.Jobs.manifest_loader.load_job_metadata`, cached in the [`CapabilityRegistry`](../tasks/overview.md#ui-and-dashboards), and surfaced to orchestrators through the `CapabilityRegistry.summary` payload.

A manifest defines:

| Field | Purpose |
| --- | --- |
| `name` | Unique identifier for the job. |
| `persona` | Persona responsible for the job orchestration. |
| `summary` / `description` | Short and extended explanations rendered in dashboards and job pickers. |
| `personas` | Optional roster of collaborating personas referenced by lifecycle validation. |
| `required_skills` / `required_tools` | Capabilities that must be available when the job activates. |
| `required_capabilities` | Registry flags enforced before activation (for example, environment or compliance requirements). |
| `schedule` | Optional cron-like payload that pairs with lifecycle transitions documented in [scheduling patterns](scheduling.md). |

Job manifests may also embed `metadata` blocks used by downstream automation (for example SLA definitions or escalation contacts). These metadata values are preserved on lifecycle events and surface in [analytics payloads](lifecycle.md#metrics-and-message-topics).

To add or update a manifest:

1. Author the JSON manifest under the persona-specific directory in `modules/Personas/<Persona>/Jobs/` (matching existing naming conventions).
2. Run existing manifest validation tests (`pytest tests/test_job_service.py`) to confirm schema compatibility.
3. Refresh the Capability Registry in your environment or restart the service so dashboards pick up the latest metadata.
