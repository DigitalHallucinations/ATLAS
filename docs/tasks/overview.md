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

## Operational task catalog

The following manifest-backed tasks are now available to operators. Create a task by calling `AtlasServer.create_task` (or the `POST /tasks` HTTP route) with the `title` set to the manifest name and include the persona routing hint in `metadata.manifest_task`. The orchestrator will look up the manifest entry and hydrate the downstream workflow with the skills, tools, and acceptance criteria defined below.

| Manifest name | Persona | Trigger phrase | Deliverable |
| --- | --- | --- | --- |
| `MissionControlWeeklyBrief` | ATLAS | `metadata.manifest_task="MissionControlWeeklyBrief"` | Weekly leadership brief covering status, risks, and upcoming decisions for mission control stakeholders. |
| `MissionControlDailyStandup` | ATLAS | `metadata.manifest_task="MissionControlDailyStandup"` | 24-hour mission progress snapshot highlighting blockers, decisions, and near-term focus areas. |
| `ChangeCalendarImpactReview` | ATLAS | `metadata.manifest_task="ChangeCalendarImpactReview"` | Pre-release audit that spots scheduling conflicts, missing approvals, and coordination gaps. |
| `OperationalRiskFlashReport` | ATLAS, ComplianceOfficer | `metadata.manifest_task="OperationalRiskFlashReport"` | Rapid triage of emerging operational or policy risks with recommended mitigations. |
| `AutomationPolicyPrecheck` | ATLAS, ResumeGenius | `metadata.manifest_task="AutomationPolicyPrecheck"` | Compliance pre-check documenting policy coverage, risks, and go/no-go guidance for automation changes. |
| `KnowledgeArchiveBackfill` | KnowledgeCurator | `metadata.manifest_task="KnowledgeArchiveBackfill"` | Backfills missing knowledge cards with citations, owners, and follow-up actions. |
| `KnowledgeArchiveRefreshSweep` | KnowledgeCurator | `metadata.manifest_task="KnowledgeArchiveRefreshSweep"` | Quarterly audit ensuring high-traffic knowledge entries stay accurate and current. |
| `WeatherOperationsSnapshot` | WeatherGenius | `metadata.manifest_task="WeatherOperationsSnapshot"` | Real-time operations snapshot that translates weather alerts into field deployment guidance. |
| `WeatherFieldDeploymentBrief` | WeatherGenius | `metadata.manifest_task="WeatherFieldDeploymentBrief"` | Field-ready brief blending alerts, logistics constraints, and recommended posture levels. |
| `SevereWeatherTabletopDrill` | WeatherGenius | `metadata.manifest_task="SevereWeatherTabletopDrill"` | Tabletop exercise plan with scenario injects, expected actions, and evaluation checklists. |
| `ClinicalEvidenceSnapshot` | MEDIC, DocGenius | `metadata.manifest_task="ClinicalEvidenceSnapshot"` | Concise evidence digest summarizing current guidelines, key studies, and safety considerations. |
| `ClinicalGuidelineUpdate` | MEDIC, DocGenius | `metadata.manifest_task="ClinicalGuidelineUpdate"` | Synthesizes new evidence into actionable guideline updates for care teams. |
| `AdverseEventRapidReview` | MEDIC | `metadata.manifest_task="AdverseEventRapidReview"` | Rapid literature and policy check supporting urgent adverse event investigations. |
| `ResumePipelineQualityReview` | ResumeGenius | `metadata.manifest_task="ResumePipelineQualityReview"` | Persona-scoped pipeline audit assessing ATS scores, systemic gaps, and remediation actions. (Source: `modules/Personas/ResumeGenius/Tasks/tasks.json`.) |

### Persona-specific tooling quick reference

Routing and playbooks can now lean on purpose-built async helpers that live alongside each persona's toolbox. ATS-focused manifests now live solely in the ResumeGenius persona manifest (`modules/Personas/ResumeGenius/Tasks/tasks.json`):

- **ATLAS** – `task_catalog_snapshot` surfaces filtered results from `AtlasServer.get_task_catalog`, making it easy to preview task availability before dispatching work (e.g., `{"persona": "DocGenius", "tags": ["clinical"]}`).
- **Cleverbot** – `persona_backstory_sampler(style)` returns seeded openers/callbacks for `humor`, `sarcasm`, or `casual` banter so operators can warm up conversations with consistent tone.
- **DocGenius** – `generate_doc_outline(signature, language="python")` parses function/class signatures and emits the standard DocGenius docstring scaffold (sections, parameter prompts, follow-ups).
- **Einstein** – `relativity_scenario(topic)` provides curated thought experiments (`time_dilation`, `energy`, `gravity`) with prompts and variables to anchor physics explanations.
- **Nikola Tesla** – `wireless_power_brief(goal, constraints)` assembles inventor-style briefs that map constraints to resonant coupling adaptations.
- **Genius** – `metaphor_palette(theme, audience)` returns metaphor sets and delivery notes tuned to poetic storytelling for the specified audience.
- **HealthCoach** – `habit_stack_planner(goal, schedule, blockers)` blends nutrition, stress, and sleep interventions from curated guidance tables to produce habit stacks with blocker mitigation.
- **FitnessCoach** – `microcycle_plan(goal, equipment, days)` generates structured weekly programming with equipment notes and recovery guidance.
- **LanguageTutor** – `dialogue_drill(topic, level, language)` produces conversational scripts, correction cues, and follow-up coaching suited to learner proficiency.
- **MathTutor** – `stepwise_solution(problem, focus)` enumerates numbered reasoning steps and performs symbolic verification when SymPy is available.
- **ComplianceOfficer** – `regulatory_gap_audit(domain, controls)` flags gaps against predefined checklists (GDPR, HIPAA, ISO 27001) to prioritize remediation.
- **KnowledgeCurator** – `knowledge_card_builder(query, sources, review_cadence=None)` constructs standardized cards including findings, citations, and follow-up prompts.

### Operator workflow

1. Resolve the correct persona for the request (for example, ATLAS for mission control reporting).
2. Call `create_task` with a descriptive `description`, `conversation_id`, and the manifest name in both the `title` and `metadata.manifest_task` fields. Include additional metadata such as affected teams or change ticket references when available.
3. Route the resulting task to the persona by setting `metadata.persona` to the target persona identifier (e.g., `"ATLAS"` or `"WeatherGenius"`). The routing layer will reconcile the manifest metadata with the persona's enabled skills and tools.
4. Monitor the acceptance criteria within the manifest to verify completion. The criteria are intentionally phrased as checklists so reviewers can record pass/fail outcomes.

Escalations are documented inside each manifest entry. When a condition in the `triggers` list occurs, page the listed contact within the specified `timeframe` and follow the recorded `actions`.

## Lifecycle states

The task domain model defines the following lifecycle states (`modules/task_store/models.py`):

- `draft` – task created but not yet queued for execution.
- `ready` – task queued and waiting for assignment.
- `in_progress` – task actively being worked on.
- `review` – task awaiting validation or escalation.
- `done` – task successfully completed.
- `cancelled` – task aborted before completion.

`modules/task_store/service.py` enforces the transition graph (`_ALLOWED_TRANSITIONS`) and emits `task.created`, `task.updated`, and `task.status_changed` events via the message bus. Invalid transitions raise `TaskTransitionError`, and dependency checks prevent advancing tasks with incomplete prerequisites.

## Service APIs

`TaskService` exposes `create_task`, `update_task`, and `transition_task`, delegating persistence to `TaskStoreRepository`. Each operation records lifecycle analytics using `modules.analytics.persona_metrics.record_task_lifecycle_event`, which persists metrics in `TaskLifecycleEvent` records and publishes `task_metrics.lifecycle` messages.

## Analytics and rollups

Aggregated task metrics—including success rates, latency, reassignment counts, and recent activity—are available through `get_task_lifecycle_metrics`. The results surface in the `CapabilityRegistry.summary` payload alongside [job lifecycle analytics](../jobs/lifecycle.md). Operators can subscribe to both `task_metrics.lifecycle` and `jobs.metrics.lifecycle` to build composite funnels that track persona throughput from individual tasks up to recurring jobs.

## UI and dashboards

The `CapabilityRegistry.summary` helper bundles tool, skill, task, and job catalogs (including compatibility flags and tool health) for persona-aware dashboards. The new `analytics` section combines task and job lifecycle metrics so UI layers can visualize job throughput, SLA adherence, and task success rollups in a single workspace. See [job dashboards and UI workflows](../jobs/ui.md) for implementation guidance.
