# Persona definitions

The personas shipped with ATLAS live under `modules/Personas/<Persona Name>/Persona/<Persona Name>.json`.
Each file is validated against [`modules/Personas/schema.json`](../modules/Personas/schema.json), a
[JSON Schema](https://json-schema.org/) contract that guards the most important parts of every
persona definition.

## Schema overview

The schema enforces that each persona document:

- wraps persona entries in a top-level `persona` array (only the first entry is used today),
- provides a `name` and the `content.start_locked` / `content.editable_content` /
  `content.end_locked` fields, and
- lists any `allowed_tools` as canonical tool identifiers that exist in
  `modules/Tools/tool_maps/functions.json`.

Personas must explicitly enumerate their allowed tools. When `allowed_tools`
is missing or `null`, the loader treats the persona as having no enabled tools
even though the UI continues to display the global catalog for awareness. Add
the identifiers you want to enable to the list to opt-in.

When the persona loader runs it automatically injects the set of known tool identifiers into the
schema so the validator can confirm that `allowed_tools` only contains supported tools. Any
additional persona fields remain opt-in and are still passed through unchanged.

## Manifest-backed tasks per persona

Each production persona exposes a curated task manifest under `modules/Personas/<Persona>/Tasks/tasks.json`. Operators should populate the `metadata.manifest_task` field when creating a task so routing logic can hydrate the correct capabilities.

- **ATLAS** – Use `MissionControlWeeklyBrief` for the weekly leadership brief and `AutomationPolicyPrecheck` for governance reviews. ATLAS now has the full job toolkit (`browser`, `notebook`, `notification_service`, `dashboard_service`, `spreadsheet`, `atlas_dashboard`, `roadmap_service`, `ticketing_system`) so include sprint or change ticket identifiers in the task metadata to streamline follow-up.
- **WeatherGenius** – Trigger `WeatherOperationsSnapshot` when a field team needs real-time weather posture guidance. Attach region codes or GPS coordinates in the metadata payload.
- **MEDIC** – Trigger `ClinicalEvidenceSnapshot` to assemble bedside-ready evidence summaries. Record patient acuity and key symptoms to focus the literature search.
- **DocGenius** – Reuse `ClinicalEvidenceSnapshot` for documentation handoffs and add preferred citation styles in metadata when necessary.
- **KnowledgeCurator** – Dispatch `KnowledgeArchiveDigest` to transform ad-hoc updates into durable knowledge cards. Beyond `context_tracker`, `google_search`, and `get_current_info`, KnowledgeCurator now leans on shared job primitives like `browser`, `notebook`, `notification_service`, `dashboard_service`, and `spreadsheet` to capture, route, and broadcast archival updates.
- **ResumeGenius** – Trigger `AutomationPolicyPrecheck` before enabling a new resume-generation automation. ResumeGenius now has direct access to `browser`, `labor_market_feed`, `crm_service`, `analytics_dashboard`, `content_repository`, and `email_service` (alongside `notebook` and `notification_service`) so capture consent or privacy exceptions up front for governance review. The persona also allowlists the shared `Automation` skill so nurture campaigns can be orchestrated end-to-end inside the CRM/email stack with built-in safeguards.
- **ComplianceOfficer** – Partner with this persona whenever teams need policy interpretation, risk triage, or governance review. In addition to `policy_reference`, `google_search`, and `get_current_info`, ComplianceOfficer can now synthesize dashboards and spreadsheets while coordinating escalations through `browser`, `notification_service`, `atlas_dashboard`, and `roadmap_service`.

### KnowledgeCurator workflow

KnowledgeCurator acts as the team's archivist. Every request is normalized with `context_tracker` so the persona can see prior answers and link related artifacts. It validates external claims through `google_search`, timestamps deliverables with `get_current_info`, and then responds with a structured knowledge card that highlights the governing question, confirmed findings, citations, retention caveats, and open follow-ups. Mention owners or review cadences when the user supplies them so the archive stays actionable.

Personal-assistant personas expose two calendar toggles in their `type` entry:

- `access_to_calendar` grants read-only access to the Debian 12 calendar tool so the persona
  can enumerate events and inspect details.
- `calendar_write_enabled` builds on read access and allows the persona to create, update, or
  delete events. This flag is stored alongside the other persona settings and defaults to `False`
  so installs must opt in explicitly to calendar writes.

## Validating personas locally

Persona validation happens automatically whenever `modules.Personas.load_persona_definition` reads a
persona file, so most developer entry points (UI loading, tests, etc.) will fail fast if a persona is
invalid. To run a dedicated validation sweep locally, execute the schema test suite:

```bash
pytest tests/test_persona_schema.py
```

The positive test loads every persona in the repository and the negative test confirms that
misconfigured personas raise a descriptive `PersonaValidationError`.

## Updating personas

When you add or modify a persona:

1. Update the JSON definition under `modules/Personas/<Persona Name>/Persona/`.
2. Ensure any referenced tools exist in `modules/Tools/tool_maps/functions.json` (or add them there).
3. Confirm that every skill listed in `allowed_skills` has its `required_tools` also present in `allowed_tools`.
   Validation now fails fast with an error such as `Skill 'ResearchBrief' requires missing tools: google_search`
   whenever a dependency is omitted. The GTK editor mirrors this by highlighting the affected skills and
   prompting you to enable the missing tools before retrying the save.
4. When updating a persona toolbox (`modules/Personas/<Persona Name>/Toolbox/functions.json`), include the
   extended tool metadata required by `modules/Tools/tool_maps/schema.json`. All entries must specify an
   `idempotency_key`, and read-only tools should copy the capability tags, provider list, and cost hints from
   the shared tool manifest so the metadata stays consistent across personas.
5. Run `pytest tests/test_persona_schema.py` (or the full `pytest` suite) before opening a PR.

### Skill catalog overrides

Skill manifests (`modules/Skills/skills.json` and the optional persona-local
`modules/Personas/<Persona Name>/Skills/skills.json`) are merged into a shared
catalog. Skill names must remain unique within a single manifest file, but a
persona-specific manifest may intentionally reuse the name of a shared skill to
provide custom metadata or prompts for that persona. When a duplicate name is
present:

- The persona-specific entry is only enabled for that persona and is surfaced
  as a distinct catalog entry in the editor.
- The shared skill continues to be exposed to other personas so they can opt in
  without losing access to the default behavior.
- Personas that own an override still see the shared variant in the catalog, but
  it is marked read-only to indicate the override applies.

Each manifest entry may also provide descriptive metadata beyond the required
prompt and capability lists:

- `summary` &mdash; one-sentence description shown in the UI hover text.
- `category` &mdash; high-level grouping used for catalog filtering.
- `capability_tags` &mdash; free-form keywords that surface dependencies or
  downstream effects.

These fields remain optional, but shared skills in `modules/Skills/skills.json`
should supply them so personas inherit a consistent UX out of the box.

This separation lets authors layer persona-specific behavior on top of shared
skills without removing the shared version from the catalog.

### Collaborative response orchestration

Personas can opt into the multi-agent negotiation layer by adding a
`collaboration` block to their JSON definition. When enabled, ATLAS will gather
proposals from the listed participants and run one of the supported protocols
(`vote`, `critique`, or `contract_net`) before returning the final answer.

```json
{
  "persona": [
    {
      "name": "Planner",
      "content": { "start_locked": "", "editable_content": "", "end_locked": "" },
      "collaboration": {
        "enabled": true,
        "protocol": "vote",
        "quorum": 0.75,
        "timeout": 0.5,
        "participants": [
          { "id": "planner", "provider": "OpenAI", "model": "gpt-4o" },
          { "id": "reviewer", "provider": "Anthropic", "model": "claude-3" }
        ]
      }
    }
  ]
}
```

The quorum value represents the fraction of participants that must respond
before a proposal can be selected. Timeouts are expressed in seconds and guard
the per-participant call; when a participant exceeds the timeout they are marked
as timed out and the protocol either fails (when quorum cannot be met) or
continues with the available responses.

Skill manifests support the same structure. If a skill definition includes a
`collaboration` block the session will merge its settings with the persona's
defaults whenever that skill is active. Skill overrides can append additional
participants or tweak the protocol without modifying the persona document.

## Exporting and importing personas

Persona definitions can be exchanged as signed bundles so that the `allowed_tools`
list remains trustworthy. Two workflows are available:

### Command line tools

The `scripts/persona_tools.py` helper exposes `export` and `import` commands. The
commands require a signing key (either directly via `--signing-key` or from a
file with `--signing-key-file`). The examples below assume the repository root as
the working directory; pass `--app-root` when working in a temporary test tree.

```bash
# Export the "Atlas" persona to bundle.json using the key stored in signing.key
python scripts/persona_tools.py export Atlas bundle.json \
    --signing-key-file signing.key

# Import the bundle, persisting the persona definition and recording an audit rationale
python scripts/persona_tools.py import bundle.json \
    --signing-key-file signing.key \
    --rationale "Imported from staging"
```

The importer validates the payload against the persona schema and compares the
referenced tools against the current catalog. Any missing tools are removed and
reported as warnings.

### GTK UI

The persona manager window now includes buttons to export the persona currently
being edited and to import new personas. Both actions prompt for a signing key
and use the same validation logic as the CLI. The import option is available on
the persona list window and opens a file chooser dialog to select the bundle
before verification and persistence.
4. If the schema needs to change, edit `modules/Personas/schema.json` and keep these docs in sync.

## Persona analytics metrics

Persona tool usage is persisted under `modules/analytics/persona_metrics.json`. The
file stores an `events` array where each entry records:

- `persona`: the persona identifier at execution time,
- `tool`: the invoked tool name,
- `success`: a boolean flag indicating whether the call completed without
  raising an error,
- `latency_ms`: the measured execution time in milliseconds, and
- `timestamp`: an ISO 8601 timestamp (UTC) describing when the event finished.

Aggregated metrics such as totals, success rates, per-tool breakdowns, and the
most recent executions are computed on demand by the analytics helpers in
`modules/analytics/persona_metrics.py`. The GTK persona manager exposes these
metrics in an **Analytics** tab, including optional start/end filters for the
captured time window.

## Persona review policy

Production personas must be re-attested on a recurring schedule to ensure their
tool access remains appropriate. ATLAS tracks persona changes in the
`logs/persona_audit.jsonl` file and records review attestations in
`logs/persona_reviews.jsonl`. Each attestation captures the reviewer identity
and an expiration timestamp; once expired, the persona is considered overdue for
review.

A cron-friendly helper at `scripts/persona_review.py` scans persona history and
queues review tasks in `logs/persona_review_queue.json`. By default personas are
due every 90 days, but the script accepts an `--interval-days` override and a
`--dry-run` mode for reporting. The scheduler is safe to run multiple times and
only enqueues missing tasks.

The GTK persona manager surfaces the current review status at the top of the
settings window. Overdue personas display a banner and a **Mark Review
Complete** action that records a fresh attestation through the backend API. When
an attestation is submitted the pending queue entry is cleared and the banner is
updated with the next review date.
