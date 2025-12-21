---
audience: Persona authors and UI integrators
status: in_review
last_verified: 2025-12-21
source_of_truth: AtlasServer blackboard routes/streaming; SkillExecutionContext.blackboard
---

# Shared Blackboard Overview

The shared blackboard provides a lightweight collaboration surface where skills
can publish hypotheses, claims, and artifacts during a conversation or project.
Entries are grouped by scope so multiple skills – and even separate agents – can
reference the same evolving plan.

## Posting from skills

Skills receive a `SkillExecutionContext` instance during execution.  The context
now exposes a scoped `blackboard` client with convenience helpers:

```python
entry = context.blackboard.publish_hypothesis(
    "Validate assumptions",
    "Run the latest telemetry through the anomaly detector",
    tags=["telemetry", "ml"],
)
```

The client automatically resolves the appropriate scope using the active
conversation identifier.  If the persona metadata contains a `project_id`, the
entries are grouped under that project instead.

Each entry captures a title, rich content, optional author metadata, tags, and
timestamps.  Additional fields can be supplied through the generic
`publish`/`update_entry` helpers when a skill needs to persist custom metadata.

## Querying and coordination patterns

Skills can review existing contributions to avoid duplicate work:

```python
summary = context.blackboard.summary()
if summary["counts"]["artifact"]:
    artifacts = context.blackboard.list_entries(category="artifact")
    # decide whether another artifact is required
```

The GTK conversation UI now includes a **Blackboard** tab that displays the
latest summary so persona authors can monitor coordination in real time.  All
blackboard changes also emit message-bus events, enabling other processes to
subscribe via the `/blackboard/<scope>/<identifier>` WebSocket stream or the REST
endpoints exposed by `AtlasServer`.

## REST and streaming access

External agents can manage entries through the `AtlasServer` utility.  Every
route requires a tenant-scoped `RequestContext`; requests without one are
rejected to preserve tenant isolation.

- `GET /blackboard/<scope>/<id>` – list entries
- `GET /blackboard/<scope>/<id>?summary=1` – retrieve counts and summaries
- `POST /blackboard/<scope>/<id>` – create an entry
- `PATCH /blackboard/<scope>/<id>/<entry_id>` – update an entry
- `DELETE /blackboard/<scope>/<id>/<entry_id>` – delete an entry

For near-real-time updates, call `AtlasServer.stream_blackboard_events(scope, id,
context)` to receive an asynchronous iterator of change notifications scoped to
the requesting tenant.  These APIs allow
persona authors to wire up dashboards or additional automations without
re-implementing the storage plumbing.
