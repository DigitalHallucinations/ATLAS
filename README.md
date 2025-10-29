# ATLAS

## Overview
ATLAS combines a GTK desktop shell, configurable personas, and an orchestration back end to coordinate multi-agent work across conversations, scheduled jobs, and automation services. The core application wires configuration, provider management, persona loading, conversation storage, task orchestration, and job scheduling into a single runtime that tools and user interfaces share.【F:ATLAS/ATLAS.py†L42-L115】

## Major subsystems
- **GTK desktop shell** – `main.py` boots a GTK 4 application that initializes ATLAS instances on demand, launches the first-run coordinator, and keeps setup, shell, and window controllers alive for the duration of the session.【F:main.py†L18-L55】
- **Persona runtime** – Persona definitions, toolboxes, and validation rules live under `modules/Personas/` and are documented in the persona guide. They control which tools, skills, and collaboration patterns each persona can access, and include task manifests for persona-specific workflows.【F:docs/Personas.md†L1-L105】
- **Orchestration back end** – The orchestration layer manages message-bus communication, task dispatch, job planning, and capability registry services that feed both automation APIs and UI analytics.【F:modules/orchestration/message_bus.py†L1-L118】【F:modules/orchestration/job_manager.py†L1-L92】

## High-level architecture
At startup the application configures message-bus backends, speech services, persona and provider managers, and the PostgreSQL-backed conversation repository via the central `ConfigManager`. The orchestration stack layers task and job managers on top of that state, while the embedded `AtlasServer` exposes REST routes for conversations, tasks, jobs, tools, skills, and collaboration surfaces. This shared infrastructure lets the GTK shell, automation jobs, and external callers operate against the same message bus, storage, and capability registries.【F:ATLAS/ATLAS.py†L51-L114】【F:modules/orchestration/message_bus.py†L1-L118】【F:modules/Server/routes.py†L92-L157】

## Runtime prerequisites
- **Python 3.10 or newer** – The codebase uses Python 3.10 union type syntax (for example `ATLAS | None`), so run the environment bootstrap with a modern `python3` interpreter.【F:main.py†L22-L34】【F:docs/setup-wizard.md†L45-L54】
- **PostgreSQL 14+** – Conversation history, key-value state, and scheduling primitives are all backed by PostgreSQL. Setup helpers verify the server, install client utilities when needed, and refuse to start without a PostgreSQL DSN.【F:docs/release-notes.md†L21-L23】【F:modules/conversation_store/bootstrap.py†L242-L312】【F:docs/tools/task_queue.md†L1-L37】【F:docs/tools/kv_store.md†L19-L39】
- **Redis (optional)** – Redis Streams provide a durable message-bus backend for production deployments; in-memory queues remain available for local development and as a fallback when Redis is absent.【F:docs/ops/messaging.md†L3-L41】
- Install Python dependencies by running the provided helper script inside your virtual environment.【F:docs/setup-wizard.md†L45-L54】

## Launching the desktop shell and automation APIs
After completing setup, start the GTK shell from the repository root with:

```bash
python3 main.py
```

The application will initialize the ATLAS runtime and present the primary window or, when configuration is missing, guide you through the setup wizard.【F:main.py†L18-L55】【F:docs/setup-wizard.md†L56-L70】 Server and automation surfaces live in `modules/Server/`, where `AtlasServer` wires REST and streaming routes for conversations, tasks, jobs, tools, skills, and shared blackboard collaboration. Tool discovery endpoints (for example `/tools`) and capability registries feed downstream automations and dashboards.【F:modules/Server/routes.py†L92-L157】【F:docs/tool-manifest.md†L6-L18】【F:docs/blackboard.md†L40-L58】

## Documentation map
- [Setup wizard](docs/setup-wizard.md) – Guided configuration flow, CLI helper, and environment bootstrap instructions.【F:docs/setup-wizard.md†L1-L81】
- [Persona definitions](docs/Personas.md) – Schema, validation workflow, and persona-specific tooling guidance.【F:docs/Personas.md†L1-L136】
- [Task lifecycle overview](docs/tasks/overview.md) – Task manifests, routing, analytics, and UI integration details.【F:docs/tasks/overview.md†L1-L104】
- [Job services](docs/jobs/api.md) and [job dashboards](docs/jobs/ui.md) – API entry points, lifecycle expectations, and UI analytics guidance.【F:docs/jobs/api.md†L1-L19】【F:docs/jobs/ui.md†L1-L16】
- [Tool manifest metadata](docs/tool-manifest.md) and the [generated tool catalog](docs/generated/tools.md) – Schema, discovery endpoints, and persona-scoped tool inventories.【F:docs/tool-manifest.md†L1-L156】【F:docs/generated/tools.md†L1-L61】
- [Task queue](docs/tools/task_queue.md) and [key-value store](docs/tools/kv_store.md) tools – PostgreSQL-backed automation primitives with configuration and deployment guidance.【F:docs/tools/task_queue.md†L1-L77】【F:docs/tools/kv_store.md†L1-L70】
- [Conversation retention](docs/conversation_retention.md) – Policy knobs and background workers that manage store retention windows.【F:docs/conversation_retention.md†L1-L34】
- [Shared blackboard](docs/blackboard.md) – Collaboration surface for skills and external agents with REST and streaming APIs.【F:docs/blackboard.md†L1-L58】
