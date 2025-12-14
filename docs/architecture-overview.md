# ATLAS Architecture & Codebase Tour

This guide expands on the README with a deeper walkthrough of the runtime, files, and development workflows. It is aimed at new contributors who want an end-to-end picture of how the GTK shell, persona runtime, orchestration layer, and server fit together.

## Entry Points & Configuration Flow
- **`main.py`** starts a `Gtk.Application`, runs the first-run coordinator, and lazily instantiates the core `ATLAS` runtime only after setup passes the `is_setup_complete()` check.
- **`ATLAS/ATLAS.py`** constructs the application runtime: it loads configuration via `ConfigManager`, establishes the message bus, initializes speech, personas, providers, chat session plumbing, and creates the `AtlasServer` plus repositories/services used by the UI and APIs.
- **Configuration sources** come from `config.yaml` plus environment variables (see `docs/configuration.md`). `ConfigManager` also verifies the PostgreSQL conversation store before the runtime proceeds.

## Runtime Composition
- **Messaging**: `ConfigManager.configure_message_bus()` wires Redis streams or in-memory queues depending on the environment. Tool events can still flow through the legacy `modules/Tools/tool_event_system` adapters.
- **Conversation store**: `modules/conversation_store/` holds the SQLAlchemy models and `ConversationStoreRepository` that enforces retention policies and backs conversations, tasks, and job metadata.
- **Speech**: `modules/Speech_Services/` contains the `SpeechManager` plus TTS/STT integrations. The runtime exposes a `SpeechService` facade so the UI and APIs can request status or streaming updates.
- **Personas & providers**: persona manifests and schemas live under `modules/Personas/`. The `PersonaManager` loads persona definitions, while `ProviderManager` resolves which LLM provider/model is active and dispatches tool calls through the shared tooling service.
- **User accounts**: `modules/user_accounts/` adds login and lockout flows. `UserAccountFacade` links auth state to the conversation repository so multi-tenant data stays isolated via `tenant_id`.

## Orchestration & Automation
- **Tasks & jobs**: The `modules/orchestration/` package hosts the `TaskManager`, `JobManager`, `JobScheduler`, capability registry, and supporting blackboard for collaborative state. These services share the message bus and repository state initialized by the core runtime.
- **Background tasks**: Utilities in `modules/background_tasks.py` run async work in threads so long-running orchestration can coexist with the GTK event loop and API handlers.
- **Tooling**: `ATLAS/services/tooling.py` wraps the `ToolManager` module to validate tool usage against persona manifests and emit events that the orchestration layer can observe.

## Server & Interfaces
- **AtlasServer**: Defined under `modules/Server/`, the server exposes REST and streaming endpoints for conversations, tasks, jobs, tools, skills, and blackboard operations. The GTK shell and external clients both call into this API surface.
- **GTK UI**: The `GTKUI/` package contains the desktop shell. `GTKUI/Setup/first_run.py` coordinates the setup wizard, while `GTKUI/sidebar.py` hosts the main window and routes UI actions to the runtime through injected factories.

## Personas in Practice
- **Layout**: Each persona resides in `modules/Personas/<Name>/Persona/` with JSON manifest files, prompts, and optional tools/skills manifests.
- **Schema**: The repository ships a canonical schema at `modules/Personas/schema.json`; run `pytest tests/test_persona_schema.py` to validate new personas.
- **Access control**: Persona manifests enumerate allowed tools/skills and can toggle collaboration protocols, which the tooling service enforces during request execution.

## Data & Persistence
- **Database**: PostgreSQL backs conversations, tasks, jobs, and retention. The runtime will abort startup if `ConfigManager.is_conversation_store_verified()` fails.
- **Audit & logging**: `modules/logging/` provides the structured logger and persona-aware audit log hooks so automation and UI actions share consistent telemetry.
- **Key-value helpers**: Support utilities in `docs/tools/kv_store.md` and the queue helpers in `docs/tools/task_queue.md` describe the supporting data-plane utilities used by orchestration services.

## Development & Testing
- **Environment setup**: Follow `docs/ops/developer-setup.md` or run `python3 scripts/install_environment.py --with-accelerators` to provision the virtualenv and optional GPU extras.
- **Running the app**: Start the desktop shell with `python3 main.py`. The first-run wizard will prompt for required configuration before the runtime is created.
- **Tests**: Execute `pytest` for the full suite and `pytest tests/test_persona_schema.py` for fast persona validation. Conversation-store verification and background worker checks run automatically during runtime construction, so failures there usually indicate configuration gaps.
- **Code search**: Use `rg` for repository searches (avoid `grep -R` in this codebase) and rely on the documentation map in `README.md` for topic-specific docs.

## How to Explore Next
- Trace a conversation lifecycle by following `ChatSession` in `modules/Chat/chat_session.py` and its integration with the `ConversationService` facade.
- Inspect the `modules/Server/routes/` definitions to see how REST handlers map to orchestration functions.
- Review persona manifests and skill/tool metadata under `modules/Personas/` to understand how capabilities are granted.
- Look at the job dashboard docs (`docs/jobs/ui.md`) and API reference (`docs/jobs/api.md`) to connect orchestration data to UI analytics.
