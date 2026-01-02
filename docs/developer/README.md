---
audience: Contributors building features, extending personas or tools, and maintaining local environments
status: in_review
last_verified: 2025-12-21
source_of_truth: README.md; scripts/install_environment.py; docs/architecture-overview.md
---

# Developer Docs Landing

**Audience:** Contributors building features, extending personas or tools, and maintaining local environments.

**Purpose:** Centralize setup, configuration, and architecture references so developers can provision ATLAS quickly and understand how components fit together before shipping changes.

**When to use this guide:** Use this entry point when you are cloning the repo, configuring providers, validating personas, or integrating new APIs. For production governance or policy review, pivot to the Enterprise docs.

## Quick links
- [Developer environment setup](../ops/developer-setup.md) — Virtualenv creation, dependency installation, and CLI/GTK bootstrap steps.
- [Configuration reference](../configuration.md) — Environment variables and YAML blocks consumed by the runtime.
- [Architecture overview](../architecture-overview.md) — System walkthrough and codebase map for new contributors.
- [Context management](context-management.md) — ExecutionContext and LLMContextManager for request-scoped and LLM context.
- [Persona guide](../Personas.md) — Schema, validation, and manifest expectations.
- [Tool manifest guide](../tool-manifest.md) — Metadata required for adding or updating tools.
- [AtlasServer API reference](../server/api.md) — REST endpoints for conversations, tasks, jobs, and tools.
