---
audience: Operators and DevOps
status: in_review
last_verified: 2025-12-21
last_updated_hint: Validated runbook links and expanded the ops landing list.
source_of_truth: Links to ops subpages
---

# Operations Runbooks

Use these guides to provision and maintain ATLAS environments. Pick an audience landing page first to confirm prerequisites and policy expectations:

- [Developer docs](../developer/README.md) — Environment setup, configuration references, and contributor guidance.
- [Enterprise docs](../enterprise/README.md) — Compliance checkpoints and operational guardrails for production teams.

Runbooks:

- [Developer environment setup](developer-setup.md) – Bootstrap the Python virtual environment, run the CLI setup helper, and decide when to rely on the GTK wizard for configuration.
- [Messaging bus deployment](messaging.md) – Configure and operate the asynchronous message bus, including Redis-backed deployments.
- [Background worker health](background-workers.md) – Size worker pools, monitor scheduler health, and tune queue depth for Redis or in-memory backends.
- [Manual QA preflight](manual-qa-preflight.md) – Validate GTK dependencies, Redis/PostgreSQL services, and wizard remediation flows before manual testing.
- [Legacy speech services](speech-services-legacy.md) – Historical reference for the deprecated speech stack; consult current audio docs for active development.
