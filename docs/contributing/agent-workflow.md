---
audience: All ATLAS contributors and scoped agents
status: in_review
last_verified: 2026-02-26
source_of_truth: docs/contributing/agent-workflow.md; repository AGENTS.md files
---

# Shared Agent Workflow

This workflow keeps contributors aligned on intent, guardrails, and validation across the ATLAS codebase. Use it before writing code, updating docs, or proposing configuration changes.

## 1) Capture intent
- Identify the request, expected outputs, and success criteria (code changes, docs, tests, or handoff notes).
- Clarify scope boundaries early (affected components, in/out of scope behaviors, and follow-up items).

## 2) Discover guardrails
- Locate `AGENTS.md` files from the repository root to the target directory to gather scope, style, and required checks. Use directory-scoped files with the deepest path for the most specific rules.
- Check the [agent owner registry](./agent-owners.md) to confirm the responsible owner, cadence, and escalation contact for the audited subsystem before proposing changes.
- Confirm your role for the change (UI, Backend, Data/DB, Infra/Config, Docs, Testing, Security) to avoid cross-scope edits.

## 3) Align the design
- Draft a minimal plan: intended approach, touch-points (files, modules, schemas), and validation strategy.
- Check for related specs (schema files, config templates, UI behaviors) to keep changes consistent with established patterns.

## 4) Confirm execution constraints
- Follow the **non-overlap principle**: keep each change within a single agent scope whenever possible.
- Respect safety rules: avoid secrets, production configuration changes, and unreviewed migration paths.
- Honor repository conventions (naming, front matter for docs, existing manifest patterns) and any nested instructions in the relevant directories.

## 4a) Coordinate cross-cutting work
- Assign a **lead agent** for scope-spanning efforts and a **secondary agent** from each affected domain. The lead owns sequencing, scope boundaries, and review readiness; secondaries ensure domain-specific guardrails are applied.
- Define **interface freeze windows** for any shared contracts (APIs, schemas, events, UI surface areas) while dependent work lands. Communicate freeze start/end times in the PR description and do not merge interface changes during the freeze without lead + affected secondary approvals.
- Require approvals from the lead plus at least one secondary from every touched scope (e.g., Backend, UI, Docs) before merging.
- Bundle multi-scope changes only when they form a single, inseparable unit of value. Otherwise, stage changes by scope: land backend contracts behind feature flags, follow with UI wiring, then finalize docs. Each stage should be merge-safe and revertible.
- When bundling is unavoidable (tight coupling), use a **stacked branch strategy** or sequence of PRs: base branch for shared contracts, dependent branches for UI/backend/docs layering. Merge in order, with explicit dependency notes in each PR.
- For rollouts, prefer **feature flags or config gates** by default. Keep new paths backward-compatible, support configuration-driven rollbacks, and document default flag states. Remove flags only after the feature has run stably and dependencies confirm compatibility.

## 5) Validation checklist by domain
| Domain | Required validations |
| --- | --- |
| Docs (this directory and `/docs`) | Ensure every page includes YAML front matter and that new or changed links resolve. Preview Markdown locally if possible. When canonical references move, update `docs/_audit/inventory.md` and `docs/_audit/architecture-alignment-report.md`. |
| Backend (`ATLAS/`, `modules/`, `atlas_provider.py`, `main.py`) | Run persona validation and the full suite: `pytest tests/test_persona_schema.py` and `pytest`. Add targeted tests for new behaviors when applicable. |
| Server (`server/`) | Run route coverage: `pytest tests/server/test_conversation_server_routes.py tests/server/test_task_routes.py tests/server/test_job_routes.py`. For startup/config wiring, also run `pytest tests/test_setup_controller.py`. |
| Data/DB (`modules/conversation_store/` and related migrations) | Run persistence coverage: `pytest tests/test_conversation_store.py tests/test_conversation_retention.py tests/test_conversation_store_bootstrap_helper.py tests/test_sqlite_integration.py`. Include vector/graph suites when touched: `pytest tests/modules/conversation_store/test_shared_vectors.py tests/modules/conversation_store/test_module_splits.py`. Document migration/rollback plans before applying schema changes. |
| UI (`GTKUI/`, `Icons/`) | Perform GTK smoke verification with `python3 main.py` when feasible. Capture screenshots for visible changes and include them in the handoff. |
| Infrastructure/Config (`server/`, `config.yaml`, `ATLAS/config/`, `scripts/`) | Validate configuration flows with `pytest tests/test_setup_controller.py` and any route tests impacted by the change. Keep runtime defaults aligned with deployment expectations. |
| Testing-only changes (`tests/`) | Keep changes isolated to tests. Run the targeted suites you modify and the relevant domain suites to confirm coverage. |
| Security reviews | Ensure no secrets or sensitive endpoints are introduced. Confirm configuration hardening aligns with existing policies before merging. |

## 6) Traceability
- Record commands executed (tests, linting, previews) for inclusion in commit and PR notes.
- Keep changeset-focused commits with clear messages that describe the scope and validations performed.
- Reference relevant issues, specs, or schemas in commit/PR descriptions when they inform the implementation.

## 7) Handoff and review readiness
- Update `AGENTS.md` files when new guardrails or validation steps are introduced.
- Provide a concise summary of changes, validations, and any known follow-ups in the PR body.
- Attach artifacts as required: test logs, screenshots for UI updates, and migration/rollback notes for database changes.
