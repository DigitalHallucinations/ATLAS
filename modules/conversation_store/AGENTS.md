# Conversation Store Agent Guidelines (Data/DB)

## Scope
- Data/DB Agent owns persistence work in `modules/conversation_store/`, related models, repositories, and helpers.
- Coordinate with Backend Agent for orchestration changes and with Infra/Config Agent for configuration surfaces (e.g., DSN selection, pool settings).

## Required checks before merge
- Run targeted persistence tests: `pytest tests/test_conversation_store.py tests/test_conversation_retention.py`.
- Run bootstrap/adapter coverage: `pytest tests/test_conversation_store_bootstrap_helper.py tests/test_sqlite_integration.py`.
- When vector or graph functionality changes, include `pytest tests/modules/conversation_store/test_shared_vectors.py tests/modules/conversation_store/test_module_splits.py`.
- If changes impact user accounts or routes that rely on the store, run the broader suite: `pytest`.

## Coordination
- Follow the **non-overlap principle**: keep persistence edits confined to this scope. For cross-cutting backend + data changes, split commits by scope and document which tests were run for each area.
- Consult the shared workflow in [`../../docs/contributing/agent-workflow.md`](../../docs/contributing/agent-workflow.md) to capture intent, confirm guardrails, align design, respect execution constraints, run required validations (including migration notes), and prepare handoffs.

## Migration workflow and safety
- Order schema changes to minimize locking: additive columns first, backfills next, then indexes and constraints. Avoid destructive DDL in the same deployment window as backfills unless explicitly coordinated.
- Keep `upgrade`/`downgrade` pairs in [`../../scripts/migrations/conversation_store.py`](../../scripts/migrations/conversation_store.py) monotonic and idempotent; record the intended sequence (including pre/post hooks) in PR notes when adding steps.
- Define rollback paths alongside every migration: specify data preservation steps, downgrade coverage, and any follow-up cleanup required when toggling features off.
- Write backfill plans before merging: chunked/tenant-aware updates, retry strategy, throttling, and validation queries that confirm parity before and after the backfill.
- Run performance checks for new DDL: gather index creation cost estimates, confirm query plans on hot paths, and verify that pagination and retention queries keep using the expected composite indexes.

## Change checklist (indexes, pagination, retention)
- Index changes: document target predicates and column order, validate with `EXPLAIN` on representative datasets, and ensure multi-tenant keys stay leading for all hot paths.
- Pagination: enforce deterministic ordering keys, maintain bounded default limits, and confirm cursor/offset semantics match existing API contracts.
- Retention: call out TTL/cleanup implications, coordinate with background workers, and ensure migrations do not re-inflate expired records.
- Record schema touchpoints and last audited migration IDs in [`../../docs/_audit/inventory.md`](../../docs/_audit/inventory.md) when altering tables, indexes, or retention-related fields.
