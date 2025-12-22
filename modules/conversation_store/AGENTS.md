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
