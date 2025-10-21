# Conversation store retention

The conversation store supports automated cleanup of messages and conversations
through a combination of configuration-driven policies and background workers.
The repository reads retention settings from the configuration block passed to
`ConversationStoreRepository` during construction. The following keys are
recognized:

| Key | Description |
| --- | --- |
| `message_retention_days` | Permanently delete any message older than the configured number of days. When omitted the legacy `days` key is treated as an alias. |
| `soft_delete_after_days` | Soft-delete (set `deleted_at` and mark the status as `deleted`) for active messages older than this age. |
| `soft_delete_grace_days` | Permanently remove soft-deleted messages once they have been in the deleted state for at least this many days. |
| `conversation_archive_days` | Archive conversations that have been inactive for the specified number of days. |
| `archived_conversation_retention_days` | Hard-delete conversations that have been archived longer than this window. |
| `tenant_limits` | Mapping of tenant identifiers to per-tenant policies. Currently the worker honours `max_conversations`, archiving the oldest conversations when the tenant exceeds the configured number of active conversations. |

All durations are interpreted as whole days. Values that cannot be coerced to an
integer are ignored.

## Background worker

`modules.background_tasks.retention.RetentionWorker` provides a lightweight
threaded scheduler that repeatedly calls `ConversationStoreRepository.run_retention`
at a fixed interval (default: one hour). The worker can also be triggered
manually through `run_once()` to execute an immediate retention pass.

## Administrative trigger

Administrative tooling can invoke retention on demand through
`AtlasServer.run_conversation_retention(context=...)`. The supplied request
context must include an `admin` or `system` role; otherwise a
`ConversationAuthorizationError` is raised. The method returns the aggregated
retention statistics reported by the repository.
