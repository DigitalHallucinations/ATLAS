---
audience: Operators and backend developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/background_tasks/retention.py; modules/background_tasks/conversation_summary.py; modules/Tools/Base_Tools/memory_episodic.py; modules/conversation_store/conversations.py; modules/orchestration/followups.py
---

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

### Conversation summaries and episodic memories

`ConversationSummaryWorker` listens for conversation message events (or polls the
store when no bus is configured), batches activity per conversation, and stores a
structured snapshot through `EpisodicMemoryTool`.【F:modules/background_tasks/conversation_summary.py†L61-L206】【F:modules/background_tasks/conversation_summary.py†L267-L314】【F:modules/Tools/Base_Tools/memory_episodic.py†L74-L107】 The
resulting episodic records land in `append_episodic_memory` and are queryable via
the existing episodic memory APIs, so long-term conversation context appears next
to manually captured memories when issuing `memory_episodic_query` requests.【F:modules/conversation_store/conversations.py†L772-L838】【F:modules/Tools/Base_Tools/memory_episodic.py†L108-L137】 Each summary includes
metadata describing the batch window and participants, enabling downstream tools
to filter or expire snapshots independently of the raw chat history.【F:modules/background_tasks/conversation_summary.py†L267-L314】 Operators can control the cadence, batch
size, and retention window through the `conversation_summary` configuration block
documented in [configuration.md](configuration.md).【F:ATLAS/config/conversation_summary.py†L9-L96】 Once enabled, summaries become
visible in memory queries alongside other episodic entries, providing a concise
timeline that survives message-level retention policies.

When follow-up templates are configured the worker also evaluates each snapshot
for outstanding questions or keywords, publishing structured
`conversation.followups` events whenever actionable items are detected.【F:modules/background_tasks/conversation_summary.py†L225-L356】 These
events feed the orchestration layer where `FollowUpOrchestrator` can launch
tasks or trigger escalations automatically based on the detected items.【F:modules/orchestration/followups.py†L12-L185】

## Administrative trigger

Administrative tooling can invoke retention on demand through
`AtlasServer.run_conversation_retention(context=...)`. The supplied request
context must include an `admin` or `system` role; otherwise a
`ConversationAuthorizationError` is raised. The method returns the aggregated
retention statistics reported by the repository.

## Desktop trigger

The GTK conversation history page now exposes a **Run retention** button for
administrators. ATLAS evaluates whether the current user has an `admin` or
`system` role (configured through the active user's role list) and whether the
server facade is available before enabling the control. When roles are missing
the button remains disabled and the UI presents an explanatory tooltip that the
feature requires administrative privileges. Attempts to invoke retention without
the necessary role return a descriptive error so operators immediately know why
the action was blocked.
