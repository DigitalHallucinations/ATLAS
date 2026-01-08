# Calendar Service Issues

> **Epic**: Modularize Calendar System
> **Parent**: [README.md](./README.md)
> **Priorities**: High

## ✅ Completed

### CAL-001: Create Calendar Services Package ✅

**Description**: Initialize the `core/services/calendar` structure.
**Acceptance Criteria**:

- ✅ Package `core/services/calendar/` created.
- ✅ `types.py` defining `Event`, `Calendar`, `Reminder`.
- ✅ `events.py` defining `CalendarEventCreated` etc.

### CAL-002: Implement CalendarEventService ✅

**Description**: Core CRUD for calendar events, decoupled from GTK.
**Acceptance Criteria**:

- ✅ `create_event`, `update_event`, `delete_event`, `list_events`.
- ✅ Validation logic (start < end, valid timezone).
- ✅ Returns `OperationResult`.

### CAL-003: Implement ReminderService ✅

**Description**: Separate service for handling reminders and alerts.
**Acceptance Criteria**:

- ✅ `schedule_reminder(actor, event, minutes_before)`.
- ✅ `process_reminders()` (background task method).
- ✅ Integration with MessageBus (`ReminderScheduled`, `ReminderTriggered`, `ReminderDelivered`).
- ✅ DesktopNotificationService with notify-send fallback.
- ✅ 9 unit tests passing.

## 📋 Ready for Development

### CAL-004: Implement CalendarSyncService (Stub)

**Description**: Scaffold the sync service (implementation can follow).
**Acceptance Criteria**:

- Interface for `sync_external(source)`.
- Data model for `SyncState`.

### CAL-005: Migrate `debian12_calendar.py`

**Description**: Replace the old tool backend with the new service.
**Acceptance Criteria**:

- `modules/Tools/Base_Tools/debian12_calendar.py` modified to use `core.services.calendar`.
