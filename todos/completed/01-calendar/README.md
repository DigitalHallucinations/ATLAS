# Calendar Services

> **Status**: ✅ Complete  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 1-2 weeks  
> **Created**: 2026-01-07  
> **Completed**: 2026-01-08  
> **Tests**: 230 passing

---

## Overview

Refactor the calendar system into three focused services:

1. **CalendarEventService** - Event and category CRUD, validation, queries
2. **CalendarSyncService** - External calendar synchronization  
3. **ReminderService** - Reminder scheduling and delivery

These services will:

- Move business logic from GTK UI to backend
- Provide unified API for UI, agents, tools, and jobs
- Enable cross-system integration (jobs ↔ tasks ↔ calendar)
- Support agent calendar awareness with explicit permissions
- Replace `modules/Tools/Base_Tools/debian12_calendar.py` backends

---

## Architectural Decisions

| Decision | Choice | Rationale |
| -------- | ------ | ---------- |
| Service structure | Split into 3 services | Narrower responsibilities, easier testing |
| Reminder handling | Separate ReminderService | Independent scheduling concerns |
| Timezone conversion | Service layer | Repository stores UTC; service converts |
| Agent permissions | Explicit required | Security-first approach |
| Event notifications | MessageBus (comprehensive) | Cross-service communication |
| Legacy backends | Replace | Unified PostgreSQL-backed approach |

---

## Phases

### Phase 1: CalendarEventService Foundation ✅

- [x] **1.1** Create `core/services/calendar/` package structure
- [x] **1.2** Define protocols and types
- [x] **1.3** Implement `CalendarPermissionChecker`
- [x] **1.4** Implement `CalendarEventService`
- [x] **1.5** Add validation logic (moved from UI)
- [x] **1.6** Export from `core/services/__init__.py`
- [x] **1.7** Write unit tests (68 tests passing)

### Phase 2: Move Business Logic from UI ✅

- [x] **2.1** Identify all direct repository access in GTKUI/Calendar_manager/
- [x] **2.2** Move validation logic from UI to service  
- [x] **2.3** Update UI components to use CalendarEventService
- [x] **2.4** Remove direct repository imports from UI files
- [x] **2.5** Subscribe UI to MessageBus events for reactive updates (async patterns added)

### Phase 3: ReminderService ✅

- [x] **3.1** Create `core/services/calendar/reminder_service.py`
- [x] **3.2** Define reminder types
- [x] **3.3** Implement ReminderService
- [x] **3.4** Implement background reminder scheduler
- [x] **3.5** Connect to notification system (DesktopNotificationService with notify-send fallback)
- [x] **3.6** Write unit tests (9 tests passing)

### Phase 4: Job/Task Integration ✅

- [x] **4.1** Add event linking tables to calendar schema
- [x] **4.2** Add linking methods to CalendarEventService
- [x] **4.3** Add job/task-to-event creation
- [x] **4.4** Subscribe to job/task MessageBus events
- [x] **4.5** Write integration tests (23 tests passing)

### Phase 5: CalendarSyncService ✅

- [x] **5.1** Create `core/services/calendar/sync_service.py`
- [x] **5.2** Define sync types
- [x] **5.3** Implement CalendarSyncService
- [x] **5.4** Move SyncEngine orchestration from UI (via CalendarSyncService wrapper)
- [x] **5.5** Emit comprehensive MessageBus events
- [x] **5.6** Update SyncStatusPanel to use service
- [x] **5.7** Write unit and integration tests (29 tests passing)

### Phase 6: Agent Calendar Integration ✅

- [x] **6.1** Add agent schedule methods to CalendarEventService
  - `search_events()` - Full-text search with PostgreSQL FTS
  - `get_upcoming_events()` - Get next N events within hours
  - `check_availability()` - Check if time slot is free
  - `find_free_time()` - Find available time slots
  - `get_calendar_summary()` - Get period summary (today/week/month)
  - `suggest_meeting_times()` - Suggest optimal meeting times
- [x] **6.2** Add scheduling assistance
  - Conflict detection integrated into availability check
  - Working hours and weekend exclusion support
- [x] **6.3** Create CalendarServiceProvider in `modules/Tools/providers/`
  - Backward-compatible with debian12_calendar operations
  - New agent operations: upcoming, availability, find_free_time, summary, suggest_times
  - Registered as "calendar_service" provider
- [x] **6.4** Add agent context awareness
  - Created `core/services/calendar/context.py`
  - CalendarContextInjector for LLM prompt injection
  - Format helpers for natural language responses
- [x] **6.5** Configure agent permissions in `config.yaml`
  - Added `calendar.agent_access` section with read/write permissions
  - Context injection settings
  - Scheduling assistance defaults
- [x] **6.6** Write Phase 6 tests (39 tests passing)

### Phase 7: UI Updates for Integration ✅

- [x] **7.1** Update EventDialog to show linked jobs/tasks
- [x] **7.2** Add "Create from Job" / "Create from Task" quick actions
- [x] **7.3** Show job/task status on calendar events
- [x] **7.4** Add calendar view filters
- [x] **7.5** Add mini-calendar widget to Job/Task detail views
- [x] **7.6** Subscribe all UI components to MessageBus

---

## MessageBus Events

| Event Type | Payload | Emitted By |
| ---------- | ------- | ---------- |
| `calendar.event.created` | `CalendarEventChanged` | CalendarEventService |
| `calendar.event.updated` | `CalendarEventChanged` | CalendarEventService |
| `calendar.event.deleted` | `CalendarEventChanged` | CalendarEventService |
| `calendar.event.linked` | `CalendarEventLinked` | CalendarSyncService |
| `calendar.sync.started` | `CalendarSyncStatus` | CalendarSyncService |
| `calendar.sync.completed` | `CalendarSyncStatus` | CalendarSyncService |
| `calendar.sync.conflict` | `CalendarSyncConflict` | CalendarSyncService |
| `calendar.reminder.scheduled` | `ReminderEvent` | ReminderService |
| `calendar.reminder.triggered` | `ReminderEvent` | ReminderService |

---

## Files to Create

| File | Purpose | Status |
| ---- | ------- | ------ |
| `core/services/calendar/__init__.py` | Package exports | ✅ Done |
| `core/services/calendar/types.py` | Protocols, dataclasses, result types | ✅ Done |
| `core/services/calendar/permissions.py` | CalendarPermissionChecker | ✅ Done |
| `core/services/calendar/validation.py` | Event validation | ✅ Done |
| `core/services/calendar/event_service.py` | CalendarEventService | ✅ Done |
| `core/services/calendar/reminder_service.py` | ReminderService | ✅ Done |
| `core/services/calendar/events.py` | Domain events | ✅ Done |
| `core/services/calendar/context.py` | Agent context injection utilities | ✅ Done |
| `core/services/notifications/__init__.py` | Notification service package | ✅ Done |
| `core/services/notifications/types.py` | Notification types (enhanced) | ✅ Done |
| `core/services/notifications/notification_service.py` | Desktop/fallback notifications | ✅ Done |
| `core/services/notifications/history.py` | Notification history tracking | ✅ Done |
| `core/services/notifications/scheduler.py` | Smart scheduling & DND | ✅ Done |
| `core/services/notifications/actions.py` | DBus action button handling | ✅ Done |
| `core/services/notifications/email_service.py` | Email notifications (stub) | ✅ Done |
| `core/services/notifications/mobile_push.py` | Mobile push NTFY/Pushover (stub) | ✅ Done |
| `core/services/calendar/sync_service.py` | CalendarSyncService | ✅ Done |
| `modules/calendar_store/link_models.py` | Job/task link tables | ✅ Done |
| `core/services/calendar/job_task_integration.py` | MessageBus handler for jobs/tasks | ✅ Done |
| `modules/Tools/providers/calendar_service.py` | CalendarServiceProvider for agents | ✅ Done |
| `tests/calendar/test_sync_service.py` | Sync service tests | ✅ Done (29 tests) |
| `tests/calendar/test_agent_integration.py` | Agent calendar integration tests | ✅ Done (39 tests) |
| `tests/calendar/` | Service tests | ✅ Done (230 tests) |

---

## Notification System Features

The notification service now includes:

| Feature | Status | Description |
| ------- | ------ | ----------- |
| Desktop notifications (GNotification) | ✅ | Native GTK notifications |
| notify-send fallback | ✅ | CLI fallback for headless environments |
| Sound alerts | ✅ | Via paplay/aplay/play |
| Text-to-speech | ✅ | Via espeak-ng/espeak/spd-say |
| Action buttons | ✅ | Snooze/Dismiss/Open via DBus |
| Notification history | ✅ | Track sent/missed notifications |
| Do Not Disturb | ✅ | Suppress non-critical notifications |
| Quiet hours | ✅ | Scheduled DND with per-day rules |
| Focus mode | ✅ | Temporary DND with duration |
| Smart scheduling | ✅ | Queue for delivery window, critical bypass |
| Rate limiting | ✅ | Prevent notification spam |
| Email notifications | ⏳ | Stub ready for email service integration |
| Mobile push (NTFY/Pushover/Gotify) | ⏳ | Stub ready for implementation |

---

## Files to Modify

| File | Changes |
| ---- | ------- |
| `core/services/__init__.py` | Export calendar services |
| `core/ATLAS.py` | Add service properties |
| `config.yaml` | Add agent calendar permissions |
| `GTKUI/Calendar_manager/*.py` | Use services instead of direct repo access |
| `modules/calendar_store/schema.py` | Add link tables |

---

## Files to Delete

| File                                            | Reason                             |
| ----------------------------------------------- | ---------------------------------- |
| `modules/Tools/Base_Tools/debian12_calendar.py` | Replaced by CalendarEventService   |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- `modules/calendar_store/` - Repository layer (exists)
- `modules/job_store/` - Job integration target
- `modules/task_store/` - Task integration target
- `core/messaging/` - MessageBus for events
- `modules/background_tasks/` - Reminder scheduling

---

## Success Criteria

1. Zero business logic in GTKUI Calendar_manager files
2. Three focused services with clear boundaries
3. Typed events, results, and protocols throughout
4. All operations require explicit actor + permission
5. UTC storage, user timezone presentation
6. Jobs and tasks can create/link calendar events
7. Agents can query and modify calendar (with permissions)
8. UI updates via MessageBus, not polling
9. >90% test coverage on all services
10. Event queries <100ms for 1000 events

---

## Open Questions

| Question | Options | Decision |
| -------- | ------- | -------- |
| Should reminders support custom notification channels? | System only / Pluggable channels | TBD |
| How to handle sync conflicts with job-linked events? | Auto-resolve / User choice / Job priority | TBD |
| Should agent calendar writes require user approval? | Always / Configurable / Never | TBD |
| Support for multi-day event spanning? | Full support / Split into days | TBD |

---

## Future Considerations

1. Multi-tenant calendar sharing
2. Calendar event versioning
3. External calendar write-back (bidirectional sync)
4. Calendar delegation
