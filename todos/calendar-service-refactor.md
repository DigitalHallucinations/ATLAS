# Calendar Service Refactoring & Integration

> **Status**: Planning  
> **Priority**: High  
> **Target**: Centralized calendar service with job/task/agent integration  
> **Created**: 2026-01-06

---

## Overview

Refactor the calendar system to introduce a proper service layer (`CalendarService`) that:
1. Moves business logic from GTK UI to backend
2. Provides unified API for UI, agents, tools, and jobs
3. Enables cross-system integration (jobs ↔ tasks ↔ calendar)
4. Supports agent calendar awareness and scheduling

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CalendarService                              │
│                    (core/services/calendar.py)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Event Operations:           │  Integration Operations:             │
│  • create_event()            │  • link_event_to_job()               │
│  • update_event()            │  • link_event_to_task()              │
│  • delete_event()            │  • create_event_from_job()           │
│  • get_events_in_range()     │  • create_event_from_task()          │
│  • search_events()           │  • get_events_for_job()              │
│                              │  • get_events_for_task()             │
├──────────────────────────────┼──────────────────────────────────────┤
│  Category Operations:        │  Agent Operations:                   │
│  • list_categories()         │  • get_agent_schedule()              │
│  • create_category()         │  • check_availability()              │
│  • update_category()         │  • block_time_for_agent()            │
│  • delete_category()         │  • get_busy_slots()                  │
│                              │  • suggest_meeting_times()           │
├──────────────────────────────┼──────────────────────────────────────┤
│  Sync Operations:            │  Reminder Operations:                │
│  • sync_external_source()    │  • schedule_reminders()              │
│  • get_sync_status()         │  • get_pending_reminders()           │
│  • configure_sync_mapping()  │  • dismiss_reminder()                │
│  • resolve_conflict()        │  • snooze_reminder()                 │
└──────────────────────────────┴──────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────────┐
         ▼                          ▼                              ▼
┌─────────────────┐      ┌───────────────────┐      ┌──────────────────┐
│ CalendarStore   │      │    SyncEngine     │      │ ReminderScheduler│
│   Repository    │      │   + Providers     │      │ (background task)│
└─────────────────┘      └───────────────────┘      └──────────────────┘
         │                          │                              │
         └──────────────────────────┴──────────────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  StorageManager   │
                          │   (PostgreSQL)    │
                          └───────────────────┘
```

---

## Implementation Phases

### Phase 1: Create CalendarService Foundation

- [ ] **1.1** Create `core/services/calendar.py` with base `CalendarService` class
- [ ] **1.2** Add constructor accepting repository, logger, tenant_id (like ConversationService)
- [ ] **1.3** Implement event CRUD methods delegating to repository
- [ ] **1.4** Implement category CRUD methods delegating to repository
- [ ] **1.5** Add validation logic (moved from UI):
  - [ ] Event time validation (end > start)
  - [ ] Category slug uniqueness
  - [ ] Required field validation
  - [ ] Recurrence rule validation
- [ ] **1.6** Add listener/notification pattern for UI updates (like ConversationService)
- [ ] **1.7** Export from `core/services/__init__.py`
- [ ] **1.8** Write unit tests for CalendarService

### Phase 2: Move Business Logic from UI to Service

- [ ] **2.1** Identify all direct repository access in GTKUI/Calendar_manager/
  - [ ] `event_dialog.py` - `_load_categories()`, `_get_session_factory()`
  - [ ] `category_panel.py` - category CRUD operations
  - [ ] `sync_status.py` - CalendarProviderRegistry access
  - [ ] `calendar_view_stack.py` - event loading
  - [ ] `calendar_month_view.py` - event queries
  - [ ] `calendar_week_view.py` - event queries
  - [ ] `calendar_day_view.py` - event queries
  - [ ] `calendar_agenda_view.py` - event queries

- [ ] **2.2** Move validation logic from UI to service:
  - [ ] `event_dialog.py#_collect_form_data()` - time validation
  - [ ] `event_dialog.py#_build_rrule()` - recurrence building
  - [ ] `category_dialog.py` - category validation

- [ ] **2.3** Update UI components to use CalendarService:
  - [ ] Add `calendar_service` property to ATLAS class
  - [ ] Update EventDialog to call service methods
  - [ ] Update CategoryPanel to call service methods
  - [ ] Update SyncStatusPanel to call service methods
  - [ ] Update all calendar views to call service methods

- [ ] **2.4** Remove direct repository imports from UI files

### Phase 3: Sync Integration

- [ ] **3.1** Move SyncEngine orchestration to CalendarService
- [ ] **3.2** Add sync methods to service:
  - [ ] `sync_source(source_id)` - sync single source
  - [ ] `sync_all_sources()` - sync all configured sources
  - [ ] `get_sync_status(source_id)` - get status for source
  - [ ] `list_sync_sources()` - list configured sources
  - [ ] `configure_import_mapping()` - set category mapping
- [ ] **3.3** Add conflict resolution interface
- [ ] **3.4** Emit MessageBus events for sync progress/completion

### Phase 4: Reminder System Integration

- [ ] **4.1** Create `core/services/calendar_reminders.py` (or integrate into CalendarService)
- [ ] **4.2** Implement background reminder scheduler:
  - [ ] Poll for upcoming reminders
  - [ ] Trigger notification delivery
  - [ ] Handle snooze/dismiss
- [ ] **4.3** Connect to notification system (in-app + system notifications)
- [ ] **4.4** Add reminder methods to CalendarService:
  - [ ] `schedule_event_reminders(event_id)`
  - [ ] `cancel_event_reminders(event_id)`
  - [ ] `get_pending_reminders()`
  - [ ] `dismiss_reminder(reminder_id)`
  - [ ] `snooze_reminder(reminder_id, minutes)`

### Phase 5: Job Integration

- [ ] **5.1** Add event-job linking table to calendar schema:
  ```sql
  calendar_event_job_links (
      event_id UUID REFERENCES calendar_events(id),
      job_id UUID REFERENCES jobs(id),
      link_type VARCHAR(50),  -- 'scheduled_execution', 'deadline', 'milestone'
      created_at TIMESTAMP
  )
  ```
- [ ] **5.2** Add linking methods to CalendarService:
  - [ ] `link_event_to_job(event_id, job_id, link_type)`
  - [ ] `unlink_event_from_job(event_id, job_id)`
  - [ ] `get_events_for_job(job_id)`
  - [ ] `get_job_for_event(event_id)`
- [ ] **5.3** Add job-to-event creation:
  - [ ] `create_event_from_job(job_id, event_type)` - create calendar event for job
  - [ ] Auto-create events when job is scheduled
- [ ] **5.4** Sync job status changes to calendar events:
  - [ ] Job completed → Mark event as done
  - [ ] Job cancelled → Cancel/delete event
  - [ ] Job rescheduled → Update event time
- [ ] **5.5** Add MessageBus handlers for job events

### Phase 6: Task Integration

- [ ] **6.1** Add event-task linking table to calendar schema:
  ```sql
  calendar_event_task_links (
      event_id UUID REFERENCES calendar_events(id),
      task_id UUID REFERENCES tasks(id),
      link_type VARCHAR(50),  -- 'due_date', 'work_block', 'reminder'
      created_at TIMESTAMP
  )
  ```
- [ ] **6.2** Add linking methods to CalendarService:
  - [ ] `link_event_to_task(event_id, task_id, link_type)`
  - [ ] `unlink_event_from_task(event_id, task_id)`
  - [ ] `get_events_for_task(task_id)`
  - [ ] `get_task_for_event(event_id)`
- [ ] **6.3** Add task-to-event creation:
  - [ ] `create_event_from_task(task_id)` - create due date event
  - [ ] `create_work_block_for_task(task_id, duration)` - schedule work time
- [ ] **6.4** Sync task status changes to calendar events:
  - [ ] Task completed → Mark event complete
  - [ ] Task due date changed → Update event
- [ ] **6.5** Add MessageBus handlers for task events

### Phase 7: Agent Calendar Integration

- [ ] **7.1** Add agent schedule methods to CalendarService:
  - [ ] `get_agent_schedule(agent_id, start, end)` - get agent's calendar
  - [ ] `check_agent_availability(agent_id, start, end)` - is agent free?
  - [ ] `get_agent_busy_slots(agent_id, start, end)` - list busy periods
  - [ ] `block_time_for_agent(agent_id, start, end, reason)` - reserve time
- [ ] **7.2** Add scheduling assistance:
  - [ ] `suggest_meeting_times(participants, duration, constraints)`
  - [ ] `find_free_slots(start, end, duration)`
- [ ] **7.3** Create calendar tool for agent access:
  - [ ] Update `modules/Tools/Base_Tools/calendar.py` to use CalendarService
  - [ ] Add operations: `check_availability`, `schedule_meeting`, `block_time`
- [ ] **7.4** Add agent context awareness:
  - [ ] Include upcoming events in agent context
  - [ ] Alert agent to scheduling conflicts
  - [ ] Allow agent to propose calendar changes

### Phase 8: UI Updates for Integration

- [ ] **8.1** Update EventDialog to show linked jobs/tasks
- [ ] **8.2** Add "Create from Job" / "Create from Task" quick actions
- [ ] **8.3** Show job/task status on calendar events
- [ ] **8.4** Add calendar view filters:
  - [ ] Show only job-related events
  - [ ] Show only task-related events
  - [ ] Show agent work blocks
- [ ] **8.5** Add mini-calendar to Job/Task detail views

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/calendar.py` | Main CalendarService class |
| `modules/calendar_store/models.py` | Add job/task link tables |
| `modules/calendar_store/dataclasses.py` | Add link dataclasses |
| `tests/services/test_calendar_service.py` | Service unit tests |
| `tests/calendar/test_job_integration.py` | Job integration tests |
| `tests/calendar/test_task_integration.py` | Task integration tests |

## Files to Modify

| File | Changes |
|------|---------|
| `core/services/__init__.py` | Export CalendarService |
| `core/ATLAS.py` | Add calendar_service property |
| `GTKUI/Calendar_manager/event_dialog.py` | Use service instead of repository |
| `GTKUI/Calendar_manager/category_panel.py` | Use service instead of repository |
| `GTKUI/Calendar_manager/sync_status.py` | Use service for sync operations |
| `GTKUI/Calendar_manager/calendar_view_stack.py` | Use service for queries |
| `modules/calendar_store/schema.py` | Add link tables |
| `modules/calendar_store/repository.py` | Add link CRUD methods |
| `modules/Tools/Base_Tools/calendar.py` | Use CalendarService |

---

## Validation Checklist

Before each phase completion:

- [ ] All existing calendar tests pass (`pytest tests/calendar/`)
- [ ] New service tests pass (`pytest tests/services/test_calendar_service.py`)
- [ ] UI still functions correctly (manual testing)
- [ ] No direct repository access from UI (grep check)
- [ ] MessageBus events firing correctly
- [ ] Documentation updated

---

## Dependencies

- `modules/calendar_store/` - Repository layer (exists)
- `modules/storage/` - StorageManager integration (exists)
- `modules/job_store/` - Job integration target
- `modules/task_store/` - Task integration target
- `core/messaging/` - MessageBus for events
- `modules/background_tasks/` - Reminder scheduling

---

## Success Criteria

1. **Separation of Concerns**: Zero business logic in GTKUI files
2. **Unified API**: Single CalendarService for all calendar operations
3. **Job Integration**: Jobs can create/link calendar events
4. **Task Integration**: Tasks can create/link calendar events
5. **Agent Awareness**: Agents can query and modify calendar
6. **Test Coverage**: >90% coverage on CalendarService
7. **Performance**: Event queries <100ms for 1000 events

---

## Open Questions

1. Should reminder delivery be in CalendarService or separate ReminderService?
2. How to handle timezone conversion - service layer or repository?
3. Should agent calendar access require explicit permission?
4. Multi-tenant calendar sharing - future consideration?
5. Calendar event versioning for conflict resolution?

---

## References

- Existing pattern: `core/services/conversations.py`
- Job store: `modules/job_store/repository.py`
- Task store: `modules/task_store/repository.py`
- Calendar spec: `todos/master-calendar.md`
