# Calendar Manager AGENTS.md

## Module Scope

The `GTKUI/Calendar_manager/` module provides GTK 4 user interface components for the ATLAS Master Calendar system. This module is owned by the **UI Agent** role.

## File Inventory

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports for all calendar widgets |
| `calendar_agenda_view.py` | Agenda/list view of events |
| `calendar_day_view.py` | Day view with hourly timeline |
| `calendar_dialog.py` | Dialog for adding external calendar sources |
| `calendar_list.py` | Panel listing configured calendar sources |
| `calendar_management.py` | Main workspace controller (orchestrates all views) |
| `calendar_month_view.py` | Month grid view |
| `calendar_settings.py` | Settings panel for calendar preferences |
| `calendar_view_stack.py` | GTK Stack managing Month/Week/Day/Agenda views |
| `calendar_week_view.py` | Week view with columns per day |
| `category_dialog.py` | Dialog for creating/editing categories |
| `category_panel.py` | Category list with color/visibility toggles |
| `color_chooser.py` | Color picker widget with preset palette |
| `event_card.py` | Compact event display widget |
| `event_dialog.py` | Full event create/edit dialog |
| `import_mapping_panel.py` | UI for mapping external calendars to categories |
| `mini_calendar.py` | Compact month picker for sidebar |
| `sync_status.py` | Panel showing sync status and controls |

## Architecture Integration

```
GTKUI/Calendar_manager/         ← UI layer (this module)
        │
        ▼
modules/calendar_store/         ← Data layer (repository, models)
        │
        ▼
modules/storage/                ← Storage layer (StorageManager)
        │
        ▼
PostgreSQL / SQLite             ← Database
```

## Key Patterns

### Widget Hierarchy

- `CalendarManagement` (workspace controller)
  - `CalendarViewStack` (main calendar views)
    - `CalendarMonthView`
    - `CalendarWeekView`
    - `CalendarDayView`
    - `CalendarAgendaView`
  - `CalendarListPanel` (external sources)
  - `CategoryPanel` (category management)
  - `SyncStatusPanel` (sync controls)
  - `CalendarSettingsPanel` (preferences)

### Data Flow

1. UI components access storage via `ATLAS.storage.calendars`
2. Repository methods return dataclasses from `modules/calendar_store/dataclasses.py`
3. Events are displayed via `EventCard` widgets
4. Changes trigger MessageBus events for cross-component updates

## Required Tests

Before modifying UI components:

```bash
# Run calendar tests
python -m pytest tests/calendar/ -v

# Verify imports
python -c "from GTKUI.Calendar_manager import CalendarManagement"
```

## Style Guidelines

- Follow GTK 4 patterns (no deprecated GTK 3 APIs)
- Use `Gtk.Box`, `Gtk.Grid`, `Gtk.Stack` for layout
- Apply CSS classes from `GTKUI/Utils/utils.py`
- Connect signals in `__init__` or `_build_ui` methods
- Handle cleanup in explicit `cleanup()` methods

## Cross-Module Dependencies

| Dependency | Usage |
|------------|-------|
| `modules/calendar_store/` | Data access (repository, dataclasses) |
| `modules/storage/` | StorageManager integration |
| `core/messaging/` | MessageBus for events |
| `GTKUI/Utils/` | Shared CSS and window utilities |

## Configuration

Calendar settings are stored in `config.yaml` under the `calendar` key:

```yaml
calendar:
  enabled: true
  default_view: month
  default_reminder_minutes: 15
  work_hours:
    start: "09:00"
    end: "17:00"
    days: [1, 2, 3, 4, 5]
  sync:
    auto_sync: true
    sync_interval_minutes: 15
    conflict_resolution: ask
```

## Change Checklist

- [ ] Run `pytest tests/calendar/` before committing
- [ ] Verify imports in `__init__.py` if adding new widgets
- [ ] Update this AGENTS.md if adding new files
- [ ] Test with GTK Inspector for layout issues (`GTK_DEBUG=interactive`)
