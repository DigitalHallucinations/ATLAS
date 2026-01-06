# Calendar Store Agent Guidelines

## Scope

This module handles the ATLAS Master Calendar persistence layer:

- **Writable**: `modules/calendar_store/`
- **Read-only for reference**: `modules/conversation_store/`, `modules/store_common/`

## Responsibilities

- Calendar category CRUD (Work, Personal, Health, Family, Holidays, Birthdays, custom)
- Calendar event CRUD with recurrence support
- Import mapping configuration (external calendar → category)
- Sync state tracking for external calendar sources
- Full-text search on events
- External calendar sync (ICS files/URLs, CalDAV servers)

## File Inventory

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `dataclasses.py` | Pure Python dataclasses for API layer |
| `models.py` | SQLAlchemy ORM models |
| `repository.py` | CRUD operations with session management |
| `schema.py` | Database table creation utilities |
| `recurrence.py` | RRULE parsing, expansion, reminder scheduling |
| `sync_engine.py` | External calendar synchronization engine |
| `providers/__init__.py` | Sync provider package |
| `providers/ics_provider.py` | ICS file/URL sync provider |
| `providers/caldav_provider.py` | CalDAV server sync provider |

## Architecture Integration

```
GTKUI/Calendar_manager/         ← UI layer
        │
        ▼
modules/calendar_store/         ← Data layer (this module)
        │
        ▼
modules/storage/                ← Storage layer (StorageManager)
        │
        ▼
PostgreSQL / SQLite             ← Database
```

## Patterns

Follow the established patterns from `conversation_store`:

1. **Models**: SQLAlchemy models in `models.py` with dialect-aware types
2. **Dataclasses**: Pure Python dataclasses in `dataclasses.py` for API layer
3. **Repository**: CRUD operations in `repository.py` with session management
4. **Schema**: Database creation utilities in `schema.py`

## StorageManager Integration

The calendar store is accessed via the central `StorageManager`:

```python
from modules.storage import get_storage_manager_sync

storage = get_storage_manager_sync()
calendars = storage.calendars  # Lazy-loaded CalendarStoreRepository

# Use repository methods
events = calendars.list_events(start_date=..., end_date=...)
category = calendars.get_category_by_slug("work")
```

## Validation Rules

- Run `pytest tests/calendar/` before submitting changes
- Ensure migrations are backward-compatible
- Category slugs must be unique and URL-safe
- Event times must include timezone information
- All external IDs must be tracked for sync deduplication

## Key Constraints

- Use PostgreSQL-specific features (JSONB, TSVECTOR) with SQLite fallbacks
- Support both single events and recurring event series
- Maintain sync state to avoid duplicate imports
- Color codes must be valid hex format (#RRGGBB)
- Recurrence rules follow RFC 5545 (iCalendar) RRULE format

## Test Coverage

```bash
# Run all calendar tests
python -m pytest tests/calendar/ -v

# Run specific test files
python -m pytest tests/calendar/test_calendar_store.py -v
python -m pytest tests/calendar/test_recurrence.py -v
python -m pytest tests/calendar/test_sync_engine.py -v
```

## Dependencies

- `python-dateutil`: Date parsing and recurrence
- `icalendar`: ICS file parsing (optional but recommended)
- `caldav`: CalDAV protocol support (optional)

## Change Checklist

- [ ] Run `pytest tests/calendar/` before committing
- [ ] Update dataclasses if modifying models
- [ ] Verify StorageManager integration still works
- [ ] Update this AGENTS.md if adding new files
