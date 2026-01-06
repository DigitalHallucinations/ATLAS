# ATLAS Master Calendar - Feature Specification

> **Status**: Phase 4 Complete ✅  
> **Priority**: High  
> **Target**: Core calendar infrastructure for ATLAS  
> **Last Updated**: 2026-01-06

### Implementation Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core Infrastructure (Schema, Models, CRUD) | ✅ Complete |
| Phase 2 | Category Features (GTK Panel) | ✅ Complete |
| Phase 3 | Event Features (Recurrence) | ✅ Complete |
| Phase 4 | GTK Calendar Views | ✅ Complete |
| Phase 5 | Sync Integration | 🔲 Not Started |
| Phase 6 | Enhanced Features | 🔲 Not Started |

---

## Overview

The ATLAS Master Calendar is a centralized, database-backed calendar system that serves as the authoritative source for all calendar data within ATLAS. External calendars (Google, Outlook, CalDAV, ICS) sync into the master calendar, providing a unified view with fast queries, offline access, and AI-powered features.

---

## Architecture

```Text
┌─────────────────────────────────────────────────────────────────┐
│                     ATLAS Master Calendar                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              PostgreSQL Calendar Store                     │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐   │  │
│  │  │ Events  │  │Categories│  │Reminders│  │ Sync State  │   │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Sync Engine                             │  │
│  │  • Bidirectional sync with external sources                │  │
│  │  • Conflict detection & resolution                         │  │
│  │  • Change tracking (etags, sync tokens, delta queries)     │  │
│  │  • Import mapping (external calendar → category)           │  │
│  └───────────────────────────────────────────────────────────┘  │
│         ↕              ↕              ↕              ↕          │
│    ┌────────┐    ┌─────────┐    ┌────────┐    ┌──────────┐     │
│    │ Google │    │ Outlook │    │ CalDAV │    │ICS Files │     │
│    └────────┘    └─────────┘    └────────┘    └──────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Backend

**PostgreSQL** (already in ATLAS stack)

- Leverages existing connection pooling and migration infrastructure
- Full-text search via `tsvector`
- JSON/JSONB for flexible metadata
- Row-level security for future multi-user support

---

## Calendar Categories

### Built-in Categories

| Category | Color (Default) | Icon | Special Features |
| -------- | --------------- | ---- | ---------------- |
| **Work** | `#4285F4` (Blue) | 💼 | Work hours awareness, meeting detection, busy/free status, video call link detection |
| **Personal** | `#34A853` (Green) | 👤 | Default category for new events, general purpose |
| **Health** | `#EA4335` (Red) | ❤️ | Appointment prep reminders, medication schedules, fasting alerts |
| **Family** | `#FF6D01` (Orange) | 👨‍👩‍👧 | Shared events, birthday tracking, school schedules |
| **Holidays** | `#9334E6` (Purple) | 🎉 | Auto-populated from locale, country-selectable, read-only |
| **Birthdays** | `#E91E63` (Pink) | 🎂 | Auto-import from contacts, annual recurrence, age tracking |

### System-Generated Categories (Read-Only)

- **Holidays**: Populated from public holiday APIs based on user's country/region
- **Birthdays**: Extracted from contacts (when contact integration exists)

### Custom Categories

Users can create unlimited custom categories with:

- Custom name
- Custom color (color picker or preset palette)
- Custom icon (emoji or icon set)
- Optional description
- Visibility toggle
- Sync source mapping

**Common examples**: Sports, Gaming, Side Projects, Freelance, Education, Travel, Social, Finances

---

## Category Configuration

### Category Properties

```yaml
category:
  id: uuid
  name: string              # Display name
  slug: string              # URL-safe identifier
  color: string             # Hex color code
  icon: string              # Emoji or icon identifier
  description: string       # Optional description
  is_builtin: boolean       # System category vs custom
  is_visible: boolean       # Show/hide in calendar views
  is_default: boolean       # New events go here (only one)
  is_readonly: boolean      # For system categories like Holidays
  sort_order: integer       # Display order in lists
  
  # Sync configuration
  sync_sources: list        # Which external calendars map here
  sync_direction: enum      # pull_only | push_only | bidirectional
  
  created_at: timestamp
  updated_at: timestamp
```

### Color Coding

- **Preset Palette**: 12 carefully chosen colors with good contrast
- **Custom Colors**: Full color picker for advanced users
- **Accessibility**: Colors tested for color-blind visibility
- **Dark Mode**: Automatic color adjustment for dark themes

**Preset Palette**:

```text
#4285F4 (Blue)      #34A853 (Green)     #EA4335 (Red)
#FBBC05 (Yellow)    #FF6D01 (Orange)    #9334E6 (Purple)
#E91E63 (Pink)      #00BCD4 (Cyan)      #795548 (Brown)
#607D8B (Gray)      #009688 (Teal)      #673AB7 (Indigo)
```

### Visibility Toggles

- Per-category visibility (show/hide in main view)
- Quick toggle in sidebar category list
- "Show All" / "Hide All" bulk actions
- Visibility persists across sessions
- Keyboard shortcuts for quick toggling

### Default Category

- One category marked as default
- New events without explicit category go here
- Settings UI to change default
- Initially set to "Personal"
- Quick-create uses default category

### Import Mapping

When connecting external calendars:

```yaml
import_mapping:
  google_account:
    "Work": work              # Google "Work" → ATLAS Work
    "Personal": personal      # Google "Personal" → ATLAS Personal
    "jeremy@gmail.com": personal
    "Birthdays": birthdays
    "Holidays in United States": holidays
    
  outlook_account:
    "Calendar": personal      # Default Outlook calendar
    "Work Calendar": work
    
  caldav_nextcloud:
    "personal": personal
    "shared-family": family
```

**Mapping Features**:

- Auto-suggest mappings based on name similarity
- Manual override per external calendar
- Create new category from import
- Unmapped calendars go to default category
- Remember mappings for future syncs

---

## Event Model

### Core Event Properties

```yaml
event:
  # Identity
  id: uuid                    # ATLAS internal ID
  external_id: string         # Original ID from source (for sync)
  external_source: string     # google | outlook | caldav | ics | local
  
  # Content
  title: string
  description: text           # Rich text / markdown
  location: string            # Free-form or structured
  
  # Timing
  start: timestamp            # Start datetime (with timezone)
  end: timestamp              # End datetime (with timezone)
  timezone: string            # IANA timezone
  is_all_day: boolean         # All-day event flag
  
  # Recurrence
  recurrence_rule: string     # RRULE format
  recurrence_id: timestamp    # For exceptions to recurring events
  original_start: timestamp   # Original start for moved occurrences
  
  # Organization
  category_id: uuid           # FK to category
  tags: string[]              # Additional tags
  color_override: string      # Override category color
  
  # Status
  status: enum                # confirmed | tentative | cancelled
  visibility: enum            # public | private | confidential
  busy_status: enum           # busy | free | tentative | out_of_office
  
  # Attendees
  organizer: jsonb            # {name, email}
  attendees: jsonb[]          # [{name, email, status, role}]
  
  # Reminders
  reminders: jsonb[]          # [{minutes_before, method}]
  
  # Metadata
  url: string                 # Video call link, event URL
  attachments: jsonb[]        # File attachments
  custom_properties: jsonb    # Extensible properties
  
  # Sync tracking
  etag: string                # For conflict detection
  sync_status: enum           # synced | pending | conflict | error
  last_synced_at: timestamp
  
  # Audit
  created_at: timestamp
  updated_at: timestamp
  created_by: string          # local | sync
```

### Recurrence Support

Full RFC 5545 RRULE support:

- Daily, weekly, monthly, yearly patterns
- By day of week, day of month, month of year
- Count limit or end date
- Exceptions (EXDATE)
- Recurrence overrides (modified instances)

---

## Feature Set

### Priority 1 (MVP)

- [x] **Database schema & migrations** ✅ *Completed 2026-01-06*
  - Events table with full model
  - Categories table with built-in seeds
  - Import mappings table
  - Sync state tracking table

- [x] **Category management** ✅ *Completed 2026-01-06*
  - CRUD for custom categories
  - Built-in category initialization
  - Color normalization
  - Visibility toggles
  - Default category setting

- [x] **Event CRUD** ✅ *Completed 2026-01-06*
  - Create event with category
  - Read events by date range
  - Update event (single instance)
  - Delete event
  - All-day event support

- [x] **Basic sync infrastructure** ✅ *Completed 2026-01-06*
  - Sync state tracking table
  - Import mapping configuration (repository layer)
  - External ID tracking on events

- [ ] **GTK UI - Calendar view**
  - Month view with events
  - Day view with time slots
  - Week view
  - Category color coding
  - Category visibility sidebar

### Priority 2 (Enhanced)

- [ ] **Sync engine**
  - Import from external calendars (one-way)
  - Manual sync trigger
  - Bidirectional sync

- [ ] **Recurrence**
  - RRULE parsing and generation
  - Recurring event display
  - Edit single vs series
  - Exception handling

- [ ] **Reminders**
  - Multiple reminders per event
  - Notification integration
  - Reminder methods (popup, sound, etc.)

- [x] **Search** ✅ *Completed 2026-01-06*
  - Full-text search (PostgreSQL tsvector + SQLite LIKE fallback)
  - Filter by category, date range

- [ ] **Import mapping UI**
  - Visual mapping editor
  - Auto-suggest based on names
  - Bulk mapping operations

### Priority 3 (Advanced)

- [ ] **AI Integration**
  - Natural language event creation ("Meeting with Bob tomorrow at 3pm")
  - Smart scheduling (find free time)
  - Meeting prep summaries
  - Conflict alerts

- [ ] **Attendees**
  - Add attendees to events
  - Attendee status tracking
  - Email invitations (optional)

- [ ] **Holiday auto-population**
  - Country/region selection
  - Holiday API integration
  - Multi-region support

- [ ] **Birthday import**
  - Contact integration
  - Age calculation
  - Annual recurrence auto-creation

- [ ] **Advanced views**
  - Agenda view (list)
  - Year view (heatmap)
  - Multi-week view
  - Print view

### Priority 4 (Nice-to-Have)

- [ ] **Travel time**
  - Location-aware travel estimates
  - Buffer time suggestions

- [ ] **Video call detection**
  - Auto-detect Zoom/Meet/Teams links
  - One-click join button

- [ ] **Event templates**
  - Save common event patterns
  - Quick-create from template

- [ ] **Keyboard shortcuts**
  - Vim-style navigation
  - Quick create, navigate, edit

- [ ] **Undo/Redo**
  - Undo recent changes
  - Change history

- [ ] **Export**
  - Export to ICS
  - Export to PDF
  - Selective export by category/date

---

## Database Schema

### Tables

```sql
-- Calendar categories
CREATE TABLE calendar_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    color VARCHAR(7) NOT NULL DEFAULT '#4285F4',
    icon VARCHAR(50),
    description TEXT,
    is_builtin BOOLEAN DEFAULT FALSE,
    is_visible BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,
    is_readonly BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0,
    sync_direction VARCHAR(20) DEFAULT 'bidirectional',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Calendar events
CREATE TABLE calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(500),
    external_source VARCHAR(50),
    
    title VARCHAR(500) NOT NULL,
    description TEXT,
    location VARCHAR(500),
    
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    timezone VARCHAR(100) DEFAULT 'UTC',
    is_all_day BOOLEAN DEFAULT FALSE,
    
    recurrence_rule TEXT,
    recurrence_id TIMESTAMPTZ,
    original_start TIMESTAMPTZ,
    
    category_id UUID REFERENCES calendar_categories(id) ON DELETE SET NULL,
    tags TEXT[],
    color_override VARCHAR(7),
    
    status VARCHAR(20) DEFAULT 'confirmed',
    visibility VARCHAR(20) DEFAULT 'public',
    busy_status VARCHAR(20) DEFAULT 'busy',
    
    organizer JSONB,
    attendees JSONB DEFAULT '[]',
    reminders JSONB DEFAULT '[]',
    
    url TEXT,
    attachments JSONB DEFAULT '[]',
    custom_properties JSONB DEFAULT '{}',
    
    etag VARCHAR(100),
    sync_status VARCHAR(20) DEFAULT 'synced',
    last_synced_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Full-text search
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(location, '')), 'C')
    ) STORED
);

-- Indexes
CREATE INDEX idx_events_start ON calendar_events(start_time);
CREATE INDEX idx_events_end ON calendar_events(end_time);
CREATE INDEX idx_events_category ON calendar_events(category_id);
CREATE INDEX idx_events_external ON calendar_events(external_source, external_id);
CREATE INDEX idx_events_search ON calendar_events USING GIN(search_vector);

-- Import mappings
CREATE TABLE calendar_import_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL,        -- google | outlook | caldav
    source_account VARCHAR(200) NOT NULL,    -- account identifier
    source_calendar VARCHAR(200) NOT NULL,   -- external calendar name/id
    target_category_id UUID REFERENCES calendar_categories(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sync state tracking
CREATE TABLE calendar_sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL,
    source_account VARCHAR(200) NOT NULL,
    source_calendar VARCHAR(200),
    sync_token TEXT,                         -- For incremental sync
    last_sync_at TIMESTAMPTZ,
    last_sync_status VARCHAR(20),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Seed Data

```sql
-- Built-in categories
INSERT INTO calendar_categories (name, slug, color, icon, is_builtin, is_default, sort_order) VALUES
    ('Work', 'work', '#4285F4', '💼', TRUE, FALSE, 1),
    ('Personal', 'personal', '#34A853', '👤', TRUE, TRUE, 2),
    ('Health', 'health', '#EA4335', '❤️', TRUE, FALSE, 3),
    ('Family', 'family', '#FF6D01', '👨‍👩‍👧', TRUE, FALSE, 4),
    ('Holidays', 'holidays', '#9334E6', '🎉', TRUE, FALSE, 5),
    ('Birthdays', 'birthdays', '#E91E63', '🎂', TRUE, FALSE, 6);
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

- [ ] Database schema and migrations
- [ ] Category model and repository
- [ ] Event model and repository
- [ ] Basic CRUD operations

### Phase 2: Category Features

- [ ] Category CRUD endpoints
- [ ] Color coding system
- [ ] Visibility toggles
- [ ] Default category logic
- [ ] GTK category management panel

### Phase 3: Event Features

- [ ] Event CRUD endpoints
- [ ] Date range queries
- [ ] All-day event support
- [ ] Basic recurrence (daily, weekly, monthly)

### Phase 4: GTK Calendar View

- [ ] Month view component
- [ ] Day view component
- [ ] Week view component
- [ ] Event display with category colors
- [ ] Event creation/edit dialogs

### Phase 5: Sync Integration

- [ ] Import from existing multi-calendar backends
- [ ] Import mapping configuration
- [ ] Sync state tracking
- [ ] Manual sync trigger

### Phase 6: Enhanced Features

- [ ] Full recurrence support
- [ ] Reminders and notifications
- [ ] Search functionality
- [ ] Bidirectional sync

---

## Configuration

```yaml
# config.yaml additions
calendar:
  master:
    enabled: true
    database: postgresql        # postgresql | sqlite
    
    # Default settings
    default_category: personal
    default_reminder_minutes: 15
    default_view: month         # month | week | day | agenda
    
    # Work hours (for busy/free detection)
    work_hours:
      start: "09:00"
      end: "17:00"
      days: [1, 2, 3, 4, 5]     # Monday-Friday
      timezone: "America/New_York"
    
    # Sync settings
    sync:
      auto_sync: true
      sync_interval_minutes: 15
      conflict_resolution: ask   # ask | local_wins | remote_wins | newest_wins
    
    # Holidays
    holidays:
      enabled: true
      country: US
      regions: []               # Optional: state-specific holidays
    
    # Category settings
    categories:
      color_palette: default    # default | pastel | vibrant | custom
```

---

## File Structure

```Text
modules/
├── calendar_store/
│   ├── __init__.py
│   ├── AGENTS.md
│   ├── models/
│   │   ├── __init__.py
│   │   ├── category.py         # Category dataclass
│   │   ├── event.py            # Event dataclass
│   │   ├── reminder.py         # Reminder dataclass
│   │   └── sync_state.py       # Sync state dataclass
│   ├── repository/
│   │   ├── __init__.py
│   │   ├── category_repo.py    # Category CRUD
│   │   ├── event_repo.py       # Event CRUD + queries
│   │   ├── mapping_repo.py     # Import mapping CRUD
│   │   └── sync_repo.py        # Sync state tracking
│   ├── services/
│   │   ├── __init__.py
│   │   ├── calendar_service.py # High-level calendar operations
│   │   ├── sync_service.py     # Sync engine
│   │   ├── recurrence.py       # RRULE handling
│   │   └── search.py           # Full-text search
│   └── migrations/
│       └── 001_initial.py

GTKUI/
├── Calendar/
│   ├── __init__.py
│   ├── calendar_view.py        # Main calendar container
│   ├── month_view.py           # Month grid
│   ├── week_view.py            # Week columns
│   ├── day_view.py             # Day timeline
│   ├── event_card.py           # Event display widget
│   ├── event_dialog.py         # Create/edit event
│   ├── category_sidebar.py     # Category list + toggles
│   ├── category_dialog.py      # Create/edit category
│   └── import_mapping.py       # Import mapping UI
```

---

## API / Tool Interface

The calendar tool will be extended:

```json
{
  "name": "calendar",
  "operations": [
    "list_events",
    "get_event",
    "search_events",
    "create_event",
    "update_event",
    "delete_event",
    "list_calendars",
    "list_categories",
    "create_category",
    "update_category",
    "delete_category",
    "set_default_category",
    "toggle_category_visibility",
    "sync_calendar",
    "get_sync_status",
    "configure_import_mapping"
  ]
}
```

---

## Dependencies

```Text
# requirements.txt additions
python-dateutil          # Date parsing and recurrence
icalendar               # Already added - RRULE support
pytz                    # Timezone handling
```

---

## Success Criteria

1. **Performance**: Load 1000 events in <100ms
2. **Sync reliability**: 99.9% successful syncs
3. **Offline capable**: Full read/write without network
4. **Search speed**: Full-text search in <50ms
5. **UI responsiveness**: View transitions <16ms (60fps)

---

## Open Questions

1. Should we support multiple "accounts" (e.g., Work Account, Personal Account) or just categories?
2. Do we need event sharing/collaboration features?
3. Should reminders use system notifications or in-app only?
4. Integration with ATLAS task system (events ↔ tasks)?

---

## References

- [RFC 5545 - iCalendar](https://tools.ietf.org/html/rfc5545)
- [Google Calendar API](https://developers.google.com/calendar)
- [Microsoft Graph Calendar](https://docs.microsoft.com/en-us/graph/api/resources/calendar)
- [CalDAV RFC 4791](https://tools.ietf.org/html/rfc4791)
