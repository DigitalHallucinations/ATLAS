# Multi-Calendar Support Implementation

## Overview

Add support for aggregating multiple calendar sources (ICS, Google, Outlook, CalDAV, Apple) into a unified calendar interface with configurable sync behavior.

**Status:** � In Progress (Phase 4 Complete)  
**Owner:** Backend Agent  
**Priority:** High

---

## Architecture Decision

**Approach:** Composite Backend + Provider Registry

```Text
┌─────────────────────────────────────────────────────────┐
│                  CalendarProviderRegistry               │
│  ┌─────────────┬──────────────┬──────────────────────┐  │
│  │ "personal"  │   "work"     │    "family"          │  │
│  │ ICSBackend  │ GoogleBackend│  CalDAVBackend       │  │
│  └─────────────┴──────────────┴──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              CompositeCalendarBackend                   │
│  • list_events() → parallel query all, merge           │
│  • create_event(calendar="work") → route to Google     │
│  • get_event() → search all until found                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Debian12CalendarTool                    │
│            (existing facade, unchanged API)             │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Infrastructure ✅

| # | Task | File | Status |
| - | ---- | ---- | ------ |
| 1.1 | Create `CalendarConfig` dataclass for per-calendar settings | `modules/Tools/Base_Tools/calendar/config.py` | ✅ |
| 1.2 | Create `CalendarProviderRegistry` class | `modules/Tools/Base_Tools/calendar/registry.py` | ✅ |
| 1.3 | Create `CompositeCalendarBackend` implementation | `modules/Tools/Base_Tools/calendar/composite.py` | ✅ |
| 1.4 | Add calendar config section to `config.yaml` | `config.yaml` | ✅ |
| 1.5 | Create `modules/Tools/Base_Tools/calendar/__init__.py` | new package | ✅ |

---

## Phase 2: Calendar Backends ✅

| # | Task | File | Status |
| - | ---- | ---- | ------ |
| 2.1 | Refactor `ICSCalendarBackend` to new package | `calendar/backends/ics.py` | ✅ |
| 2.2 | Refactor `DBusCalendarBackend` to new package | `calendar/backends/dbus.py` | ✅ |
| 2.3 | Implement `GoogleCalendarBackend` | `calendar/backends/google.py` | ✅ |
| 2.4 | Implement `OutlookCalendarBackend` (MS Graph API) | `calendar/backends/outlook.py` | ✅ |
| 2.5 | Create `calendar/backends/__init__.py` with exports | `calendar/backends/__init__.py` | ✅ |
| 2.6 | Create `calendar/factory.py` for backend wiring | `calendar/factory.py` | ✅ |
| 2.7 | Implement `CalDAVBackend` (Nextcloud, Fastmail, etc.) | `calendar/backends/caldav.py` | ⬜ |
| 2.8 | Implement `AppleCalendarBackend` (via CalDAV or local) | `calendar/backends/apple.py` | ⬜ |

---

## Phase 3: Configuration & Settings ✅

| # | Task | File | Status |
| - | ---- | ---- | ------ |
| 3.1 | Add `calendars` section to config schema | `core/config/calendar.py` | ✅ |
| 3.2 | Add `default_calendar` setting | config | ✅ |
| 3.3 | Add per-calendar `sync_mode` setting (realtime/on-demand/read-only) | config | ✅ |
| 3.4 | Add per-calendar `write_enabled` flag | config | ✅ |
| 3.5 | Create GTK UI for calendar management | `GTKUI/Calendar_manager/` | ✅ |

---

## Phase 4: Integration

| # | Task | File | Status |
| - | ---- | ---- | ------ |
| 4.1 | Update `Debian12CalendarTool` to use registry | `debian12_calendar.py` | ✅ |
| 4.2 | Add `list_calendars` operation to tool | tool | ✅ |
| 4.3 | Add OAuth2 flow support for Google/Outlook | `calendar/auth/` | ⬜ |
| 4.4 | Add credential storage integration | `calendar/auth/` | ⬜ |
| 4.5 | Update tool manifest for new operations | manifest | ⬜ |

---

## Phase 5: Testing & Documentation

| # | Task | File | Status |
| - | ---- | ---- | ------ |
| 5.1 | Unit tests for CompositeCalendarBackend | `tests/tools/test_multi_calendar.py` | ✅ |
| 5.2 | Unit tests for each backend | `tests/tools/test_multi_calendar.py` | ✅ |
| 5.3 | Integration tests with mock services | `tests/tools/calendar/` | ⬜ |
| 5.4 | Documentation for calendar setup | `docs/tools/calendar.md` | ✅ |
| 5.5 | Update tool-manifest.md | `docs/tool-manifest.md` | ⬜ |

---

## Configuration Schema

```yaml
calendars:
  default_calendar: personal  # Used when no calendar specified
  
  sources:
    personal:
      type: ics
      path: ~/.local/share/calendars/personal.ics
      write_enabled: true
      sync_mode: on-demand  # realtime | on-demand | read-only
      color: "#4285f4"
      
    work:
      type: google
      account: jeremy@company.com
      calendar_id: primary
      write_enabled: true
      sync_mode: realtime
      color: "#ea4335"
      
    family:
      type: caldav
      url: https://nextcloud.example.com/remote.php/dav/calendars/jeremy/family
      username: jeremy
      # password from secrets store
      write_enabled: true
      sync_mode: on-demand
      color: "#34a853"
      
    holidays:
      type: ics
      url: https://calendar.google.com/calendar/ical/en.usa%23holiday%40group.v.calendar.google.com/public/basic.ics
      write_enabled: false
      sync_mode: daily  # special mode for remote ICS
      color: "#fbbc04"
```

---

## Write Routing Logic

When `create_event()` is called:

1. If `calendar` parameter provided → use that backend
2. Else if persona has `default_calendar` set → use persona's default
3. Else use global `default_calendar` from config
4. If target calendar has `write_enabled: false` → raise error

---

## Calendar Identification

Events will include unified metadata:

```python
@dataclass
class CalendarEvent:
    id: str                    # Unique within backend
    global_id: str            # "{backend_name}:{id}" for cross-backend uniqueness
    title: str
    calendar: str             # Backend name ("personal", "work")
    calendar_account: str     # Account identifier if applicable
    backend_type: str         # "ics", "google", "caldav", etc.
    # ... rest of fields
```

---

## Sync Modes

| Mode | Behavior |
| ---- | -------- |
| `realtime` | Push notifications / webhooks where supported |
| `on-demand` | Fetch fresh data on each query |
| `read-only` | Fetch but never write |
| `daily` | Cache for 24h, background refresh |
| `manual` | Only sync when user triggers |

---

## Dependencies

New packages needed:

- `google-auth` + `google-api-python-client` (Google Calendar)
- `msal` (Microsoft Outlook)
- `caldav` (CalDAV/Apple)
- `icalendar` (already likely present for ICS parsing)

---

## Open Questions

- [ ] Should we support shared/delegated calendars?
- [ ] Free/busy lookup across all calendars?
- [ ] Calendar color in UI?
- [ ] Conflict detection when event appears in multiple calendars?

---

## Acceptance Criteria

- [ ] Can configure 3+ calendar sources of different types
- [ ] `list_events()` returns merged results from all sources
- [ ] `create_event(calendar="work")` routes correctly
- [ ] Events display which calendar they belong to
- [ ] Sync mode is respected per calendar
- [ ] Read-only calendars cannot be written to
- [ ] OAuth flow works for Google/Outlook
- [ ] GTK UI allows adding/removing calendars
- [ ] 80%+ test coverage on new code

---

## Next Steps

Start with Phase 1 (Core Infrastructure) to establish the patterns, then implement backends incrementally.
