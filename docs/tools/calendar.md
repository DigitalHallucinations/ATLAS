---
audience: Persona authors and operators
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/Tools/Base_Tools/debian12_calendar.py; modules/Tools/Base_Tools/calendar/; config.yaml
---

# Calendar Tool

The ATLAS Calendar Tool provides unified access to multiple calendar sources
through a single interface. It supports reading, creating, updating, and
deleting events across various calendar backends including local ICS files,
Google Calendar, Microsoft Outlook/365, and the Debian 12 desktop DBus service.

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| **ICS** | Local ICS/iCalendar files | None (built-in) |
| **DBus** | Debian 12 desktop calendar service | D-Bus access |
| **Google** | Google Calendar API | `google-auth`, `google-auth-oauthlib`, `google-api-python-client` |
| **Outlook** | Microsoft 365 / Outlook | `msal`, `requests` |
| **CalDAV** | CalDAV servers (Nextcloud, Fastmail) | `caldav` (future) |

## Multi-Calendar Configuration

Configure multiple calendar sources in `config.yaml`:

```yaml
calendars:
  default_calendar: personal  # Default for write operations
  
  sources:
    personal:
      type: ics
      path: ~/.local/share/calendars/personal.ics
      write_enabled: true
      sync_mode: on-demand
      color: "#4285f4"
      priority: 100
      
    work:
      type: google
      credentials_path: ~/.config/atlas/google_credentials.json
      calendar_id: primary
      write_enabled: true
      sync_mode: realtime
      color: "#0f9d58"
      priority: 90
      timezone: America/New_York
      
    family:
      type: outlook
      client_id: your-azure-app-client-id
      tenant_id: common
      calendar_id: default
      write_enabled: true
      sync_mode: on-demand
      color: "#f4b400"
```

### Configuration Options

#### Common Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | string | required | Backend type: `ics`, `google`, `outlook`, `caldav`, `dbus` |
| `write_enabled` | bool | `true` | Allow creating/updating/deleting events |
| `sync_mode` | string | `on-demand` | Sync behavior: `realtime`, `on-demand`, `read-only`, `daily`, `manual` |
| `color` | string | `#3584e4` | Display color (hex) |
| `priority` | int | `100` | Priority for conflict resolution (higher = more important) |
| `timezone` | string | `UTC` | Default timezone for this calendar |

#### ICS Backend Options

| Option | Type | Description |
|--------|------|-------------|
| `path` | string | Path to ICS file (supports `~` expansion) |

#### Google Backend Options

| Option | Type | Description |
|--------|------|-------------|
| `credentials_path` | string | Path to Google OAuth2 credentials JSON |
| `calendar_id` | string | Google Calendar ID (use `primary` for main calendar) |

#### Outlook Backend Options

| Option | Type | Description |
|--------|------|-------------|
| `client_id` | string | Azure AD application (client) ID |
| `tenant_id` | string | Azure AD tenant ID (use `common` for multi-tenant) |
| `calendar_id` | string | Calendar ID (use `default` for main calendar) |

## Legacy Configuration

The tool also supports legacy single-backend configuration for backward
compatibility.  Provide at least one source using the global configuration
file (`config.yaml`) or environment variables:

| Key | Type | Description |
| --- | ---- | ----------- |
| `DEBIAN12_CALENDAR_BACKEND` | string | Selects which backend to use.  Defaults to the file-based `ics` backend.  Set to `dbus` to route all calls through the Debian 12 calendar DBus service. |
| `DEBIAN12_CALENDAR_DBUS_FACTORY` | string | Optional dotted path (either `package.module:callable` or `package.module.callable`) used to instantiate a DBus client when the backend is set to `dbus`.  The callable must return an object that implements the backend CRUD methods. |
| `DEBIAN12_CALENDAR_PATHS` | string or array | One or more absolute paths to Debian 12 `.ics` files.  Multiple values can be supplied as an array in YAML or separated by colons in an environment variable.  Paths must be writable to enable create/update/delete operations when using the `ics` backend. |
| `DEBIAN12_CALENDAR_TZ` | string | Optional default timezone identifier (IANA/Olson style, e.g., `America/Chicago`).  Falls back to `UTC` when omitted. |

When these values are populated the tool automatically loads events from
the configured sources.  Per-installation deployments can also provide the
paths through the `ConfigManager` overrides used in automated tests.  When
`DEBIAN12_CALENDAR_BACKEND` is set to `dbus` the tool relies entirely on the
DBus client returned by the configured factory (or one supplied directly to
`Debian12CalendarTool`) and ignores the `.ics` path settings.

### Defaults and fallbacks

* Backend defaults to `ics` with UTC timezone and no preconfigured calendar
  paths; when no paths are supplied the tool falls back to the null backend.
* `DEBIAN12_CALENDAR_DBUS_FACTORY` must resolve to a callable; when missing or
  failing to load, the tool logs a warning and returns a null backend instead of
  attempting DBus operations.
* Write operations acquire advisory locks for `.ics` files; DBus writes defer to
  the desktop service.

Write operations acquire an exclusive advisory lock around the targeted
`.ics` file before applying modifications so concurrent tool invocations do
not corrupt the calendar when running against local files.  The DBus backend
delegates writes to the desktop calendar service instead; create, update, and
delete actions are available as soon as the DBus client successfully connects.

## Persona Access

Personas gain read-only access to the calendar tool when they have
`personal_assistant.access_to_calendar` enabled and the persona's
`allowed_tools` list includes `debian12_calendar`.  Enabling the companion
`personal_assistant.calendar_write_enabled` flag elevates the permission so
the persona may create, update, or delete events in addition to listing and
inspecting them.  The default ATLAS persona definition
(`modules/Personas/ATLAS/Persona/ATLAS.json`) already exposes the read toggle so
the tool appears in the Persona Tools tab; write access remains opt-in for
installations that are
comfortable with automated edits.

The tool manifest now declares a `requires_flags` map for the `create`,
`update`, and `delete` operations.  `ToolManager` consults this metadata before
each invocation and blocks the call when the active persona is missing either
`type.personal_assistant.access_to_calendar` or
`type.personal_assistant.calendar_write_enabled`, returning a clear policy
reason.  The runtime also injects the persona snapshot into the tool context so
`Debian12CalendarTool.run` can enforce the same guardrail, ensuring write
attempts from read-only personas fail fast while list/detail/search remain
available.  The Persona Tools tab surfaces these constraints via the tool state
so operators understand why write actions are unavailable.

## Usage

The tool supports six operations, each mapped to the manifest entry:

* `list` – Return events within an optional time window.
* `detail` – Fetch a specific event by identifier.
* `search` – Locate events whose title, description, or location matches a
  supplied query string.
* `create` – Create a new event with a title, start/end times, optional
  attendees, location, and description.
* `update` – Modify an existing event by identifier.
* `delete` – Remove an event by identifier.

Read operations remain idempotent and inexpensive.  Create/update/delete
mutate the underlying calendar file and should only be issued when the
persona has the appropriate policy permissions.

See `modules/Tools/Base_Tools/debian12_calendar.py` for the runtime entry
point and `tests/tools/test_calendar_tool.py` for concrete examples of invoking
the tool with mocked calendar data and writable `.ics` fixtures.

## Operations

The tool supports seven operations, each mapped to the manifest entry:

* `list` – Return events within an optional time window.
* `detail` – Fetch a specific event by identifier.
* `search` – Locate events whose title, description, or location matches a
  supplied query string.
* `create` – Create a new event with a title, start/end times, optional
  attendees, location, and description.
* `update` – Modify an existing event by identifier.
* `delete` – Remove an event by identifier.
* `list_calendars` – List all configured calendar sources.

### list_calendars Operation

Returns information about all configured calendar sources:

```python
await tool.run("list_calendars")
```

**Returns:**
```json
[
  {
    "name": "personal",
    "display_name": "Personal Calendar",
    "type": "ics",
    "write_enabled": true,
    "is_default": true,
    "color": "#4285f4"
  },
  {
    "name": "work",
    "display_name": "Work Calendar",
    "type": "google",
    "write_enabled": true,
    "is_default": false,
    "color": "#0f9d58"
  }
]
```

## Setting Up OAuth

### Google Calendar

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the Google Calendar API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download the credentials JSON
6. Configure in `config.yaml`:

```yaml
calendars:
  sources:
    google_cal:
      type: google
      credentials_path: ~/.config/atlas/google_oauth.json
      calendar_id: primary
```

7. On first use, a browser window will open for authorization

### Microsoft Outlook / Office 365

1. Go to [Azure Portal](https://portal.azure.com/)
2. Register a new application in Azure AD
3. Configure API permissions for Microsoft Graph:
   - `Calendars.ReadWrite`
   - `User.Read`
4. Note the Application (client) ID and Tenant ID
5. Configure in `config.yaml`:

```yaml
calendars:
  sources:
    outlook_cal:
      type: outlook
      client_id: your-client-id-here
      tenant_id: common
      calendar_id: default
```

6. On first use, you'll be prompted to authenticate

## Environment Variables

Override configuration with environment variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_CALENDAR_CREDENTIALS` | Path to Google OAuth credentials |
| `AZURE_CLIENT_ID` | Microsoft Azure application client ID |
| `AZURE_TENANT_ID` | Microsoft Azure tenant ID |

## Architecture

The multi-calendar system uses a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│              CompositeCalendarBackend                   │
│  • list_events() → parallel query all, merge           │
│  • create_event(calendar="work") → route to Google     │
│  • get_event() → search all until found                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 CalendarProviderRegistry                │
│  ┌─────────────┬──────────────┬──────────────────────┐  │
│  │ "personal"  │   "work"     │    "family"          │  │
│  │ ICSBackend  │ GoogleBackend│  OutlookBackend      │  │
│  └─────────────┴──────────────┴──────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Debian12CalendarTool                   │
│              (unified tool manifest API)                │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "Calendar access has not been configured"

Ensure at least one calendar is configured in `config.yaml` or set legacy config:
```yaml
DEBIAN12_CALENDAR_PATHS: "~/.local/share/calendars/default.ics"
```

### Google OAuth errors

- Ensure credentials file exists and is valid JSON
- Delete token cache and re-authorize: `~/.config/atlas/google_token.json`
- Check that Calendar API is enabled in Google Cloud Console

### Outlook authentication fails

- Verify client ID and tenant ID are correct
- Ensure app has correct API permissions
- Check that redirect URI is configured in Azure

### Events not syncing

- Check sync_mode setting (`realtime` vs `on-demand`)
- Use Sync Status panel in Calendar Manager UI to force manual sync
- Check network connectivity to calendar services
