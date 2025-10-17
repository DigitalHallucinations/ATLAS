# Debian 12 Calendar Tool

The Debian 12 calendar tool exposes the desktop calendar so personas can
review upcoming events, inspect meeting details, search for entries by
keyword, and now create or modify calendar entries.  The tool is
intentionally lightweight and operates entirely on-device using the local
Debian 12 calendar backends.

## Configuration

The tool is disabled until calendar access has been configured.  Provide at
least one source using the global configuration file (`config.yaml`) or the
equivalent environment variables.  The module accepts the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `DEBIAN12_CALENDAR_PATHS` | string or array | One or more absolute paths to Debian 12 `.ics` files.  Multiple values can be supplied as an array in YAML or separated by colons in an environment variable.  Paths must be writable to enable create/update/delete operations. |
| `DEBIAN12_CALENDAR_TZ` | string | Optional default timezone identifier (IANA/Olson style, e.g., `America/Chicago`).  Falls back to `UTC` when omitted. |

When these values are populated the tool automatically loads events from
the configured sources.  Per-installation deployments can also provide the
paths through the `ConfigManager` overrides used in automated tests.

Write operations acquire an exclusive advisory lock around the targeted
`.ics` file before applying modifications so concurrent tool invocations do
not corrupt the calendar.  Installations that integrate with a DBus-backed
provider can supply a custom `CalendarBackend` that implements the same
`create_event`, `update_event`, and `delete_event` APIs.

## Persona Access

Personas gain read-only access to the calendar tool when they have
`personal_assistant.access_to_calendar` enabled and the persona's
`allowed_tools` list includes `debian12_calendar`.  Enabling the companion
`personal_assistant.calendar_write_enabled` flag elevates the permission so
the persona may create, update, or delete events in addition to listing and
inspecting them.  The default ATLAS persona, as well as the aggregate entry in
`ALL_PERSONAS.json`, already exposes the read toggle so the tool appears in the
Persona Tools tab; write access remains opt-in for installations that are
comfortable with automated edits.

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
point and `tests/test_calendar_tool.py` for concrete examples of invoking
the tool with mocked Debian 12 calendar data and writable `.ics` fixtures.

