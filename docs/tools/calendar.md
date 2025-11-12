# Debian 12 Calendar Tool

The Debian 12 calendar tool exposes the desktop calendar so personas can
review upcoming events, inspect meeting details, search for entries by
keyword, and now create or modify calendar entries.  The tool is
intentionally lightweight and operates entirely on-device using either the
local Debian 12 `.ics` files or the desktop DBus calendar service.

For installations that are not connected to a Debian calendar backend,
`calendar_service` offers an in-memory alternative so personas can still
book lightweight slots or retrieve ad-hoc availability.  The helper follows
the same manifest conventions, exposing `book`, `list`, and `get` style
operations without persisting data to disk.

## Configuration

The tool is disabled until calendar access has been configured.  Provide at
least one source using the global configuration file (`config.yaml`) or the
equivalent environment variables.  The module accepts the following keys:

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
point and `tests/test_calendar_tool.py` for concrete examples of invoking
the tool with mocked Debian 12 calendar data and writable `.ics` fixtures.

