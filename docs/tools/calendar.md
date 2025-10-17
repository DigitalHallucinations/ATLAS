# Devian 12 Calendar Tool

The Devian 12 calendar tool exposes a read-only view of the desktop
calendar so personas can review upcoming events, inspect meeting details,
and search for entries by keyword.  The tool is intentionally lightweight
and operates entirely on-device using the local Devian calendar backends.

## Configuration

The tool is disabled until calendar access has been configured.  Provide at
least one source using the global configuration file (`config.yaml`) or the
equivalent environment variables.  The module accepts the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `DEVIAN12_CALENDAR_PATHS` | string or array | One or more absolute paths to Devian 12 `.ics` files.  Multiple values can be supplied as an array in YAML or separated by colons in an environment variable. |
| `DEVIAN12_CALENDAR_TZ` | string | Optional default timezone identifier (IANA/Olson style, e.g., `America/Chicago`).  Falls back to `UTC` when omitted. |

When these values are populated the tool automatically loads events from
the configured sources.  Per-installation deployments can also provide the
paths through the `ConfigManager` overrides used in automated tests.

Future versions of the tool can add a DBus-backed provider.  The public API
already supports such extensions via `CalendarBackend` implementations.

## Persona Access

Personas gain access to the calendar tool when they have
`personal_assistant.access_to_calendar` enabled and the persona's
`allowed_tools` list includes `deviant12_calendar`.  The default ATLAS
persona, as well as the aggregate entry in `ALL_PERSONAS.json`, already
exposes this capability so the tool appears in the Persona Tools tab when
calendar access is toggled on.

## Usage

The tool supports three operations, each mapped to the manifest entry:

* `list` – Return events within an optional time window.
* `detail` – Fetch a specific event by identifier.
* `search` – Locate events whose title, description, or location matches a
  supplied query string.

All operations are read-only and respect the timeout and idempotency
guidelines defined in `modules/Tools/tool_maps/functions.json`.

See `modules/Tools/Base_Tools/deviant12_calendar.py` for the runtime entry
point and `tests/test_calendar_tool.py` for concrete examples of invoking
the tool with mocked Devian 12 calendar data.

