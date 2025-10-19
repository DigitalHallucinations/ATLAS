# Time Tool

The `get_current_info` tool returns formatted information about the current
moment, including the time, date, day of the week, month/year, or a combined
timestamp. It now supports configurable timezones so personas can align
responses with their locale.

## Parameters

- `format_type` – Selects the formatting style (`time`, `date`, `day`,
  `month_year`, or `timestamp`). Defaults to `timestamp` when omitted.
- `timezone` – Optional [IANA/Olson timezone identifier](https://www.iana.org/time-zones)
  such as `America/New_York`. When excluded the tool uses the configured
  default timezone, falls back to the system timezone when available, and
  ultimately uses `UTC`.

## Configuration

You can set a default timezone through configuration so callers do not need to
specify it explicitly:

```yaml
# config.yaml
TIME_TOOL_DEFAULT_TZ: "Europe/Berlin"

# or via the structured tools block
tools:
  time:
    default_timezone: "America/Chicago"
```

If neither setting is provided the tool attempts to infer the system timezone
and finally defaults to `UTC`.
