"""Utilities for normalizing and filtering application logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["LogParser", "ParsedLogEntry"]


@dataclass(frozen=True)
class ParsedLogEntry:
    """Normalized representation of a single log entry."""

    timestamp: str | None
    severity: str
    message: str
    fields: Mapping[str, Any]
    raw: str


class LogParser:
    """Parse newline-delimited logs and return structured entries."""

    _SEVERITY_FALLBACK = "info"
    _SEVERITY_KEYS = ("severity", "level", "log_level", "priority", "status")
    _MESSAGE_KEYS = ("message", "msg", "event", "log", "summary")
    _TIME_KEYS = ("timestamp", "time", "ts", "logged_at")

    def run(
        self,
        *,
        log_source: str,
        format: str = "auto",
        severities: Sequence[str] | None = None,
        contains: Sequence[str] | None = None,
        patterns: Sequence[str] | None = None,
        limit: int | None = None,
        time_window: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any]:
        """Parse and filter the supplied log payload."""

        if not isinstance(log_source, str):
            raise TypeError("log_source must be a string of newline-delimited entries")

        normalized_format = (format or "auto").strip().lower()
        if normalized_format not in {"auto", "json", "key_value"}:
            raise ValueError("format must be one of: auto, json, key_value")

        severity_filter = {
            item.strip().lower()
            for item in (severities or [])
            if isinstance(item, str) and item.strip()
        }
        contains_filter = [
            item.strip().lower()
            for item in (contains or [])
            if isinstance(item, str) and item.strip()
        ]

        compiled_patterns: list[re.Pattern[str]] = []
        for pattern in patterns or []:
            if not isinstance(pattern, str) or not pattern:
                continue
            try:
                compiled_patterns.append(re.compile(pattern, flags=re.IGNORECASE))
            except re.error:
                # Skip invalid patterns but keep parsing the rest of the payload.
                continue

        start_dt: datetime | None = None
        end_dt: datetime | None = None
        if isinstance(time_window, Mapping):
            start_dt = _parse_datetime(time_window.get("start"))
            end_dt = _parse_datetime(time_window.get("end"))

        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer when provided")

        lines = [line for line in log_source.splitlines() if line.strip()]
        entries: list[ParsedLogEntry] = []
        parse_errors = 0
        unique_severities: set[str] = set()

        for line in lines:
            entry = self._parse_line(line, normalized_format)
            if entry is None:
                parse_errors += 1
                continue

            if severity_filter and entry.severity not in severity_filter:
                continue

            lowered_line = entry.raw.lower()
            if contains_filter and not all(fragment in lowered_line for fragment in contains_filter):
                continue

            if compiled_patterns and not any(pattern.search(entry.raw) for pattern in compiled_patterns):
                continue

            if (start_dt or end_dt) and entry.timestamp is not None:
                entry_dt = _parse_datetime(entry.timestamp)
                if entry_dt is None:
                    continue
                if start_dt and entry_dt < start_dt:
                    continue
                if end_dt and entry_dt > end_dt:
                    continue

            unique_severities.add(entry.severity)
            entries.append(entry)

        if limit is not None:
            entries = entries[:limit]

        payload = {
            "total": len(lines),
            "matched": len(entries),
            "entries": [asdict(entry) for entry in entries],
            "filters": {
                "severities": sorted(severity_filter) if severity_filter else [],
                "contains": contains_filter,
                "patterns": [pattern.pattern for pattern in compiled_patterns],
                "time_window": {
                    key: value
                    for key, value in {"start": _format_datetime(start_dt), "end": _format_datetime(end_dt)}.items()
                    if value is not None
                },
            },
            "unique_severities": sorted(unique_severities),
            "parse_errors": parse_errors,
        }
        return payload

    def _parse_line(self, line: str, format_hint: str) -> ParsedLogEntry | None:
        if format_hint in {"auto", "json"}:
            parsed = self._try_parse_json(line)
            if parsed is not None:
                return parsed

        if format_hint in {"auto", "key_value"}:
            parsed = self._try_parse_key_value(line)
            if parsed is not None:
                return parsed

        return self._fallback_entry(line)

    def _try_parse_json(self, line: str) -> ParsedLogEntry | None:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(candidate, Mapping):
            return None

        normalized = _normalize_mapping(candidate)
        timestamp = self._extract_timestamp(normalized)
        severity = self._extract_severity(normalized)
        message = self._extract_message(normalized, default=line.strip())

        return ParsedLogEntry(
            timestamp=timestamp,
            severity=severity,
            message=message,
            fields=normalized,
            raw=line,
        )

    def _try_parse_key_value(self, line: str) -> ParsedLogEntry | None:
        tokens = [token for token in re.split(r"\s+", line.strip()) if token]
        if not tokens:
            return None

        fields: MutableMapping[str, Any] = {}
        for token in tokens:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"')
            if not value:
                continue
            fields[key] = value

        if not fields:
            return None

        timestamp = self._extract_timestamp(fields)
        severity = self._extract_severity(fields)
        message = self._extract_message(fields, default=line.strip())
        return ParsedLogEntry(
            timestamp=timestamp,
            severity=severity,
            message=message,
            fields=dict(fields),
            raw=line,
        )

    def _fallback_entry(self, line: str) -> ParsedLogEntry | None:
        trimmed = line.strip()
        if not trimmed:
            return None
        return ParsedLogEntry(
            timestamp=None,
            severity=self._SEVERITY_FALLBACK,
            message=trimmed,
            fields={"message": trimmed},
            raw=line,
        )

    def _extract_timestamp(self, fields: Mapping[str, Any]) -> str | None:
        for key in self._TIME_KEYS:
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                normalized = value.strip()
                if _parse_datetime(normalized) is not None:
                    return normalized
            if isinstance(value, (int, float)):
                try:
                    dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    continue
                return dt.isoformat()
        return None

    def _extract_severity(self, fields: Mapping[str, Any]) -> str:
        for key in self._SEVERITY_KEYS:
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            if isinstance(value, (int, float)):
                numeric = float(value)
                if numeric >= 50:
                    return "critical"
                if numeric >= 40:
                    return "error"
                if numeric >= 30:
                    return "warning"
                if numeric >= 20:
                    return "notice"
                return "info"
        return self._SEVERITY_FALLBACK

    def _extract_message(self, fields: Mapping[str, Any], *, default: str) -> str:
        for key in self._MESSAGE_KEYS:
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default


def _normalize_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key)] = value
    return normalized


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    candidate = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()
