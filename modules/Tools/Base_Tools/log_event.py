"""Structured logging tool that forwards events to the message bus."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Mapping

from modules.Tools.tool_event_system import publish_bus_event
from core.messaging import MessagePriority

_REDACTION_REPLACEMENT = "[REDACTED]"
_SENSITIVE_MARKERS = (
    "password",
    "secret",
    "token",
    "apikey",
    "key",
    "session",
    "bearer",
)
_DEFAULT_LOGGER_NAME = "atlas.events"

_SEVERITY_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}

_SEVERITY_PRIORITIES = {
    "debug": MessagePriority.LOW,
    "info": MessagePriority.NORMAL,
    "notice": MessagePriority.NORMAL,
    "warning": MessagePriority.NORMAL,
    "error": MessagePriority.HIGH,
    "critical": MessagePriority.HIGH,
    "alert": MessagePriority.HIGH,
    "emergency": MessagePriority.HIGH,
}


def log_event(
    *,
    event_name: str,
    severity: str,
    payload: Mapping[str, Any] | None = None,
    correlation_id: str | None = None,
    parent_correlation_id: str | None = None,
    persistence: Mapping[str, Any] | None = None,
    logger: str | None = None,
) -> Mapping[str, Any]:
    """Emit a structured event to the message bus and optional logger."""

    if not isinstance(event_name, str) or not event_name:
        raise ValueError("event_name must be a non-empty string")

    severity_key = severity.casefold() if isinstance(severity, str) else ""
    if severity_key not in _SEVERITY_LEVELS:
        valid = ", ".join(sorted(_SEVERITY_LEVELS))
        raise ValueError(f"Unsupported severity '{severity}'. Expected one of: {valid}")

    normalized_payload = _normalize_payload(payload or {}, path="payload")
    sanitized_payload = _redact_payload(normalized_payload)

    event_timestamp = datetime.now(tz=timezone.utc)
    event_payload: dict[str, Any] = {
        "severity": severity_key,
        "emitted_at": event_timestamp.isoformat(),
        "emitted_at_ms": int(event_timestamp.timestamp() * 1000),
        "payload": sanitized_payload,
    }
    if parent_correlation_id:
        event_payload["parent_correlation_id"] = parent_correlation_id

    resolved_priority = _SEVERITY_PRIORITIES[severity_key]
    emit_legacy = True
    metadata_payload: dict[str, Any] = {}
    tracing_payload: dict[str, Any] | None = None
    logger_name = logger or _DEFAULT_LOGGER_NAME

    if persistence is not None:
        if not isinstance(persistence, Mapping):
            raise TypeError("persistence must be a mapping when provided")

        if "priority" in persistence:
            resolved_priority = _resolve_priority(persistence["priority"])

        if "emit_legacy" in persistence:
            emit_legacy = bool(persistence["emit_legacy"])

        if "metadata" in persistence and persistence["metadata"] is not None:
            metadata_payload = _ensure_mapping(persistence["metadata"], path="persistence.metadata")

        if "tracing" in persistence and persistence["tracing"] is not None:
            tracing_payload = _ensure_mapping(persistence["tracing"], path="persistence.tracing")

        if "logger" in persistence and persistence["logger"]:
            logger_name = str(persistence["logger"])

    logger_instance = logging.getLogger(logger_name)
    log_level = _SEVERITY_LEVELS[severity_key]
    if logger_instance.isEnabledFor(log_level):
        logger_instance.log(
            log_level,
            "event=%s payload=%s",  # consistent structured key/value style
            event_name,
            sanitized_payload,
        )

    correlation = publish_bus_event(
        event_name,
        event_payload,
        priority=resolved_priority,
        correlation_id=correlation_id,
        tracing=tracing_payload,
        metadata=metadata_payload or None,
        emit_legacy=emit_legacy,
    )

    return {
        "correlation_id": correlation,
        "emitted_at": event_payload["emitted_at"],
        "priority": resolved_priority,
    }


def _normalize_payload(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_payload(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_payload(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"{path} contains a non-finite float value")
        return value
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                redacted[str(key)] = _REDACTION_REPLACEMENT
            else:
                redacted[str(key)] = _redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, str) and _is_sensitive_value(value):
        return _REDACTION_REPLACEMENT
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _is_sensitive_value(value: str) -> bool:
    lowered = value.casefold()
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _resolve_priority(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.casefold()
        if normalized == "high":
            return MessagePriority.HIGH
        if normalized == "normal":
            return MessagePriority.NORMAL
        if normalized == "low":
            return MessagePriority.LOW
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unrecognized priority value '{value}'") from exc
    raise TypeError("priority hint must be an integer or string")


def _ensure_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return _normalize_payload(value, path=path)


__all__ = ["log_event"]
