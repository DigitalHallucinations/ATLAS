"""Caching and activity log helpers for tool execution."""
from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from modules.Tools.tool_event_system import publish_bus_event
from modules.logging.logger import setup_logger
from modules.orchestration.message_bus import MessagePriority

from ATLAS.config import ConfigManager

logger = setup_logger(__name__)

_DEFAULT_CONFIG_MANAGER: Optional[ConfigManager] = None
_CONFIG_MANAGER_LOCK = threading.Lock()

_TOOL_ACTIVITY_EVENT = "tool_activity"
_tool_activity_log: deque = deque(maxlen=100)
_tool_activity_lock = threading.Lock()

_SENSITIVE_PAYLOAD_FIELDS = ("arguments", "result", "stdout", "stderr")
_REDACTION_REPLACEMENT = "<redacted>"
_SECRET_PATTERNS = (
    # Common OpenAI style keys
    __import__("re").compile(r"sk-[A-Za-z0-9\-]{8,}"),
    __import__("re").compile(r"rk-[A-Za-z0-9\-]{8,}"),
    __import__("re").compile(r"pk-[A-Za-z0-9\-]{8,}"),
    # AWS access keys
    __import__("re").compile(r"AKIA[0-9A-Z]{12,}"),
    # Google API keys
    __import__("re").compile(r"AIza[0-9A-Za-z\-_]{20,}"),
)
_SECRET_ASSIGNMENT_PATTERN = __import__("re").compile(
    r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([A-Za-z0-9\-_=]{6,})"
)
_SECRET_JSON_PATTERN = __import__("re").compile(
    r"(?i)(\"(?:api[_-]?key|token|secret|password)\"\s*:\s*\")([^\"\\]{4,})(\")"
)


def get_config_manager(candidate: Optional[ConfigManager] = None) -> ConfigManager:
    """Return a :class:`ConfigManager`, caching the default instance."""

    global _DEFAULT_CONFIG_MANAGER

    if candidate is not None:
        return candidate

    with _CONFIG_MANAGER_LOCK:
        if _DEFAULT_CONFIG_MANAGER is None:
            _DEFAULT_CONFIG_MANAGER = ConfigManager()
        return _DEFAULT_CONFIG_MANAGER


def get_config_section(config_manager, key: str):
    if config_manager is None:
        return None

    if hasattr(config_manager, "config_data"):
        data = getattr(config_manager, "config_data")
        if isinstance(data, Mapping):
            return data.get(key)

    if hasattr(config_manager, "get_config"):
        try:
            return config_manager.get_config(key)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Failed to retrieve config section '%s'", key)
    return None


def clone_json_compatible(value: Any) -> Any:
    """Return a JSON-compatible clone of ``value`` when possible."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


def stringify_tool_value(value: Any) -> str:
    """Return a human-readable string for tool payload data."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _redact_text(value: str) -> str:
    if not value:
        return value

    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTION_REPLACEMENT, redacted)

    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTION_REPLACEMENT}",
        redacted,
    )

    redacted = _SECRET_JSON_PATTERN.sub(
        lambda match: f"{match.group(1)}{_REDACTION_REPLACEMENT}{match.group(3)}",
        redacted,
    )

    return redacted


def _redact_payload_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, Mapping):
        return {key: _redact_payload_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_redact_payload_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact_payload_value(item) for item in value)

    if isinstance(value, set):
        return {_redact_payload_value(item) for item in value}

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return _redact_text(str(value))


def _summarize_payload_value(value: Any, *, limit: int) -> str:
    text = stringify_tool_value(value).strip()
    if limit <= 0 or len(text) <= limit:
        return text

    if limit == 1:
        return "…"

    return text[: limit - 1] + "…"


def get_tool_logging_preferences(config_manager=None) -> Dict[str, Any]:
    section = get_config_section(config_manager, "tool_logging")
    if not isinstance(section, Mapping):
        return {"log_full_payloads": False, "payload_summary_length": 256}

    preferences = dict(section)
    preferences.setdefault("log_full_payloads", False)
    preferences.setdefault("payload_summary_length", 256)
    return preferences


def build_tool_metrics(entry: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    status = entry.get("status")
    if status is not None:
        metrics["status"] = status

    latency = entry.get("duration_ms")
    if latency is not None:
        metrics["latency_ms"] = latency

    started = entry.get("started_at")
    if started is not None:
        metrics["started_at"] = started

    completed = entry.get("completed_at")
    if completed is not None:
        metrics["completed_at"] = completed

    timeout_seconds = entry.get("timeout_seconds")
    if timeout_seconds is not None:
        metrics["timeout_seconds"] = timeout_seconds

    error_type = entry.get("error_type")
    if error_type is not None:
        metrics["error_type"] = error_type

    result_value = entry.get("result")
    if isinstance(result_value, list):
        metrics["result_item_count"] = len(result_value)

    tool_version = entry.get("tool_version")
    metadata = entry.get("metadata")
    if tool_version is not None:
        metrics["tool_version"] = tool_version
    elif isinstance(metadata, Mapping):
        candidate = metadata.get("version") or metadata.get("tool_version")
        if candidate is not None:
            metrics["tool_version"] = candidate

    return metrics


def build_public_tool_entry(entry: Mapping[str, Any], *, config_manager=None) -> Dict[str, Any]:
    preferences = get_tool_logging_preferences(config_manager)
    log_full_payloads = bool(preferences.get("log_full_payloads"))
    summary_limit = int(preferences.get("payload_summary_length", 256) or 0)

    public_entry: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in _SENSITIVE_PAYLOAD_FIELDS:
            continue
        if key.startswith("_"):
            continue
        if isinstance(value, str) and key in {"arguments_text", "result_text", "error"}:
            redacted_value = _redact_text(value)
            if not log_full_payloads and key in {"arguments_text", "result_text"}:
                redacted_value = _summarize_payload_value(
                    redacted_value, limit=summary_limit
                )
            public_entry[key] = redacted_value
            continue
        public_entry[key] = clone_json_compatible(value)

    sanitized_payload: Dict[str, Any] = {}
    payload_preview: Dict[str, str] = {}
    for field in _SENSITIVE_PAYLOAD_FIELDS:
        sanitized_value = _redact_payload_value(entry.get(field))
        sanitized_payload[field] = clone_json_compatible(sanitized_value)
        payload_preview[field] = _summarize_payload_value(
            sanitized_value, limit=summary_limit
        )

    if log_full_payloads:
        public_entry.update({field: sanitized_payload[field] for field in _SENSITIVE_PAYLOAD_FIELDS})
        public_entry["payload"] = dict(sanitized_payload)
    else:
        public_entry.update({field: payload_preview[field] for field in _SENSITIVE_PAYLOAD_FIELDS})
        public_entry["payload"] = None

    public_entry["payload_preview"] = payload_preview
    public_entry["payload_included"] = log_full_payloads
    public_entry["metrics"] = build_tool_metrics(entry)

    return public_entry


def record_tool_activity(
    entry: Dict[str, Any], *, replace: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    stored_entry = {
        **entry,
        "arguments": clone_json_compatible(entry.get("arguments")),
        "result": clone_json_compatible(entry.get("result")),
        "stdout": clone_json_compatible(entry.get("stdout")),
        "stderr": clone_json_compatible(entry.get("stderr")),
    }

    with _tool_activity_lock:
        if replace is not None:
            replace.clear()
            replace.update(stored_entry)
            stored_entry = replace
        else:
            _tool_activity_log.append(stored_entry)

    public_entry = build_public_tool_entry(stored_entry)
    correlation_id = stored_entry.get("tool_call_id") or stored_entry.get("id")
    tracing_metadata = {
        "conversation_id": stored_entry.get("conversation_id"),
        "persona": stored_entry.get("persona"),
        "tool_name": stored_entry.get("tool_name"),
        "status": stored_entry.get("status"),
    }
    tracing = {key: value for key, value in tracing_metadata.items() if value is not None}
    publish_bus_event(
        _TOOL_ACTIVITY_EVENT,
        public_entry,
        priority=MessagePriority.NORMAL,
        correlation_id=correlation_id,
        tracing=tracing or None,
        metadata={"component": "ToolManager"},
    )
    return stored_entry


def record_tool_failure(
    conversation_history,
    user,
    conversation_id,
    *,
    tool_call_id: Optional[str],
    function_name: Optional[str],
    message: str,
    error_type: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    if conversation_history is None or not hasattr(conversation_history, "add_message"):
        return None

    timestamp_value = timestamp or datetime.now()
    metadata: Dict[str, Any] = {"status": "error"}
    if function_name:
        metadata["name"] = function_name
    if error_type:
        metadata["error_type"] = error_type

    entry_kwargs: Dict[str, Any] = {}
    if tool_call_id is not None:
        entry_kwargs["tool_call_id"] = tool_call_id

    conversation_history.add_message(
        role="system",
        content=message,
        metadata=metadata,
        timestamp=timestamp_value,
        user=user,
        conversation_id=conversation_id,
        **entry_kwargs,
    )

    return metadata


def get_tool_activity_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with _tool_activity_lock:
        entries = list(_tool_activity_log)

    if limit is not None and limit >= 0:
        return entries[-limit:]

    return entries


__all__ = [
    "build_public_tool_entry",
    "build_tool_metrics",
    "clone_json_compatible",
    "get_config_manager",
    "get_config_section",
    "get_tool_activity_log",
    "get_tool_logging_preferences",
    "record_tool_activity",
    "record_tool_failure",
    "stringify_tool_value",
]
