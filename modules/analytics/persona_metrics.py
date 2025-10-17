"""Persona analytics storage and aggregation helpers."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "PersonaMetricEvent",
    "PersonaMetricsStore",
    "record_persona_tool_event",
    "record_persona_skill_event",
    "get_persona_metrics",
    "reset_persona_metrics",
]


def _resolve_app_root(config_manager: Optional[Any] = None) -> str:
    """Return the configured application root directory."""

    candidates: List[Optional[str]] = []
    if config_manager is not None:
        getter = getattr(config_manager, "get_app_root", None)
        if callable(getter):
            try:
                candidates.append(getter())
            except Exception:  # pragma: no cover - defensive fall back
                pass
    env_candidate = os.environ.get("ATLAS_APP_ROOT")
    candidates.extend([env_candidate, os.getcwd()])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return os.path.abspath(candidate)
    return os.getcwd()


def _isoformat(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except ValueError:
        return None


@dataclass(frozen=True)
class PersonaMetricEvent:
    """Represents a single persona/tool invocation record."""

    persona: str
    tool: str
    success: bool
    latency_ms: float
    timestamp: datetime
    category: str = "tool"

    def __post_init__(self) -> None:
        normalized_category = str(self.category or "tool").strip().lower()
        if normalized_category not in {"tool", "skill"}:
            normalized_category = "tool"
        object.__setattr__(self, "category", normalized_category)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "persona": self.persona,
            "tool": self.tool,
            "success": self.success,
            "latency_ms": float(self.latency_ms),
            "timestamp": _isoformat(self.timestamp),
            "category": self.category,
        }
        if self.category == "skill":
            payload.setdefault("skill", self.tool)
        return payload


class PersonaMetricsStore:
    """Persist persona/tool metrics and expose aggregation helpers."""

    _default_filename = "persona_metrics.json"

    def __init__(
        self,
        *,
        storage_path: Optional[str] = None,
        app_root: Optional[str] = None,
        config_manager: Optional[Any] = None,
    ) -> None:
        if storage_path is None:
            root = app_root or _resolve_app_root(config_manager)
            storage_path = os.path.join(root, "modules", "analytics", self._default_filename)
        self._storage_path = os.path.abspath(storage_path)
        self._lock = threading.RLock()
        directory = os.path.dirname(self._storage_path)
        os.makedirs(directory, exist_ok=True)

    # ------------------------ Mutation helpers ------------------------

    def record_event(self, event: PersonaMetricEvent) -> None:
        """Append ``event`` to the persisted metrics log."""

        with self._lock:
            payload = self._load_payload()
            events = payload.setdefault("events", [])
            events.append(event.as_dict())
            self._write_payload(payload)

    def reset(self) -> None:
        """Remove all recorded metrics."""

        with self._lock:
            payload = {"events": []}
            self._write_payload(payload)

    # ------------------------ Aggregation helpers ------------------------

    def get_metrics(
        self,
        persona: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit_recent: int = 20,
    ) -> Dict[str, Any]:
        """Return aggregated metrics for ``persona`` within an optional window."""

        with self._lock:
            payload = self._load_payload()
        raw_events = payload.get("events") or []
        filtered: List[Tuple[Dict[str, Any], Optional[datetime]]] = []
        for item in raw_events:
            if not isinstance(item, Mapping):
                continue
            if item.get("persona") != persona:
                continue
            timestamp = _parse_iso(str(item.get("timestamp")))
            if start and (timestamp is None or timestamp < start):
                continue
            if end and (timestamp is None or timestamp > end):
                continue
            filtered.append((self._normalize_event_dict(item), timestamp))

        tool_metrics = self._aggregate_category(
            filtered,
            category="tool",
            name_key="tool",
            label="tool",
            limit_recent=limit_recent,
        )
        skill_metrics = self._aggregate_category(
            filtered,
            category="skill",
            name_key="skill",
            label="skill",
            limit_recent=limit_recent,
        )

        result = {
            "persona": persona,
            "window": {
                "start": _isoformat(start) if start else None,
                "end": _isoformat(end) if end else None,
            },
        }
        result.update(tool_metrics)
        result["skills"] = skill_metrics
        return result

    def _normalize_event_dict(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        event = dict(payload)
        category = str(event.get("category") or "tool").strip().lower()
        if category not in {"tool", "skill"}:
            category = "tool"
        event["category"] = category
        event.setdefault("persona", str(payload.get("persona") or ""))
        tool_name = str(event.get("tool") or "").strip()
        if category == "skill":
            skill_name = str(event.get("skill") or tool_name).strip()
            event["skill"] = skill_name
            event["tool"] = tool_name or skill_name
        else:
            event["tool"] = tool_name
        return event

    def _aggregate_category(
        self,
        events: List[Tuple[Dict[str, Any], Optional[datetime]]],
        *,
        category: str,
        name_key: str,
        label: str,
        limit_recent: int,
    ) -> Dict[str, Any]:
        category_events = [
            (event, timestamp)
            for event, timestamp in events
            if (event.get("category") or "tool") == category
        ]

        total_calls = len(category_events)
        success_count = sum(1 for event, _ in category_events if bool(event.get("success")))
        failure_count = total_calls - success_count
        latency_sum = sum(
            float(event.get("latency_ms") or 0.0) for event, _ in category_events
        )
        average_latency = latency_sum / total_calls if total_calls else 0.0

        totals_by_name: Dict[str, Dict[str, Any]] = {}
        for event, _ in category_events:
            target_name = str(event.get(name_key) or "").strip()
            if not target_name:
                continue
            bucket = totals_by_name.setdefault(
                target_name,
                {label: target_name, "calls": 0, "success": 0, "failure": 0},
            )
            bucket["calls"] += 1
            if bool(event.get("success")):
                bucket["success"] += 1
            else:
                bucket["failure"] += 1

        for bucket in totals_by_name.values():
            calls = bucket["calls"]
            successes = bucket["success"]
            bucket["success_rate"] = successes / calls if calls else 0.0

        category_events.sort(key=lambda pair: pair[1] or datetime.min)
        recent_pairs = category_events[-limit_recent:]
        recent = [
            {
                "persona": event.get("persona"),
                label: event.get(name_key),
                "success": bool(event.get("success")),
                "latency_ms": float(event.get("latency_ms") or 0.0),
                "timestamp": event.get("timestamp"),
                "category": category,
            }
            for event, _ in reversed(recent_pairs)
        ]

        return {
            "category": category,
            "totals": {
                "calls": total_calls,
                "success": success_count,
                "failure": failure_count,
            },
            "success_rate": success_count / total_calls if total_calls else 0.0,
            "average_latency_ms": average_latency,
            f"totals_by_{label}": sorted(
                totals_by_name.values(), key=lambda bucket: bucket[label]
            ),
            "recent": recent,
        }

    # ------------------------ Internal helpers ------------------------

    def _load_payload(self) -> Dict[str, Any]:
        if not os.path.exists(self._storage_path):
            return {"events": []}
        try:
            with open(self._storage_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {"events": []}

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        tmp_path = f"{self._storage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._storage_path)


_default_store: Optional[PersonaMetricsStore] = None
_store_lock = threading.Lock()


def _get_store(config_manager: Optional[Any] = None) -> PersonaMetricsStore:
    global _default_store
    app_root = _resolve_app_root(config_manager)
    expected_directory = os.path.join(app_root, "modules", "analytics")
    with _store_lock:
        current_directory = (
            os.path.dirname(getattr(_default_store, "_storage_path", ""))
            if _default_store is not None
            else None
        )
        if _default_store is None or current_directory != expected_directory:
            _default_store = PersonaMetricsStore(config_manager=config_manager, app_root=app_root)
        return _default_store


def record_persona_tool_event(
    persona: Optional[str],
    tool: Optional[str],
    *,
    success: bool,
    latency_ms: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    config_manager: Optional[Any] = None,
) -> None:
    """Persist a persona/tool invocation metric."""

    if not persona or not tool:
        return
    event = PersonaMetricEvent(
        persona=str(persona),
        tool=str(tool),
        success=bool(success),
        latency_ms=float(latency_ms or 0.0),
        timestamp=timestamp or datetime.now(timezone.utc),
    )
    store = _get_store(config_manager)
    store.record_event(event)


def record_persona_skill_event(
    persona: Optional[str],
    skill: Optional[str],
    *,
    success: bool,
    latency_ms: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    config_manager: Optional[Any] = None,
) -> None:
    """Persist a persona/skill invocation metric."""

    if not persona or not skill:
        return
    event = PersonaMetricEvent(
        persona=str(persona),
        tool=str(skill),
        success=bool(success),
        latency_ms=float(latency_ms or 0.0),
        timestamp=timestamp or datetime.now(timezone.utc),
        category="skill",
    )
    store = _get_store(config_manager)
    store.record_event(event)


def get_persona_metrics(
    persona: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit_recent: int = 20,
    config_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return aggregated metrics for ``persona`` from the shared store."""

    store = _get_store(config_manager)
    return store.get_metrics(persona, start=start, end=end, limit_recent=limit_recent)


def reset_persona_metrics(*, config_manager: Optional[Any] = None) -> None:
    """Clear recorded persona metrics in the shared store."""

    store = _get_store(config_manager)
    store.reset()
