"""Persona analytics storage and aggregation helpers."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from modules.Tools.tool_event_system import publish_bus_event
from modules.orchestration.message_bus import MessagePriority

__all__ = [
    "PersonaMetricEvent",
    "PersonaMetricsStore",
    "LifecycleEvent",
    "record_persona_tool_event",
    "record_persona_skill_event",
    "record_task_lifecycle_event",
    "record_job_lifecycle_event",
    "get_persona_metrics",
    "get_persona_comparison_summary",
    "get_task_lifecycle_metrics",
    "get_job_lifecycle_metrics",
    "reset_persona_metrics",
    "reset_task_metrics",
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


@dataclass(frozen=True)
class LifecycleEvent:
    """Represents a lifecycle transition for a task, job, or similar entity."""

    entity_id: str
    entity_key: str
    event: str
    persona: Optional[str] = None
    tenant_id: Optional[str] = None
    from_status: Optional[str] = None
    to_status: Optional[str] = None
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Mapping[str, Any]] = None
    extra: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        event_name = str(self.event or "").strip().lower()
        object.__setattr__(self, "event", event_name or "status_changed")
        metadata: Mapping[str, Any]
        if isinstance(self.metadata, Mapping):
            metadata = self.metadata
        else:
            metadata = {}
        object.__setattr__(self, "metadata", dict(metadata))

        extras: Mapping[str, Any]
        if isinstance(self.extra, Mapping):
            extras = self.extra
        else:
            extras = {}
        object.__setattr__(self, "extra", dict(extras))
        entity_key = str(self.entity_key or "").strip() or "entity_id"
        object.__setattr__(self, "entity_key", entity_key)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            self.entity_key: self.entity_id,
            "event": self.event,
            "persona": self.persona,
            "tenant_id": self.tenant_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "success": self.success,
            "latency_ms": float(self.latency_ms) if self.latency_ms is not None else None,
            "timestamp": _isoformat(self.timestamp),
            "metadata": dict(self.metadata or {}),
        }
        payload.update(dict(self.extra or {}))
        return payload


class PersonaMetricsStore:
    """Persist persona/tool metrics and expose aggregation helpers."""

    _default_filename = "persona_metrics.json"
    _baseline_alpha = 0.3
    _baseline_min_points = 5
    _baseline_z_threshold = 3.0
    _baseline_min_stddev = 1e-3
    _max_anomalies = 200

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
            payload = {
                "events": [],
                "task_events": [],
                "job_events": [],
                "baselines": {},
                "anomalies": [],
            }
            self._write_payload(payload)

    def record_task_event(self, event: LifecycleEvent) -> None:
        """Append ``event`` to the persisted task lifecycle log."""

        with self._lock:
            payload = self._load_payload()
            events = payload.setdefault("task_events", [])
            events.append(event.as_dict())
            self._write_payload(payload)

    def reset_task_metrics(self) -> None:
        """Remove all recorded task lifecycle metrics."""

        with self._lock:
            payload = self._load_payload()
            payload.setdefault("events", payload.get("events", []))
            payload["task_events"] = []
            payload.setdefault("job_events", payload.get("job_events", []))
            payload.setdefault("baselines", payload.get("baselines", {}))
            payload.setdefault("anomalies", payload.get("anomalies", []))
            self._write_payload(payload)

    def record_job_event(self, event: LifecycleEvent) -> None:
        """Append ``event`` to the persisted job lifecycle log."""

        with self._lock:
            payload = self._load_payload()
            events = payload.setdefault("job_events", [])
            events.append(event.as_dict())
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

        alerts_to_dispatch: List[Dict[str, Any]] = []

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

            payload_dirty = False
            for category_name, metrics in (
                ("tool", tool_metrics),
                ("skill", skill_metrics),
            ):
                detected, updated = self._evaluate_category_anomalies(
                    payload,
                    persona,
                    category_name,
                    metrics,
                )
                if detected:
                    alerts_to_dispatch.extend(detected)
                if updated:
                    payload_dirty = True

            persona_anomalies = self._collect_recent_anomalies(payload, persona)

            tool_anomalies = [
                dict(item)
                for item in persona_anomalies
                if (item.get("category") or "").lower() == "tool"
            ]
            skill_anomalies = [
                dict(item)
                for item in persona_anomalies
                if (item.get("category") or "").lower() == "skill"
            ]

            tool_metrics["anomalies"] = tool_anomalies
            tool_metrics["recent_anomalies"] = tool_anomalies
            skill_metrics["anomalies"] = skill_anomalies
            skill_metrics["recent_anomalies"] = skill_anomalies

            result = {
                "persona": persona,
                "window": {
                    "start": _isoformat(start) if start else None,
                    "end": _isoformat(end) if end else None,
                },
            }
            result.update(tool_metrics)
            result["anomalies"] = tool_anomalies
            result["recent_anomalies"] = persona_anomalies
            result["skills"] = skill_metrics

            if payload_dirty:
                self._write_payload(payload)

        for alert in alerts_to_dispatch:
            _dispatch_persona_metric_alert(alert)

        return result

    def get_persona_comparison_summary(
        self,
        *,
        category: str = "tool",
        personas: Optional[Iterable[str]] = None,
        search: Optional[str] = None,
        limit_recent: int = 5,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        """Return a consolidated comparison payload across personas.

        Parameters
        ----------
        category:
            Aggregate either ``"tool"`` or ``"skill"`` events when building
            per-persona summaries.
        personas:
            Optional iterable of persona identifiers to include. Matching is
            case-insensitive and trimmed; when omitted all personas are
            considered.
        search:
            Optional substring filter (case-insensitive) applied to persona
            names after normalization.
        limit_recent:
            Number of recent executions to retain per persona in the returned
            payload. This value is clamped between 1 and 200.
        page / page_size:
            Pagination controls applied to the detailed ``results`` section.
        """

        normalized_category = str(category or "tool").strip().lower()
        if normalized_category not in {"tool", "skill"}:
            normalized_category = "tool"

        try:
            recent_limit = int(limit_recent)
        except (TypeError, ValueError):
            recent_limit = 5
        recent_limit = max(1, min(recent_limit, 200))

        try:
            page_number = int(page)
        except (TypeError, ValueError):
            page_number = 1
        page_number = max(page_number, 1)

        try:
            size_value = int(page_size)
        except (TypeError, ValueError):
            size_value = 25
        size_value = max(1, min(size_value, 200))

        persona_filters: Optional[set[str]] = None
        if personas is not None:
            persona_filters = {
                str(value).strip().lower()
                for value in personas
                if isinstance(value, str) and str(value).strip()
            }
            if not persona_filters:
                persona_filters = None

        search_token = str(search or "").strip().lower()
        if not search_token:
            search_token = ""

        with self._lock:
            payload = self._load_payload()

        raw_events = payload.get("events") or []
        persona_buckets: Dict[str, List[Tuple[Dict[str, Any], Optional[datetime]]]] = {}
        for item in raw_events:
            if not isinstance(item, Mapping):
                continue
            persona_name = str(item.get("persona") or "").strip()
            if not persona_name:
                continue
            persona_token = persona_name.lower()
            if persona_filters is not None and persona_token not in persona_filters:
                continue
            if search_token and search_token not in persona_token:
                continue
            event = self._normalize_event_dict(item)
            timestamp = _parse_iso(event.get("timestamp"))
            persona_buckets.setdefault(persona_name, []).append((event, timestamp))

        name_key = "skill" if normalized_category == "skill" else "tool"
        label = name_key

        persona_summaries: List[Dict[str, Any]] = []
        for persona_name in sorted(persona_buckets):
            events = persona_buckets[persona_name]
            metrics = self._aggregate_category(
                events,
                category=normalized_category,
                name_key=name_key,
                label=label,
                limit_recent=recent_limit,
            )
            totals = metrics.get("totals") if isinstance(metrics, Mapping) else {}
            if not isinstance(totals, Mapping):
                totals = {}
            calls = int(totals.get("calls") or 0)
            success_count = int(totals.get("success") or 0)
            failure_count = int(totals.get("failure") or 0)
            failure_rate = failure_count / calls if calls else 0.0

            recent_entries = metrics.get("recent")
            if not isinstance(recent_entries, list):
                recent_entries = []

            breakdown_key = f"totals_by_{label}"
            breakdown_entries = metrics.get(breakdown_key)
            if not isinstance(breakdown_entries, list):
                breakdown_entries = []

            persona_summaries.append(
                {
                    "persona": persona_name,
                    "category": normalized_category,
                    "totals": {
                        "calls": calls,
                        "success": success_count,
                        "failure": failure_count,
                    },
                    "success_rate": (
                        float(metrics.get("success_rate") or 0.0)
                    ),
                    "failure_rate": failure_rate,
                    "average_latency_ms": float(
                        metrics.get("average_latency_ms") or 0.0
                    ),
                    breakdown_key: breakdown_entries,
                    "recent": recent_entries[:recent_limit],
                }
            )

        total_results = len(persona_summaries)
        total_pages = math.ceil(total_results / size_value) if total_results else 0
        if total_pages:
            page_number = min(page_number, total_pages)
        start_index = (page_number - 1) * size_value
        end_index = start_index + size_value
        paginated_results = persona_summaries[start_index:end_index]

        def _ranking_projection(entry: Mapping[str, Any]) -> Dict[str, Any]:
            totals_map = entry.get("totals") if isinstance(entry, Mapping) else {}
            if not isinstance(totals_map, Mapping):
                totals_map = {}
            calls_value = int(totals_map.get("calls") or 0)
            success_value = int(totals_map.get("success") or 0)
            failure_value = int(totals_map.get("failure") or 0)
            return {
                "persona": entry.get("persona"),
                "success_rate": float(entry.get("success_rate") or 0.0),
                "failure_rate": float(entry.get("failure_rate") or 0.0),
                "average_latency_ms": float(
                    entry.get("average_latency_ms") or 0.0
                ),
                "calls": calls_value,
                "success": success_value,
                "failure": failure_value,
            }

        active_entries = [
            item for item in persona_summaries if int(item["totals"]["calls"]) > 0
        ]

        top_performers = sorted(
            (_ranking_projection(item) for item in active_entries),
            key=lambda data: (data["success_rate"], data["calls"]),
            reverse=True,
        )[:5]

        worst_failure = sorted(
            (_ranking_projection(item) for item in active_entries),
            key=lambda data: (data["failure_rate"], data["calls"]),
            reverse=True,
        )[:5]

        latency_candidates = [
            projection
            for projection in (_ranking_projection(item) for item in active_entries)
            if projection["calls"] > 0
        ]

        fastest_latency = sorted(
            latency_candidates,
            key=lambda data: data["average_latency_ms"],
        )[:5]

        slowest_latency = sorted(
            latency_candidates,
            key=lambda data: data["average_latency_ms"],
            reverse=True,
        )[:5]

        return {
            "category": normalized_category,
            "results": paginated_results,
            "pagination": {
                "page": page_number,
                "page_size": size_value,
                "total": total_results,
                "total_pages": total_pages,
            },
            "rankings": {
                "top_performers": top_performers,
                "worst_failure_rates": worst_failure,
                "fastest_latency": fastest_latency,
                "slowest_latency": slowest_latency,
            },
        }

    def get_task_metrics(
        self,
        *,
        persona: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit_recent: int = 50,
    ) -> Dict[str, Any]:
        """Return aggregated lifecycle metrics filtered by persona/tenant."""

        persona_key = (persona or "").strip().lower() or None
        tenant_key = (tenant_id or "").strip().lower() or None

        with self._lock:
            payload = self._load_payload()

        raw_events = payload.get("task_events") or []
        filtered: List[Tuple[Dict[str, Any], Optional[datetime]]] = []

        for item in raw_events:
            if not isinstance(item, Mapping):
                continue
            event = self._normalize_task_event(item)
            event_persona = (event.get("persona") or "").strip().lower() or None
            event_tenant = (event.get("tenant_id") or "").strip().lower() or None
            if persona_key is not None and event_persona != persona_key:
                continue
            if tenant_key is not None and event_tenant != tenant_key:
                continue
            timestamp = _parse_iso(event.get("timestamp"))
            filtered.append((event, timestamp))

        total_events = len(filtered)
        success_events = [event for event, _ in filtered if event.get("success") is not None]
        successes = sum(1 for event in success_events if bool(event.get("success")))
        failures = sum(1 for event in success_events if event.get("success") is False)
        denominator = len(success_events)
        success_rate = successes / denominator if denominator else 0.0

        latencies = [
            float(event.get("latency_ms"))
            for event, _ in filtered
            if event.get("latency_ms") is not None
        ]
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0

        reassignment_total = sum(int(event.get("reassignments") or 0) for event, _ in filtered)

        status_totals: Dict[str, Dict[str, Any]] = {}
        task_totals: Dict[str, Dict[str, Any]] = {}
        task_sequences: Dict[str, List[Tuple[Optional[datetime], int, Dict[str, Any]]]] = {}
        for index, (event, timestamp) in enumerate(filtered):
            status = str(event.get("to_status") or "").strip().lower() or "unknown"
            status_bucket = status_totals.setdefault(
                status,
                {
                    "status": status,
                    "events": 0,
                    "success": 0,
                    "failure": 0,
                },
            )
            status_bucket["events"] += 1
            if event.get("success") is True:
                status_bucket["success"] += 1
            elif event.get("success") is False:
                status_bucket["failure"] += 1

            task_id = str(event.get("task_id") or "").strip() or "unknown"
            task_bucket = task_totals.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "events": 0,
                    "success": 0,
                    "failure": 0,
                    "reassignments": 0,
                    "last_status": None,
                    "last_timestamp": None,
                },
            )
            task_bucket["events"] += 1
            if event.get("success") is True:
                task_bucket["success"] += 1
            elif event.get("success") is False:
                task_bucket["failure"] += 1
            task_bucket["reassignments"] += int(event.get("reassignments") or 0)
            if timestamp is not None:
                task_bucket["last_status"] = status
                task_bucket["last_timestamp"] = _isoformat(timestamp)

            task_sequences.setdefault(task_id, []).append((timestamp, index, event))

        status_summary = sorted(status_totals.values(), key=lambda item: item["status"])
        task_summary = sorted(task_totals.values(), key=lambda item: item["task_id"])

        funnel_summary = self._summarize_task_funnel(task_sequences)

        filtered.sort(key=lambda pair: pair[1] or datetime.min)
        recent_pairs = filtered[-limit_recent:]
        recent_events = [
            {
                "task_id": event.get("task_id"),
                "event": event.get("event"),
                "persona": event.get("persona"),
                "tenant_id": event.get("tenant_id"),
                "from_status": event.get("from_status"),
                "to_status": event.get("to_status"),
                "success": event.get("success"),
                "latency_ms": event.get("latency_ms"),
                "reassignments": event.get("reassignments"),
                "timestamp": event.get("timestamp"),
                "metadata": dict(event.get("metadata") or {}),
            }
            for event, _ in reversed(recent_pairs)
        ]

        return {
            "persona": persona,
            "tenant_id": tenant_id,
            "totals": {
                "events": total_events,
                "success_events": successes,
                "failure_events": failures,
                "reassignments": reassignment_total,
            },
            "success_rate": success_rate,
            "average_latency_ms": average_latency,
            "status_totals": status_summary,
            "tasks": task_summary,
            "recent": recent_events,
            "funnel": funnel_summary,
        }

    def get_job_metrics(
        self,
        *,
        persona: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit_recent: int = 50,
    ) -> Dict[str, Any]:
        """Return aggregated job lifecycle metrics filtered by persona/tenant."""

        persona_key = (persona or "").strip().lower() or None
        tenant_key = (tenant_id or "").strip().lower() or None

        with self._lock:
            payload = self._load_payload()

        raw_events = payload.get("job_events") or []
        filtered: List[Tuple[Dict[str, Any], Optional[datetime]]] = []

        for item in raw_events:
            if not isinstance(item, Mapping):
                continue
            event = self._normalize_job_event(item)
            event_persona = (event.get("persona") or "").strip().lower() or None
            event_tenant = (event.get("tenant_id") or "").strip().lower() or None
            if persona_key is not None and event_persona != persona_key:
                continue
            if tenant_key is not None and event_tenant != tenant_key:
                continue
            timestamp = _parse_iso(event.get("timestamp"))
            filtered.append((event, timestamp))

        total_events = len(filtered)
        success_events = [event for event, _ in filtered if event.get("success") is not None]
        successes = sum(1 for event in success_events if bool(event.get("success")))
        failures = sum(1 for event in success_events if event.get("success") is False)
        denominator = len(success_events)
        success_rate = successes / denominator if denominator else 0.0

        latencies = [
            float(event.get("latency_ms"))
            for event, _ in filtered
            if event.get("latency_ms") is not None
        ]
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0

        status_totals: Dict[str, Dict[str, Any]] = {}
        job_totals: Dict[str, Dict[str, Any]] = {}
        completion_timestamps: List[datetime] = []
        sla_checks = 0
        sla_breaches = 0

        for event, timestamp in filtered:
            status = str(event.get("to_status") or "").strip().lower() or "unknown"
            status_bucket = status_totals.setdefault(
                status,
                {
                    "status": status,
                    "events": 0,
                    "success": 0,
                    "failure": 0,
                },
            )
            status_bucket["events"] += 1
            if event.get("success") is True:
                status_bucket["success"] += 1
            elif event.get("success") is False:
                status_bucket["failure"] += 1

            job_id = str(event.get("job_id") or "").strip() or "unknown"
            job_bucket = job_totals.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "events": 0,
                    "success": 0,
                    "failure": 0,
                    "last_status": None,
                    "last_timestamp": None,
                },
            )
            job_bucket["events"] += 1
            if event.get("success") is True:
                job_bucket["success"] += 1
            elif event.get("success") is False:
                job_bucket["failure"] += 1
            if timestamp is not None:
                job_bucket["last_status"] = status
                job_bucket["last_timestamp"] = _isoformat(timestamp)

            metadata = event.get("metadata") or {}
            if isinstance(metadata, Mapping):
                sla_flag = False
                has_sla_metadata = False
                if "sla_breached" in metadata:
                    has_sla_metadata = True
                    sla_flag = bool(metadata.get("sla_breached"))
                if "sla_breach" in metadata:
                    has_sla_metadata = True
                    sla_flag = sla_flag or bool(metadata.get("sla_breach"))
                if "sla_met" in metadata:
                    has_sla_metadata = True
                    sla_flag = sla_flag or metadata.get("sla_met") is False
                if has_sla_metadata:
                    sla_checks += 1
                    if sla_flag:
                        sla_breaches += 1

            if (
                timestamp is not None
                and event.get("event") in {"completed", "failed", "cancelled"}
            ):
                completion_timestamps.append(timestamp)

        status_summary = sorted(status_totals.values(), key=lambda item: item["status"])
        job_summary = sorted(job_totals.values(), key=lambda item: item["job_id"])

        filtered.sort(key=lambda pair: pair[1] or datetime.min)
        recent_pairs = filtered[-limit_recent:]
        recent_events = [
            {
                "job_id": event.get("job_id"),
                "event": event.get("event"),
                "persona": event.get("persona"),
                "tenant_id": event.get("tenant_id"),
                "from_status": event.get("from_status"),
                "to_status": event.get("to_status"),
                "success": event.get("success"),
                "latency_ms": event.get("latency_ms"),
                "timestamp": event.get("timestamp"),
                "metadata": dict(event.get("metadata") or {}),
            }
            for event, _ in reversed(recent_pairs)
        ]

        throughput_per_hour = 0.0
        if completion_timestamps:
            completion_timestamps.sort()
            if len(completion_timestamps) == 1:
                throughput_per_hour = 1.0
            else:
                window_seconds = (
                    completion_timestamps[-1] - completion_timestamps[0]
                ).total_seconds()
                if window_seconds <= 0:
                    throughput_per_hour = float(len(completion_timestamps))
                else:
                    throughput_per_hour = (
                        len(completion_timestamps) / (window_seconds / 3600.0)
                    )

        sla_adherence = 1.0
        if sla_checks:
            sla_adherence = (sla_checks - sla_breaches) / sla_checks

        return {
            "persona": persona,
            "tenant_id": tenant_id,
            "totals": {
                "events": total_events,
                "success_events": successes,
                "failure_events": failures,
            },
            "success_rate": success_rate,
            "average_latency_ms": average_latency,
            "status_totals": status_summary,
            "jobs": job_summary,
            "recent": recent_events,
            "throughput_per_hour": throughput_per_hour,
            "sla": {
                "checks": sla_checks,
                "breaches": sla_breaches,
                "adherence_rate": sla_adherence if sla_checks else None,
            },
        }

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

    def _normalize_job_event(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        event = dict(payload)
        event.setdefault("persona", str(payload.get("persona") or ""))
        event.setdefault("tenant_id", payload.get("tenant_id"))
        event.setdefault("job_id", payload.get("job_id"))
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        event["metadata"] = dict(metadata)
        event_name = str(event.get("event") or "").strip().lower()
        event["event"] = event_name or "status_changed"
        timestamp = event.get("timestamp")
        if isinstance(timestamp, datetime):
            event["timestamp"] = _isoformat(timestamp)
        return event

    def _normalize_task_event(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        event = dict(payload)
        event.setdefault("task_id", str(payload.get("task_id") or ""))
        event.setdefault("event", str(payload.get("event") or "status_changed"))
        event.setdefault("persona", payload.get("persona"))
        event.setdefault("tenant_id", payload.get("tenant_id"))
        event.setdefault("from_status", payload.get("from_status"))
        event.setdefault("to_status", payload.get("to_status"))
        event.setdefault("success", payload.get("success"))
        latency = payload.get("latency_ms")
        if latency is None:
            event["latency_ms"] = None
        else:
            try:
                event["latency_ms"] = float(latency)
            except (TypeError, ValueError):
                event["latency_ms"] = None
        try:
            event["reassignments"] = int(payload.get("reassignments") or 0)
        except (TypeError, ValueError):
            event["reassignments"] = 0
        timestamp = payload.get("timestamp")
        if isinstance(timestamp, datetime):
            event["timestamp"] = _isoformat(timestamp)
        else:
            event["timestamp"] = str(timestamp) if timestamp is not None else None
        metadata = payload.get("metadata")
        event["metadata"] = metadata if isinstance(metadata, Mapping) else {}
        return event

    def _summarize_task_funnel(
        self,
        sequences: Mapping[str, List[Tuple[Optional[datetime], int, Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        stage_stats: Dict[str, Dict[str, Any]] = {}
        stage_seen: Dict[str, Dict[str, Any]] = {}

        order_counter = 0
        for task_id, records in sequences.items():
            if not records:
                continue

            records = sorted(
                records,
                key=lambda entry: (
                    entry[0] is None,
                    entry[0] or datetime.min.replace(tzinfo=timezone.utc),
                    entry[1],
                ),
            )

            stage_entries: List[Dict[str, Any]] = []
            for timestamp, _, event in records:
                status = str(event.get("to_status") or "").strip().lower() or "unknown"
                success_value = event.get("success")
                normalized_success: Optional[bool]
                if success_value is None:
                    normalized_success = None
                else:
                    normalized_success = bool(success_value)

                if not stage_entries or stage_entries[-1]["status"] != status:
                    stage_entries.append(
                        {
                            "status": status,
                            "entry_time": timestamp,
                            "success": normalized_success,
                        }
                    )
                else:
                    entry = stage_entries[-1]
                    if entry.get("entry_time") is None and timestamp is not None:
                        entry["entry_time"] = timestamp
                    if normalized_success is not None:
                        entry["success"] = normalized_success

                if status not in stage_seen:
                    stage_seen[status] = {
                        "order": order_counter,
                        "first_timestamp": timestamp,
                    }
                    order_counter += 1
                elif timestamp is not None:
                    first_timestamp = stage_seen[status].get("first_timestamp")
                    if first_timestamp is None or timestamp < first_timestamp:
                        stage_seen[status]["first_timestamp"] = timestamp

            for index, entry in enumerate(stage_entries):
                status = entry["status"]
                stage_bucket = stage_stats.setdefault(
                    status,
                    {
                        "entered": set(),
                        "converted": set(),
                        "abandoned": set(),
                        "durations": [],
                    },
                )
                stage_bucket["entered"].add(task_id)

                next_entry = stage_entries[index + 1] if index + 1 < len(stage_entries) else None
                entry_time = entry.get("entry_time")

                if next_entry is not None:
                    stage_bucket["converted"].add(task_id)
                    next_time = next_entry.get("entry_time")
                    if (
                        entry_time is not None
                        and next_time is not None
                        and next_time >= entry_time
                    ):
                        duration = (next_time - entry_time).total_seconds() * 1000.0
                        stage_bucket["durations"].append(duration)
                    continue

                success_value = entry.get("success")
                if success_value is True:
                    stage_bucket["converted"].add(task_id)
                else:
                    stage_bucket["abandoned"].add(task_id)

        stages: List[Dict[str, Any]] = []

        for status, bucket in stage_stats.items():
            entered_count = len(bucket["entered"])
            converted_count = len(bucket["converted"])
            abandoned_count = len(bucket["abandoned"])
            durations = bucket["durations"]
            average_time = (
                sum(durations) / len(durations)
                if durations
                else None
            )

            stage_metadata = stage_seen.get(status, {"order": len(stage_seen)})
            first_timestamp = stage_metadata.get("first_timestamp")
            order_value = stage_metadata.get("order", 0)

            stages.append(
                {
                    "status": status,
                    "entered": entered_count,
                    "converted": converted_count,
                    "abandoned": abandoned_count,
                    "conversion_rate": converted_count / entered_count if entered_count else 0.0,
                    "abandonment_rate": abandoned_count / entered_count if entered_count else 0.0,
                    "average_time_ms": average_time,
                    "samples": len(durations),
                    "_order": (
                        first_timestamp is None,
                        first_timestamp
                        or datetime.max.replace(tzinfo=timezone.utc),
                        order_value,
                    ),
                }
            )

        stages.sort(key=lambda item: item.pop("_order"))

        return {"stages": stages}

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
        latencies = [
            float(event.get("latency_ms"))
            for event, _ in category_events
            if event.get("latency_ms") is not None
        ]
        average_latency = sum(latencies) / len(latencies) if latencies else 0.0
        latency_samples = len(latencies)

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
            "latency_samples": latency_samples,
            f"totals_by_{label}": sorted(
                totals_by_name.values(), key=lambda bucket: bucket[label]
            ),
            "recent": recent,
        }

    # ------------------------ Internal helpers ------------------------

    def _load_payload(self) -> Dict[str, Any]:
        if not os.path.exists(self._storage_path):
            return {
                "events": [],
                "task_events": [],
                "job_events": [],
                "baselines": {},
                "anomalies": [],
            }
        try:
            with open(self._storage_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            return {
                "events": [],
                "task_events": [],
                "job_events": [],
                "baselines": {},
                "anomalies": [],
            }

        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("events", [])
        payload.setdefault("task_events", [])
        payload.setdefault("job_events", [])
        payload.setdefault("baselines", {})
        payload.setdefault("anomalies", [])
        return payload

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        tmp_path = f"{self._storage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._storage_path)

    def _evaluate_category_anomalies(
        self,
        payload: Dict[str, Any],
        persona: str,
        category: str,
        metrics: Mapping[str, Any],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        totals = metrics.get("totals") if isinstance(metrics, Mapping) else {}
        if not isinstance(totals, Mapping):
            totals = {}

        calls = int(totals.get("calls") or 0)
        if calls <= 0:
            return ([], False)

        triggered: List[Dict[str, Any]] = []
        payload_dirty = False

        failures = int(totals.get("failure") or 0)
        failure_rate = failures / calls if calls else 0.0
        failure_actions = [
            "Inspect recent failures for errors or regressions.",
            "Consider disabling problematic tools until stability improves.",
        ]
        anomaly, updated = self._check_metric_anomaly(
            payload,
            persona,
            category,
            "failure_rate",
            failure_rate,
            failure_actions,
        )
        if updated:
            payload_dirty = True
        if anomaly:
            if self._record_anomaly(payload, anomaly):
                payload_dirty = True
            triggered.append(anomaly)

        latency_samples = int(metrics.get("latency_samples") or 0)
        if latency_samples > 0:
            latency_value = float(metrics.get("average_latency_ms") or 0.0)
            latency_actions = [
                "Review recent latency spikes and adjust scaling or timeouts.",
                "Audit upstream providers for degraded performance.",
            ]
            anomaly, updated = self._check_metric_anomaly(
                payload,
                persona,
                category,
                "latency_ms",
                latency_value,
                latency_actions,
            )
            if updated:
                payload_dirty = True
            if anomaly:
                if self._record_anomaly(payload, anomaly):
                    payload_dirty = True
                triggered.append(anomaly)

        return (triggered, payload_dirty)

    def _check_metric_anomaly(
        self,
        payload: Dict[str, Any],
        persona: str,
        category: str,
        metric: str,
        value: float,
        actions: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not math.isfinite(value):
            return (None, False)

        baseline_bucket = self._get_baseline_bucket(payload, persona, category, metric)
        count = int(baseline_bucket.get("count") or 0)
        mean = float(baseline_bucket.get("mean") or 0.0)
        variance = float(baseline_bucket.get("variance") or 0.0)

        anomaly: Optional[Dict[str, Any]] = None
        stddev = math.sqrt(max(variance, self._baseline_min_stddev ** 2))
        if count >= self._baseline_min_points:
            z_score = abs(value - mean) / stddev if stddev > 0 else 0.0
            if z_score >= self._baseline_z_threshold:
                anomaly = {
                    "persona": persona,
                    "category": category,
                    "metric": f"{category}.{metric}",
                    "observed": value,
                    "baseline": {
                        "mean": mean,
                        "stddev": stddev,
                        "z_score": z_score,
                        "threshold_z": self._baseline_z_threshold,
                    },
                    "suggested_actions": list(actions),
                    "timestamp": _isoformat(datetime.now(timezone.utc)),
                }

        # Update EWMA baseline after evaluating anomaly status.
        if count == 0:
            new_mean = value
            new_variance = max(self._baseline_min_stddev ** 2, 0.0)
        else:
            delta = value - mean
            new_mean = mean + self._baseline_alpha * delta
            new_variance = (1 - self._baseline_alpha) * (
                variance + self._baseline_alpha * (delta ** 2)
            )
            new_variance = max(new_variance, self._baseline_min_stddev ** 2)

        baseline_bucket.update(
            {
                "mean": new_mean,
                "variance": new_variance,
                "count": count + 1,
                "last_observed": value,
                "updated_at": _isoformat(datetime.now(timezone.utc)),
            }
        )

        return (anomaly, True)

    def _get_baseline_bucket(
        self,
        payload: Dict[str, Any],
        persona: str,
        category: str,
        metric: str,
    ) -> Dict[str, Any]:
        baselines = payload.setdefault("baselines", {})
        persona_bucket = baselines.setdefault(persona, {})
        category_bucket = persona_bucket.setdefault(category, {})
        metric_bucket = category_bucket.setdefault(metric, {})
        return metric_bucket

    def _record_anomaly(self, payload: Dict[str, Any], anomaly: Dict[str, Any]) -> bool:
        anomalies = payload.setdefault("anomalies", [])
        anomalies.append(anomaly)
        if len(anomalies) > self._max_anomalies:
            del anomalies[:-self._max_anomalies]
        return True

    def _collect_recent_anomalies(
        self, payload: Mapping[str, Any], persona: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        entries = payload.get("anomalies") if isinstance(payload, Mapping) else []
        if not isinstance(entries, list):
            return []
        filtered = [
            dict(item)
            for item in entries
            if isinstance(item, Mapping) and item.get("persona") == persona
        ]
        recent = filtered[-limit:]
        recent.reverse()
        return recent


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
    publish_bus_event(
        "persona_metrics.tool",
        event.as_dict(),
        priority=MessagePriority.LOW,
        correlation_id=f"{event.persona}:{event.tool}",
        tracing={
            "persona": event.persona,
            "category": event.category,
            "success": event.success,
        },
        metadata={"component": "analytics"},
        emit_legacy=False,
    )


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
    publish_bus_event(
        "persona_metrics.skill",
        event.as_dict(),
        priority=MessagePriority.LOW,
        correlation_id=f"{event.persona}:{event.tool}",
        tracing={
            "persona": event.persona,
            "category": event.category,
            "success": event.success,
        },
        metadata={"component": "analytics"},
        emit_legacy=False,
    )


def _build_lifecycle_event(
    *,
    entity_key: str,
    entity_id: str,
    event: str,
    persona: Optional[str],
    tenant_id: Optional[str],
    from_status: Optional[str],
    to_status: Optional[str],
    success: Optional[bool],
    latency_ms: Optional[float],
    timestamp: Optional[datetime],
    metadata: Optional[Mapping[str, Any]],
    extra: Optional[Mapping[str, Any]] = None,
) -> LifecycleEvent:
    return LifecycleEvent(
        entity_id=entity_id,
        entity_key=entity_key,
        event=event,
        persona=persona,
        tenant_id=tenant_id,
        from_status=from_status,
        to_status=to_status,
        success=success,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        timestamp=timestamp or datetime.now(timezone.utc),
        metadata=metadata,
        extra=extra,
    )


def record_task_lifecycle_event(
    *,
    task_id: Optional[str],
    event: str,
    persona: Optional[str] = None,
    tenant_id: Optional[str] = None,
    from_status: Optional[str] = None,
    to_status: Optional[str] = None,
    success: Optional[bool] = None,
    latency_ms: Optional[float] = None,
    reassignments: int = 0,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[Any] = None,
) -> None:
    """Persist task lifecycle analytics and broadcast to the message bus."""

    if not task_id:
        return

    persona_value = None
    if persona is not None:
        text = str(persona).strip()
        persona_value = text or None

    tenant_value = None
    if tenant_id is not None:
        text = str(tenant_id).strip()
        tenant_value = text or None

    event_payload = _build_lifecycle_event(
        entity_key="task_id",
        entity_id=str(task_id),
        event=event,
        persona=persona_value,
        tenant_id=tenant_value,
        from_status=from_status,
        to_status=to_status,
        success=success,
        latency_ms=latency_ms,
        timestamp=timestamp,
        metadata=metadata,
    )
    task_event = replace(
        event_payload,
        extra={**event_payload.extra, "reassignments": int(reassignments or 0)},
    )
    store = _get_store(config_manager)
    store.record_task_event(task_event)
    publish_bus_event(
        "task_metrics.lifecycle",
        task_event.as_dict(),
        priority=MessagePriority.LOW,
        correlation_id=str(task_id),
        tracing={
            "persona": task_event.persona,
            "event": task_event.event,
            "success": task_event.success,
        },
        metadata={"component": "analytics"},
        emit_legacy=False,
    )


def record_job_lifecycle_event(
    *,
    job_id: Optional[str],
    event: str,
    persona: Optional[str] = None,
    tenant_id: Optional[str] = None,
    from_status: Optional[str] = None,
    to_status: Optional[str] = None,
    success: Optional[bool] = None,
    latency_ms: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    config_manager: Optional[Any] = None,
) -> None:
    """Persist job lifecycle analytics and broadcast to the message bus."""

    if not job_id:
        return

    persona_value = None
    if persona is not None:
        text = str(persona).strip()
        persona_value = text or None

    tenant_value = None
    if tenant_id is not None:
        text = str(tenant_id).strip()
        tenant_value = text or None

    event_payload = _build_lifecycle_event(
        entity_key="job_id",
        entity_id=str(job_id),
        event=event,
        persona=persona_value,
        tenant_id=tenant_value,
        from_status=from_status,
        to_status=to_status,
        success=success,
        latency_ms=latency_ms,
        timestamp=timestamp,
        metadata=metadata,
    )
    store = _get_store(config_manager)
    store.record_job_event(event_payload)
    publish_bus_event(
        "jobs.metrics.lifecycle",
        event_payload.as_dict(),
        priority=MessagePriority.LOW,
        correlation_id=str(job_id),
        tracing={
            "persona": event_payload.persona,
            "event": event_payload.event,
            "success": event_payload.success,
        },
        metadata={"component": "analytics"},
        emit_legacy=False,
    )


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


def get_persona_comparison_summary(
    *,
    category: str = "tool",
    personas: Optional[Iterable[str]] = None,
    search: Optional[str] = None,
    limit_recent: int = 5,
    page: int = 1,
    page_size: int = 25,
    config_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a consolidated persona comparison summary from the shared store."""

    store = _get_store(config_manager)
    return store.get_persona_comparison_summary(
        category=category,
        personas=personas,
        search=search,
        limit_recent=limit_recent,
        page=page,
        page_size=page_size,
    )


def get_task_lifecycle_metrics(
    *,
    persona: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit_recent: int = 50,
    config_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return aggregated lifecycle analytics from the shared store."""

    store = _get_store(config_manager)
    return store.get_task_metrics(
        persona=persona, tenant_id=tenant_id, limit_recent=limit_recent
    )


def get_job_lifecycle_metrics(
    *,
    persona: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit_recent: int = 50,
    config_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return aggregated job lifecycle analytics from the shared store."""

    store = _get_store(config_manager)
    return store.get_job_metrics(
        persona=persona, tenant_id=tenant_id, limit_recent=limit_recent
    )


def reset_persona_metrics(*, config_manager: Optional[Any] = None) -> None:
    """Clear recorded persona metrics in the shared store."""

    store = _get_store(config_manager)
    store.reset()


def reset_task_metrics(*, config_manager: Optional[Any] = None) -> None:
    """Clear recorded task lifecycle metrics in the shared store."""

    store = _get_store(config_manager)
    store.reset_task_metrics()


def _dispatch_persona_metric_alert(alert: Mapping[str, Any]) -> None:
    if not isinstance(alert, Mapping):
        return

    persona = alert.get("persona")
    metric = alert.get("metric")
    if not persona or not metric:
        return

    payload = {
        "persona": persona,
        "metric": metric,
        "category": alert.get("category"),
        "observed": alert.get("observed"),
        "baseline": dict(alert.get("baseline") or {}),
        "timestamp": alert.get("timestamp"),
        "suggested_actions": list(alert.get("suggested_actions") or []),
    }

    publish_bus_event(
        "persona_metrics.alert",
        payload,
        priority=MessagePriority.HIGH,
        correlation_id=f"{persona}:{metric}",
        tracing={"persona": persona, "metric": metric},
        metadata={"component": "analytics"},
        emit_legacy=False,
    )
