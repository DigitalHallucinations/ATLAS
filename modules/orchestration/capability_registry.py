"""Centralized registry for tool and skill capabilities.

This module ingests shared and persona-specific manifests, normalizes the
metadata required by orchestrators, and maintains rolling health metrics that
can be queried by routers or UI layers.
"""

from __future__ import annotations

import copy
import threading
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from modules.Tools.manifest_loader import ToolManifestEntry, load_manifest_entries
from modules.Skills.manifest_loader import SkillMetadata, load_skill_metadata
from modules.Tasks.manifest_loader import TaskMetadata, load_task_metadata
from modules.Jobs.manifest_loader import JobMetadata, load_job_metadata
from modules.logging.logger import setup_logger
from modules.orchestration.utils import (
    SHARED_PERSONA_EXCLUSION_TOKENS,
    normalize_persona_identifier,
    persona_matches_filter,
)

try:  # ConfigManager may not be available in certain test scenarios
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - defensive import guard
    ConfigManager = None  # type: ignore


logger = setup_logger(__name__)


_MAX_METRIC_SAMPLES = 100


_EMPTY_METRIC_SUMMARY = MappingProxyType(
    {
        "total": 0,
        "success": 0,
        "failure": 0,
        "success_rate": 0.0,
        "average_latency_ms": None,
        "p95_latency_ms": None,
        "last_sample_at": None,
    }
)


def _coerce_string_sequence(values: Iterable[Any]) -> Tuple[str, ...]:
    tokens: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            tokens.append(text)
    return tuple(tokens)


def _dedupe_sequence(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _persona_filter_matches(persona: Optional[str], tokens: Sequence[str]) -> bool:
    return persona_matches_filter(persona, tokens)


def _job_persona_matches(personas: Sequence[str], tokens: Sequence[str]) -> bool:
    if not tokens:
        return True

    normalized_personas: set[str] = set()
    for persona in personas:
        normalized = normalize_persona_identifier(persona)
        if normalized:
            normalized_personas.add(normalized)

    positive_tokens = [
        token
        for token in tokens
        if token not in SHARED_PERSONA_EXCLUSION_TOKENS
    ]

    if not positive_tokens:
        return True

    return any(token in normalized_personas for token in positive_tokens)


def _parse_version(text: Optional[str]) -> Optional[Tuple[Any, ...]]:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped:
        return None
    components: List[Any] = []
    for token in stripped.replace("-", ".").split("."):
        if not token:
            continue
        try:
            components.append(int(token))
        except ValueError:
            components.append(token)
    return tuple(components)


def _compare_versions(lhs: Optional[str], rhs: Optional[str]) -> Optional[int]:
    left = _parse_version(lhs)
    right = _parse_version(rhs)
    if left is None or right is None:
        return None
    for a, b in zip(left, right):
        if a == b:
            continue
        if isinstance(a, int) and isinstance(b, int):
            return -1 if a < b else 1
        return -1 if str(a) < str(b) else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


def _match_version(version: Optional[str], constraint: Optional[str]) -> bool:
    if constraint is None:
        return True
    constraint_text = str(constraint).strip()
    if not constraint_text:
        return True
    if version is None:
        return False

    clauses = [clause.strip() for clause in constraint_text.split(",") if clause.strip()]
    for clause in clauses:
        if clause.startswith(">="):
            comparison = _compare_versions(version, clause[2:].strip())
            if comparison is not None and comparison < 0:
                return False
        elif clause.startswith("<="):
            comparison = _compare_versions(version, clause[2:].strip())
            if comparison is not None and comparison > 0:
                return False
        elif clause.startswith(">"):
            comparison = _compare_versions(version, clause[1:].strip())
            if comparison is not None and comparison <= 0:
                return False
        elif clause.startswith("<"):
            comparison = _compare_versions(version, clause[1:].strip())
            if comparison is not None and comparison >= 0:
                return False
        elif clause.startswith("=") or clause.startswith("=="):
            comparison = _compare_versions(version, clause.lstrip("=").strip())
            if comparison is not None and comparison != 0:
                return False
        else:
            comparison = _compare_versions(version, clause)
            if comparison is not None and comparison != 0:
                return False
    return True


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = percentile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _compatibility_flags(
    entry_persona: Optional[str],
    requested_persona: Optional[str],
    allowlist: Optional[Iterable[Any]] = None,
) -> Mapping[str, Any]:
    entry_key = normalize_persona_identifier(entry_persona)
    requested_key = normalize_persona_identifier(requested_persona)
    tokens = [
        str(token).strip()
        for token in allowlist or []
        if isinstance(token, str) and str(token).strip()
    ]
    matches = True
    if requested_key is not None:
        if entry_key is None:
            matches = True
        else:
            matches = entry_key == requested_key
    payload = {
        "persona": entry_persona,
        "persona_specific": entry_key is not None,
        "matches_request": matches,
        "allowlist": tuple(tokens),
    }
    return MappingProxyType(payload)


@dataclass
class RollingMetricWindow:
    """Rolling window aggregating success and latency metrics."""

    maxlen: int = _MAX_METRIC_SAMPLES
    samples: deque = field(default_factory=lambda: deque(maxlen=_MAX_METRIC_SAMPLES))

    def add(
        self,
        *,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        timestamp = timestamp or time.time()
        latency: Optional[float]
        if latency_ms is None:
            latency = None
        else:
            try:
                latency = float(latency_ms)
            except (TypeError, ValueError):
                latency = None
        self.samples.append((timestamp, bool(success), latency))

    def summary(self) -> Mapping[str, Any]:
        total = len(self.samples)
        successes = sum(1 for _, result, _ in self.samples if result)
        failures = total - successes
        latencies = [lat for _, _, lat in self.samples if lat is not None]
        average_latency = (sum(latencies) / len(latencies)) if latencies else None
        p95 = _percentile(latencies, 0.95) if latencies else None
        summary = {
            "total": total,
            "success": successes,
            "failure": failures,
            "success_rate": (successes / total) if total else 0.0,
            "average_latency_ms": average_latency,
            "p95_latency_ms": p95,
            "last_sample_at": self.samples[-1][0] if self.samples else None,
        }
        return MappingProxyType(summary)


@dataclass(frozen=True)
class ToolCapabilityView:
    """Projection of a tool manifest entry with normalized metadata."""

    manifest: ToolManifestEntry
    capability_tags: Tuple[str, ...]
    auth_scopes: Tuple[str, ...]
    health: Mapping[str, Any]


@dataclass(frozen=True)
class SkillCapabilityView:
    """Projection of a skill manifest entry with normalized metadata."""

    manifest: SkillMetadata
    capability_tags: Tuple[str, ...]
    required_capabilities: Tuple[str, ...]


@dataclass(frozen=True)
class TaskCapabilityView:
    """Projection of a task manifest entry with normalized metadata."""

    manifest: TaskMetadata
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    tags: Tuple[str, ...]


@dataclass(frozen=True)
class JobCapabilityView:
    """Projection of a job manifest entry with normalized metadata."""

    manifest: JobMetadata
    personas: Tuple[str, ...]
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    required_capabilities: Tuple[str, ...]
    health: Mapping[str, Any]


@dataclass
class _ToolRecord:
    manifest: ToolManifestEntry
    capability_tags: Tuple[str, ...]
    auth_scopes: Tuple[str, ...]
    raw_entry: Mapping[str, Any]


@dataclass
class _SkillRecord:
    manifest: SkillMetadata
    capability_tags: Tuple[str, ...]
    required_capabilities: Tuple[str, ...]


@dataclass
class _TaskRecord:
    manifest: TaskMetadata
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    tags: Tuple[str, ...]


@dataclass
class _JobRecord:
    manifest: JobMetadata
    personas: Tuple[str, ...]
    required_skills: Tuple[str, ...]
    required_tools: Tuple[str, ...]
    required_capabilities: Tuple[str, ...]


class CapabilityRegistry:
    """Registry providing cached manifest metadata and health statistics."""

    def __init__(self, *, config_manager: Optional[Any] = None) -> None:
        self._config_manager = config_manager
        self._lock = threading.RLock()
        self._revision = 0
        self._manifest_state: Dict[str, Optional[float]] = {}
        self._tool_records: List[_ToolRecord] = []
        self._tool_lookup: Dict[Tuple[Optional[str], str], _ToolRecord] = {}
        self._tool_payloads: Dict[Optional[str], List[Mapping[str, Any]]] = {}
        self._tool_payload_lookup: Dict[Optional[str], Dict[str, Mapping[str, Any]]] = {}
        self._persona_tool_sources: Dict[Optional[str], bool] = {}
        self._tool_metrics: Dict[Tuple[Optional[str], str], RollingMetricWindow] = {}
        self._provider_metrics: Dict[Tuple[Optional[str], str, str], RollingMetricWindow] = {}
        self._provider_snapshots: Dict[Tuple[Optional[str], str, str], Mapping[str, Any]] = {}
        self._provider_last_outcome: Dict[
            Tuple[Optional[str], str, str], Mapping[str, Any]
        ] = {}
        self._tool_last_invocation: Dict[Tuple[Optional[str], str], Mapping[str, Any]] = {}
        self._skill_records: List[_SkillRecord] = []
        self._skill_lookup: Dict[Tuple[Optional[str], str], _SkillRecord] = {}
        self._task_records: List[_TaskRecord] = []
        self._task_lookup: Dict[Tuple[Optional[str], str], _TaskRecord] = {}
        self._job_records: List[_JobRecord] = []
        self._job_lookup: Dict[Tuple[Optional[str], str], _JobRecord] = {}
        self._job_metrics: Dict[Tuple[Optional[str], str], RollingMetricWindow] = {}

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def set_config_manager(self, config_manager: Optional[Any]) -> None:
        with self._lock:
            self._config_manager = config_manager

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    # ------------------------------------------------------------------
    # Manifest ingestion
    # ------------------------------------------------------------------
    def refresh(self, *, force: bool = False) -> bool:
        with self._lock:
            app_root = self._resolve_app_root()
            if app_root is None:
                logger.warning("Capability registry unable to resolve application root")
                return False

            manifest_state = self._compute_manifest_state(app_root)
            if not force and manifest_state == self._manifest_state:
                return False

            tool_payloads, payload_lookup, persona_sources = self._load_tool_payloads(app_root)
            tool_entries = load_manifest_entries(config_manager=self._config_manager)
            tool_lookup: Dict[Tuple[Optional[str], str], _ToolRecord] = {}
            tool_records: List[_ToolRecord] = []

            for entry in tool_entries:
                persona_key = normalize_persona_identifier(entry.persona)
                name_key = entry.name
                raw_entry = payload_lookup.get(persona_key, {}).get(name_key)
                if raw_entry is None and persona_key is not None:
                    raw_entry = payload_lookup.get(None, {}).get(name_key)
                if raw_entry is None:
                    continue
                capability_tags = _coerce_string_sequence(entry.capabilities)
                auth = entry.auth if isinstance(entry.auth, Mapping) else {}
                auth_scopes = _coerce_string_sequence(auth.get("scopes", [])) if isinstance(auth, Mapping) else tuple()
                record = _ToolRecord(
                    manifest=entry,
                    capability_tags=capability_tags,
                    auth_scopes=auth_scopes,
                    raw_entry=MappingProxyType(dict(raw_entry)),
                )
                tool_records.append(record)
                tool_lookup[(persona_key, entry.name)] = record

            skill_entries = load_skill_metadata(config_manager=self._config_manager)
            skill_records: List[_SkillRecord] = []
            skill_lookup: Dict[Tuple[Optional[str], str], _SkillRecord] = {}
            for entry in skill_entries:
                persona_key = normalize_persona_identifier(entry.persona)
                record = _SkillRecord(
                    manifest=entry,
                    capability_tags=_coerce_string_sequence(entry.capability_tags),
                    required_capabilities=_coerce_string_sequence(entry.required_capabilities),
                )
                skill_records.append(record)
                skill_lookup[(persona_key, entry.name)] = record

            task_entries = load_task_metadata(config_manager=self._config_manager)
            task_records: List[_TaskRecord] = []
            task_lookup: Dict[Tuple[Optional[str], str], _TaskRecord] = {}
            for entry in task_entries:
                persona_key = normalize_persona_identifier(entry.persona)
                record = _TaskRecord(
                    manifest=entry,
                    required_skills=_coerce_string_sequence(entry.required_skills),
                    required_tools=_coerce_string_sequence(entry.required_tools),
                    tags=_coerce_string_sequence(entry.tags),
                )
                task_records.append(record)
                task_lookup[(persona_key, entry.name)] = record

            job_entries = load_job_metadata(config_manager=self._config_manager)
            job_records: List[_JobRecord] = []
            job_lookup: Dict[Tuple[Optional[str], str], _JobRecord] = {}
            for entry in job_entries:
                persona_key = normalize_persona_identifier(entry.persona)
                personas = _dedupe_sequence(entry.personas)
                required_skills = _coerce_string_sequence(entry.required_skills)
                required_tools = _coerce_string_sequence(entry.required_tools)

                capability_hints: List[str] = []
                for skill_name in required_skills:
                    skill_record = skill_lookup.get((persona_key, skill_name))
                    if skill_record is None:
                        skill_record = skill_lookup.get((None, skill_name))
                    if skill_record is None:
                        continue
                    capability_hints.extend(skill_record.required_capabilities)
                    capability_hints.extend(skill_record.capability_tags)

                for tool_name in required_tools:
                    tool_record = tool_lookup.get((persona_key, tool_name))
                    if tool_record is None:
                        tool_record = tool_lookup.get((None, tool_name))
                    if tool_record is None:
                        continue
                    capability_hints.extend(tool_record.capability_tags)
                    capabilities = getattr(tool_record.manifest, "capabilities", ())
                    if isinstance(capabilities, Iterable) and not isinstance(
                        capabilities, (str, bytes)
                    ):
                        capability_hints.extend(
                            str(token).strip()
                            for token in capabilities
                            if isinstance(token, str) and str(token).strip()
                        )

                record = _JobRecord(
                    manifest=entry,
                    personas=personas,
                    required_skills=required_skills,
                    required_tools=required_tools,
                    required_capabilities=_dedupe_sequence(capability_hints),
                )
                job_records.append(record)
                job_lookup[(persona_key, entry.name)] = record

            self._tool_records = tool_records
            self._tool_lookup = tool_lookup
            self._tool_payloads = tool_payloads
            self._tool_payload_lookup = payload_lookup
            self._persona_tool_sources = persona_sources
            self._skill_records = skill_records
            self._skill_lookup = skill_lookup
            self._task_records = task_records
            self._task_lookup = task_lookup
            self._job_records = job_records
            self._job_lookup = job_lookup
            self._manifest_state = manifest_state
            self._revision += 1
            logger.debug("Capability registry refreshed (revision=%s)", self._revision)
            return True

    def refresh_if_stale(self) -> None:
        self.refresh(force=False)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def query_tools(
        self,
        *,
        persona_filters: Optional[Sequence[str]] = None,
        capability_filters: Optional[Sequence[str]] = None,
        provider_filters: Optional[Sequence[str]] = None,
        version_constraint: Optional[str] = None,
        min_success_rate: Optional[float] = None,
    ) -> List[ToolCapabilityView]:
        with self._lock:
            self.refresh_if_stale()
            persona_tokens = [
                token.strip().lower()
                for token in persona_filters or []
                if isinstance(token, str) and token.strip()
            ]
            capability_tokens = {
                token.strip().lower()
                for token in capability_filters or []
                if isinstance(token, str) and token.strip()
            }
            provider_tokens = {
                token.strip().lower()
                for token in provider_filters or []
                if isinstance(token, str) and token.strip()
            }

            results: List[ToolCapabilityView] = []
            for record in self._tool_records:
                manifest = record.manifest
                persona_key = normalize_persona_identifier(manifest.persona)
                if persona_tokens and not _persona_filter_matches(persona_key, persona_tokens):
                    continue
                if capability_tokens:
                    record_caps = {token.lower() for token in record.capability_tags}
                    if not capability_tokens.issubset(record_caps):
                        continue
                if provider_tokens:
                    provider_names = {
                        str(provider.get("name", "")).strip().lower()
                        for provider in manifest.providers
                        if isinstance(provider, Mapping)
                    }
                    if not provider_tokens.issubset(provider_names):
                        continue
                if not _match_version(manifest.version, version_constraint):
                    continue

                tool_key = (persona_key, manifest.name)
                tool_metric = self._tool_metrics.get(tool_key)
                tool_summary = tool_metric.summary() if tool_metric else _EMPTY_METRIC_SUMMARY
                if min_success_rate is not None and tool_summary["success_rate"] < min_success_rate:
                    continue

                provider_health: Dict[str, Any] = {}
                for provider in manifest.providers:
                    if not isinstance(provider, Mapping):
                        continue
                    provider_name = str(provider.get("name", "")).strip()
                    if not provider_name:
                        continue
                    provider_key = (persona_key, manifest.name, provider_name.lower())
                    metrics = self._provider_metrics.get(provider_key)
                    snapshot = self._provider_snapshots.get(provider_key, MappingProxyType({}))
                    last_outcome = self._provider_last_outcome.get(
                        provider_key, MappingProxyType({})
                    )
                    provider_health[provider_name] = MappingProxyType(
                        {
                            "metrics": metrics.summary() if metrics else _EMPTY_METRIC_SUMMARY,
                            "router": snapshot,
                            "last_call": last_outcome,
                        }
                    )

                last_invocation = self._tool_last_invocation.get(
                    (persona_key, manifest.name), MappingProxyType({})
                )
                results.append(
                    ToolCapabilityView(
                        manifest=manifest,
                        capability_tags=record.capability_tags,
                        auth_scopes=record.auth_scopes,
                        health=MappingProxyType({
                            "tool": tool_summary,
                            "providers": MappingProxyType(provider_health),
                            "last_invocation": last_invocation,
                        }),
                    )
                )

            return results

    def query_skills(
        self,
        *,
        persona_filters: Optional[Sequence[str]] = None,
        capability_filters: Optional[Sequence[str]] = None,
        version_constraint: Optional[str] = None,
    ) -> List[SkillCapabilityView]:
        with self._lock:
            self.refresh_if_stale()
            persona_tokens = [
                token.strip().lower()
                for token in persona_filters or []
                if isinstance(token, str) and token.strip()
            ]
            capability_tokens = {
                token.strip().lower()
                for token in capability_filters or []
                if isinstance(token, str) and token.strip()
            }

            results: List[SkillCapabilityView] = []
            for record in self._skill_records:
                manifest = record.manifest
                persona_key = normalize_persona_identifier(manifest.persona)
                if persona_tokens and not _persona_filter_matches(persona_key, persona_tokens):
                    continue
                if capability_tokens:
                    record_caps = {token.lower() for token in record.capability_tags}
                    if not capability_tokens.issubset(record_caps):
                        continue
                if not _match_version(manifest.version, version_constraint):
                    continue
                results.append(
                    SkillCapabilityView(
                        manifest=manifest,
                        capability_tags=record.capability_tags,
                        required_capabilities=record.required_capabilities,
                    )
                )
            return results

    def query_tasks(
        self,
        *,
        persona_filters: Optional[Sequence[str]] = None,
        required_skill_filters: Optional[Sequence[str]] = None,
        required_tool_filters: Optional[Sequence[str]] = None,
        tag_filters: Optional[Sequence[str]] = None,
    ) -> List[TaskCapabilityView]:
        with self._lock:
            self.refresh_if_stale()
            persona_tokens = [
                token.strip().lower()
                for token in persona_filters or []
                if isinstance(token, str) and token.strip()
            ]
            skill_tokens = {
                token.strip().lower()
                for token in required_skill_filters or []
                if isinstance(token, str) and token.strip()
            }
            tool_tokens = {
                token.strip().lower()
                for token in required_tool_filters or []
                if isinstance(token, str) and token.strip()
            }
            tag_tokens = {
                token.strip().lower()
                for token in tag_filters or []
                if isinstance(token, str) and token.strip()
            }

            results: List[TaskCapabilityView] = []
            for record in self._task_records:
                manifest = record.manifest
                persona_key = normalize_persona_identifier(manifest.persona)
                if persona_tokens and not _persona_filter_matches(persona_key, persona_tokens):
                    continue
                if skill_tokens:
                    record_skills = {token.lower() for token in record.required_skills}
                    if not skill_tokens.issubset(record_skills):
                        continue
                if tool_tokens:
                    record_tools = {token.lower() for token in record.required_tools}
                    if not tool_tokens.issubset(record_tools):
                        continue
                if tag_tokens:
                    record_tags = {token.lower() for token in record.tags}
                    if not tag_tokens.issubset(record_tags):
                        continue

                results.append(
                    TaskCapabilityView(
                        manifest=manifest,
                        required_skills=record.required_skills,
                        required_tools=record.required_tools,
                        tags=record.tags,
                    )
                )
            return results

    def query_jobs(
        self,
        *,
        persona_filters: Optional[Sequence[str]] = None,
        required_capability_filters: Optional[Sequence[str]] = None,
        required_skill_filters: Optional[Sequence[str]] = None,
    ) -> List[JobCapabilityView]:
        with self._lock:
            self.refresh_if_stale()
            persona_tokens = [
                token.strip().lower()
                for token in persona_filters or []
                if isinstance(token, str) and token.strip()
            ]
            capability_tokens = {
                token.strip().lower()
                for token in required_capability_filters or []
                if isinstance(token, str) and token.strip()
            }
            skill_tokens = {
                token.strip().lower()
                for token in required_skill_filters or []
                if isinstance(token, str) and token.strip()
            }

            results: List[JobCapabilityView] = []
            for record in self._job_records:
                manifest = record.manifest
                persona_key = normalize_persona_identifier(manifest.persona)
                if persona_tokens and not _persona_filter_matches(persona_key, persona_tokens):
                    continue
                if persona_tokens and not _job_persona_matches(record.personas, persona_tokens):
                    continue
                if capability_tokens:
                    record_caps = {token.lower() for token in record.required_capabilities}
                    if not capability_tokens.issubset(record_caps):
                        continue
                if skill_tokens:
                    record_skills = {token.lower() for token in record.required_skills}
                    if not skill_tokens.issubset(record_skills):
                        continue

                job_key = (persona_key, manifest.name)
                metrics = self._job_metrics.get(job_key)
                job_health = metrics.summary() if metrics else _EMPTY_METRIC_SUMMARY

                results.append(
                    JobCapabilityView(
                        manifest=manifest,
                        personas=record.personas,
                        required_skills=record.required_skills,
                        required_tools=record.required_tools,
                        required_capabilities=record.required_capabilities,
                        health=MappingProxyType({"job": job_health}),
                    )
                )
            return results

    def get_task_catalog(self, persona: Optional[str]) -> List[TaskCapabilityView]:
        with self._lock:
            self.refresh_if_stale()
            persona_key = normalize_persona_identifier(persona)

            catalog: Dict[str, _TaskRecord] = {}
            for record in self._task_records:
                record_persona = normalize_persona_identifier(record.manifest.persona)
                if record_persona is None:
                    catalog[record.manifest.name] = record

            if persona_key is not None:
                for record in self._task_records:
                    if normalize_persona_identifier(record.manifest.persona) == persona_key:
                        catalog[record.manifest.name] = record

            ordered = sorted(catalog.values(), key=lambda rec: rec.manifest.name.lower())
            return [
                TaskCapabilityView(
                    manifest=record.manifest,
                    required_skills=record.required_skills,
                    required_tools=record.required_tools,
                    tags=record.tags,
                )
                for record in ordered
            ]

    def summary(self, *, persona: Optional[str] = None) -> Mapping[str, Any]:
        """Return an overview of tools, skills, tasks, and jobs for dashboards."""

        persona_filters: Optional[List[str]]
        if persona is None:
            persona_filters = None
        else:
            persona_filters = [persona]

        tool_views = self.query_tools(persona_filters=persona_filters)
        skill_views = self.query_skills(persona_filters=persona_filters)
        task_views = self.get_task_catalog(persona)
        job_views = self.query_jobs(persona_filters=persona_filters)

        with self._lock:
            revision = self._revision

        tool_entries = []
        for view in tool_views:
            manifest = view.manifest
            compatibility = _compatibility_flags(
                manifest.persona, persona, getattr(manifest, "persona_allowlist", None)
            )
            tool_entries.append(
                MappingProxyType(
                    {
                        "name": manifest.name,
                        "persona": manifest.persona,
                        "version": manifest.version,
                        "capabilities": list(manifest.capabilities),
                        "capability_tags": list(view.capability_tags),
                        "compatibility": compatibility,
                        "health": view.health,
                        "providers": manifest.providers,
                    }
                )
            )

        skill_entries = []
        for view in skill_views:
            manifest = view.manifest
            compatibility = _compatibility_flags(manifest.persona, persona)
            skill_entries.append(
                MappingProxyType(
                    {
                        "name": manifest.name,
                        "persona": manifest.persona,
                        "version": manifest.version,
                        "capability_tags": list(view.capability_tags),
                        "required_capabilities": list(view.required_capabilities),
                        "required_tools": list(manifest.required_tools),
                        "compatibility": compatibility,
                    }
                )
            )

        task_entries = []
        for view in task_views:
            manifest = view.manifest
            compatibility = _compatibility_flags(manifest.persona, persona)
            task_entries.append(
                MappingProxyType(
                    {
                        "name": manifest.name,
                        "persona": manifest.persona,
                        "summary": manifest.summary,
                        "required_skills": list(view.required_skills),
                        "required_tools": list(view.required_tools),
                        "tags": list(view.tags),
                        "compatibility": compatibility,
                    }
                )
            )

        job_entries = []
        for view in job_views:
            manifest = view.manifest
            compatibility = _compatibility_flags(manifest.persona, persona, view.personas)
            job_entries.append(
                MappingProxyType(
                    {
                        "name": manifest.name,
                        "persona": manifest.persona,
                        "summary": manifest.summary,
                        "description": manifest.description,
                        "personas": list(view.personas),
                        "required_skills": list(view.required_skills),
                        "required_tools": list(view.required_tools),
                        "required_capabilities": list(view.required_capabilities),
                        "health": view.health,
                        "compatibility": compatibility,
                    }
                )
            )

        analytics: Mapping[str, Any] = MappingProxyType({})
        try:
            from modules.analytics.persona_metrics import (
                get_job_lifecycle_metrics,
                get_task_lifecycle_metrics,
            )

            analytics_payload = {
                "tasks": get_task_lifecycle_metrics(persona=persona),
                "jobs": get_job_lifecycle_metrics(persona=persona),
            }
            analytics = MappingProxyType(analytics_payload)
        except Exception as exc:  # pragma: no cover - analytics optional
            logger.debug("Capability summary analytics unavailable: %s", exc)

        payload = {
            "revision": revision,
            "persona": persona,
            "tools": tuple(tool_entries),
            "skills": tuple(skill_entries),
            "tasks": tuple(task_entries),
            "jobs": tuple(job_entries),
            "analytics": analytics,
        }
        return MappingProxyType(payload)

    def get_tool_metadata_lookup(
        self,
        *,
        persona: Optional[str],
        names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Mapping[str, Any]]:
        with self._lock:
            self.refresh_if_stale()
            persona_key = normalize_persona_identifier(persona)
            requested = {name for name in names or [] if isinstance(name, str)}
            if not requested:
                requested = {
                    record.manifest.name
                    for record in self._tool_records
                    if normalize_persona_identifier(record.manifest.persona) == persona_key
                    or persona_key is None
                }

            lookup: Dict[str, Mapping[str, Any]] = {}
            for name in requested:
                record = self._resolve_tool_record(name, persona_key)
                if record is None:
                    continue
                manifest = record.manifest
                metadata: Dict[str, Any] = {
                    "persona": manifest.persona,
                    "source": manifest.source,
                    "providers": manifest.providers,
                    "capabilities": list(record.capability_tags),
                    "side_effects": manifest.side_effects,
                    "default_timeout": manifest.default_timeout,
                    "auth": manifest.auth,
                    "allow_parallel": manifest.allow_parallel,
                    "idempotency_key": manifest.idempotency_key,
                    "safety_level": manifest.safety_level,
                    "requires_consent": manifest.requires_consent,
                    "persona_allowlist": manifest.persona_allowlist,
                    "cost_per_call": manifest.cost_per_call,
                    "cost_unit": manifest.cost_unit,
                    "version": manifest.version,
                    "requires_flags": getattr(manifest, "requires_flags", None),
                    "auth_scopes": list(record.auth_scopes),
                }
                lookup[name] = MappingProxyType(metadata)
            return lookup

    def get_tool_manifest_payload(
        self,
        *,
        persona: Optional[str],
        allowed_names: Optional[Sequence[str]] = None,
    ) -> Optional[List[Mapping[str, Any]]]:
        with self._lock:
            self.refresh_if_stale()
            persona_key = normalize_persona_identifier(persona)
            allowed: Optional[List[str]]
            if allowed_names is None:
                allowed = None
            else:
                allowed = [name for name in allowed_names if isinstance(name, str) and name]

            payloads = self._tool_payloads.get(persona_key)
            has_persona_manifest = self._persona_tool_sources.get(persona_key, False)

            if allowed is None:
                if payloads is not None:
                    return [copy.deepcopy(entry) for entry in payloads]
                shared_payload = self._tool_payloads.get(None)
                if shared_payload is not None:
                    return [copy.deepcopy(entry) for entry in shared_payload]
                return None

            results: List[Mapping[str, Any]] = []
            lookup = self._tool_payload_lookup.get(persona_key, {})
            shared_lookup = self._tool_payload_lookup.get(None, {})
            for name in allowed:
                entry = lookup.get(name)
                if entry is None:
                    entry = shared_lookup.get(name)
                if entry is not None:
                    results.append(copy.deepcopy(entry))
            if not results and not has_persona_manifest:
                shared_payload = self._tool_payloads.get(None)
                if shared_payload is not None:
                    return [copy.deepcopy(entry) for entry in shared_payload]
            return results

    def persona_has_tool_manifest(self, persona: Optional[str]) -> bool:
        with self._lock:
            self.refresh_if_stale()
            persona_key = normalize_persona_identifier(persona)
            return bool(self._persona_tool_sources.get(persona_key))

    # ------------------------------------------------------------------
    # Metrics ingestion
    # ------------------------------------------------------------------
    def record_tool_execution(
        self,
        *,
        persona: Optional[str],
        tool_name: str,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        persona_key = normalize_persona_identifier(persona)
        tool_key = (persona_key, tool_name)
        with self._lock:
            window = self._tool_metrics.get(tool_key)
            if window is None:
                window = RollingMetricWindow()
                self._tool_metrics[tool_key] = window
            window.add(success=success, latency_ms=latency_ms, timestamp=timestamp)

    def record_provider_metrics(
        self,
        *,
        persona: Optional[str],
        tool_name: str,
        summary: Mapping[str, Any],
    ) -> None:
        persona_key = normalize_persona_identifier(persona)
        selected = str(summary.get("selected") or "").strip()
        success = bool(summary.get("success"))
        providers = summary.get("providers")
        timestamp_value = summary.get("timestamp")
        try:
            timestamp = float(timestamp_value) if timestamp_value is not None else time.time()
        except (TypeError, ValueError):
            timestamp = time.time()
        latency_value = summary.get("latency_ms")
        try:
            latency_ms = float(latency_value) if latency_value is not None else None
        except (TypeError, ValueError):
            latency_ms = None

        if not isinstance(providers, Mapping):
            providers = {}

        with self._lock:
            tool_key = (persona_key, tool_name)
            self._tool_last_invocation[tool_key] = MappingProxyType(
                {
                    "provider": selected or None,
                    "success": success,
                    "latency_ms": latency_ms,
                    "timestamp": timestamp,
                }
            )
            for provider_name, snapshot in providers.items():
                provider_key = (persona_key, tool_name, str(provider_name).strip().lower())
                if provider_name == selected and selected:
                    window = self._provider_metrics.get(provider_key)
                    if window is None:
                        window = RollingMetricWindow()
                        self._provider_metrics[provider_key] = window
                    window.add(success=success, latency_ms=latency_ms, timestamp=timestamp)
                    self._provider_last_outcome[provider_key] = MappingProxyType(
                        {
                            "success": success,
                            "latency_ms": latency_ms,
                            "timestamp": timestamp,
                        }
                    )
                self._provider_snapshots[provider_key] = MappingProxyType(dict(snapshot))

    def record_job_execution(
        self,
        *,
        persona: Optional[str],
        job_name: str,
        success: bool,
        latency_ms: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        persona_key = normalize_persona_identifier(persona)
        job_key = (persona_key, job_name)
        with self._lock:
            window = self._job_metrics.get(job_key)
            if window is None:
                window = RollingMetricWindow()
                self._job_metrics[job_key] = window
            window.add(success=success, latency_ms=latency_ms, timestamp=timestamp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_tool_record(
        self,
        name: str,
        persona_key: Optional[str],
    ) -> Optional[_ToolRecord]:
        lookup_key = (persona_key, name)
        record = self._tool_lookup.get(lookup_key)
        if record is not None:
            return record
        if persona_key is not None:
            return self._tool_lookup.get((None, name))
        return None

    def _resolve_app_root(self) -> Optional[Path]:
        if self._config_manager is not None:
            getter = getattr(self._config_manager, "get_app_root", None)
            if callable(getter):
                try:
                    root = getter()
                    if root:
                        return Path(root).expanduser().resolve()
                except Exception:  # pragma: no cover - defensive guard
                    logger.warning("Capability registry failed to resolve app root from config manager", exc_info=True)
        if ConfigManager is not None:
            try:
                manager = self._config_manager or ConfigManager()
                root = getattr(manager, "get_app_root", lambda: None)()
                if root:
                    return Path(root).expanduser().resolve()
            except Exception:  # pragma: no cover - defensive guard
                logger.warning("Capability registry unable to resolve app root via ConfigManager", exc_info=True)
        try:
            return Path(__file__).resolve().parents[2]
        except Exception:  # pragma: no cover - defensive guard
            return None

    def _compute_manifest_state(self, app_root: Path) -> Dict[str, Optional[float]]:
        paths = list(self._iter_manifest_paths(app_root))
        state: Dict[str, Optional[float]] = {}
        for path in paths:
            try:
                state[str(path)] = path.stat().st_mtime
            except FileNotFoundError:
                state[str(path)] = None
        return state

    def _iter_manifest_paths(self, app_root: Path) -> Iterator[Path]:
        yield app_root / "modules" / "Tools" / "tool_maps" / "functions.json"
        personas_root = app_root / "modules" / "Personas"
        if personas_root.is_dir():
            for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
                yield persona_dir / "Toolbox" / "functions.json"
                yield persona_dir / "Skills" / "skills.json"
                yield persona_dir / "Tasks" / "tasks.json"
                yield persona_dir / "Jobs" / "jobs.json"
        yield app_root / "modules" / "Skills" / "skills.json"
        yield app_root / "modules" / "Tasks" / "tasks.json"
        yield app_root / "modules" / "Jobs" / "jobs.json"

    def _load_tool_payloads(
        self, app_root: Path
    ) -> Tuple[
        Dict[Optional[str], List[Mapping[str, Any]]],
        Dict[Optional[str], Dict[str, Mapping[str, Any]]],
        Dict[Optional[str], bool],
    ]:
        payloads: Dict[Optional[str], List[Mapping[str, Any]]] = {}
        lookup: Dict[Optional[str], Dict[str, Mapping[str, Any]]] = {}
        persona_sources: Dict[Optional[str], bool] = {}

        shared_manifest = app_root / "modules" / "Tools" / "tool_maps" / "functions.json"
        shared_entries, shared_exists = self._read_tool_manifest(shared_manifest)
        if shared_entries:
            payloads[None] = shared_entries
            lookup[None] = {entry.get("name"): entry for entry in shared_entries if isinstance(entry, Mapping) and entry.get("name")}
        persona_sources[None] = shared_exists

        personas_root = app_root / "modules" / "Personas"
        if personas_root.is_dir():
            for persona_dir in sorted(p for p in personas_root.iterdir() if p.is_dir()):
                persona_name = persona_dir.name
                manifest_path = persona_dir / "Toolbox" / "functions.json"
                entries, exists = self._read_tool_manifest(manifest_path)
                persona_key = normalize_persona_identifier(persona_name)
                if entries:
                    payloads[persona_key] = entries
                    lookup[persona_key] = {
                        entry.get("name"): entry
                        for entry in entries
                        if isinstance(entry, Mapping) and entry.get("name")
                    }
                persona_sources[persona_key] = exists
        return payloads, lookup, persona_sources

    def _read_tool_manifest(self, path: Path) -> Tuple[List[Mapping[str, Any]], bool]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ([], False)
        except OSError as exc:  # pragma: no cover - defensive guard
            logger.warning("Capability registry failed to read manifest %s: %s", path, exc)
            return ([], False)

        try:
            payload = json.loads(raw) if raw.strip() else []
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Capability registry encountered invalid JSON in %s: %s", path, exc)
            return ([], True)

        entries: List[Mapping[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    entries.append(copy.deepcopy(item))
        elif isinstance(payload, Mapping):
            for item in payload.values():
                if isinstance(item, Mapping):
                    entries.append(copy.deepcopy(item))
        else:
            logger.warning("Capability registry expected manifest %s to be list or object", path)
        return entries, True


_registry_lock = threading.Lock()
_registry_singleton: Optional[CapabilityRegistry] = None


def get_capability_registry(*, config_manager: Optional[Any] = None) -> CapabilityRegistry:
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None:
            _registry_singleton = CapabilityRegistry(config_manager=config_manager)
        elif config_manager is not None:
            _registry_singleton.set_config_manager(config_manager)
        return _registry_singleton


def reset_capability_registry() -> None:
    global _registry_singleton
    with _registry_lock:
        _registry_singleton = None


__all__ = [
    "CapabilityRegistry",
    "ToolCapabilityView",
    "SkillCapabilityView",
    "TaskCapabilityView",
    "JobCapabilityView",
    "get_capability_registry",
    "reset_capability_registry",
]
