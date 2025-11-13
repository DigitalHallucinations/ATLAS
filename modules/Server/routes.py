"""Lightweight server routing helpers for tool metadata and analytics."""

from __future__ import annotations

import asyncio
import base64
import binascii
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from modules.Personas import (
    PersonaBundleError,
    PersonaValidationError,
    load_persona_definition,
    load_tool_metadata,
    normalize_allowed_tools,
    load_skill_catalog,
    normalize_allowed_skills,
    persist_persona_definition,
    export_persona_bundle_bytes,
    import_persona_bundle_bytes,
    _validate_persona_payload,
)
from modules.analytics.persona_metrics import (
    get_job_lifecycle_metrics,
    get_persona_comparison_summary,
    get_persona_metrics,
)
from modules.conversation_store import ConversationStoreRepository
from modules.logging.audit import (
    PersonaAuditEntry,
    SkillAuditEntry,
    get_persona_audit_logger,
    get_persona_review_logger,
    get_persona_review_queue,
    get_skill_audit_logger,
    parse_persona_timestamp,
)
from modules.logging.logger import setup_logger
from modules.orchestration.blackboard import get_blackboard, stream_blackboard
from modules.orchestration.capability_registry import (
    ToolCapabilityView,
    SkillCapabilityView,
    get_capability_registry,
)
from modules.orchestration.utils import persona_matches_filter
from modules.orchestration.job_manager import JobManager
from modules.orchestration.job_scheduler import JobScheduler
from modules.orchestration.message_bus import MessageBus

if TYPE_CHECKING:
    from modules.job_store.service import JobService
    from modules.task_store.service import TaskService
from modules.Personas.persona_review import REVIEW_INTERVAL_DAYS, compute_review_status
from .conversation_routes import (
    ConversationAuthorizationError,
    ConversationRoutes,
    RequestContext,
)
from .job_routes import JobRoutes
from .task_routes import TaskRoutes

logger = setup_logger(__name__)


def _parse_query_timestamp(raw_value: Optional[Any]) -> Optional[datetime]:
    """Return a normalized UTC timestamp parsed from query parameters."""

    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_bool(raw_value: Optional[Any]) -> bool:
    """Return ``True`` when *raw_value* is truthy according to query semantics."""

    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    text = str(raw_value).strip().lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "on"}


class AtlasServer:
    """Expose read-only endpoints for tool metadata."""

    def __init__(
        self,
        *,
        config_manager: Optional[object] = None,
        conversation_repository: ConversationStoreRepository | None = None,
        message_bus: Optional[MessageBus] = None,
        task_service: "TaskService" | None = None,
        job_service: "JobService" | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        self._config_manager = config_manager
        self._conversation_repository = conversation_repository
        self._message_bus = message_bus
        self._task_service: "TaskService" | None = task_service
        self._job_service: "JobService" | None = job_service
        self._job_manager: JobManager | None = job_manager
        self._job_scheduler: JobScheduler | None = None
        self._conversation_routes: ConversationRoutes | None = None
        self._task_routes: TaskRoutes | None = None
        self._job_routes: JobRoutes | None = None

    def _get_conversation_routes(self) -> ConversationRoutes:
        if self._conversation_routes is None:
            repository = self._conversation_repository or self._build_conversation_repository()
            if repository is None:
                raise RuntimeError("Conversation store repository is not configured")
            self._conversation_repository = repository

            bus = self._message_bus
            if bus is None and hasattr(self._config_manager, "configure_message_bus"):
                try:
                    bus = self._config_manager.configure_message_bus()
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logger.warning("Failed to configure message bus: %s", exc)
                    bus = None
            self._message_bus = bus

            task_service = self._task_service or self._build_task_service()
            self._task_service = task_service
            self._conversation_routes = ConversationRoutes(
                repository,
                message_bus=bus,
                task_service=task_service,
            )
        return self._conversation_routes

    def _build_conversation_repository(self) -> ConversationStoreRepository | None:
        if self._config_manager is None:
            return None
        conversation_section = getattr(getattr(self._config_manager, "persistence", None), "conversation", None)
        session_factory = None
        retention: Mapping[str, Any] = {}
        if conversation_section is not None:
            try:
                session_factory = conversation_section.get_session_factory()
            except Exception:  # pragma: no cover - defensive path
                session_factory = None
            try:
                retention = conversation_section.get_retention_policies()
            except Exception:
                retention = {}

        if session_factory is None:
            getter = getattr(self._config_manager, "get_conversation_store_session_factory", None)
            if not callable(getter):
                return None
            session_factory = getter()
        if session_factory is None:
            return None
        if not retention:
            retention_getter = getattr(self._config_manager, "get_conversation_retention_policies", None)
            retention = retention_getter() if callable(retention_getter) else {}
        repository = ConversationStoreRepository(session_factory, retention=retention)
        try:
            repository.create_schema()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to initialize conversation store schema: %s", exc)
        return repository

    def _build_task_service(self) -> "TaskService" | None:
        if self._config_manager is None:
            return None
        getter = getattr(self._config_manager, "get_task_service", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to initialize task service: %s", exc)
            return None

    def _ensure_task_routes(self) -> TaskRoutes | None:
        if self._task_routes is not None:
            return self._task_routes

        task_service = self._task_service or self._build_task_service()
        if task_service is None:
            logger.warning(
                "Task service is not configured; task endpoints will be disabled"
            )
            return None

        self._task_service = task_service
        self._task_routes = TaskRoutes(
            task_service,
            message_bus=self._message_bus,
        )
        return self._task_routes

    def _require_task_routes(self) -> TaskRoutes:
        routes = self._ensure_task_routes()
        if routes is None:
            raise RuntimeError("Task service is not configured")
        return routes

    def _build_job_service(self) -> "JobService" | None:
        if self._config_manager is None:
            return None
        getter = getattr(self._config_manager, "get_job_service", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to initialize job service: %s", exc)
            return None

    def _build_job_manager(self) -> JobManager | None:
        if self._config_manager is None:
            return None
        getter = getattr(self._config_manager, "get_job_manager", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to initialize job manager: %s", exc)
            return None

    def _build_job_scheduler(self) -> JobScheduler | None:
        if self._config_manager is None:
            return None
        getter = getattr(self._config_manager, "get_job_scheduler", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to initialize job scheduler: %s", exc)
            return None

    def _ensure_job_routes(self) -> JobRoutes | None:
        if self._job_routes is not None:
            return self._job_routes

        job_service = self._job_service or self._build_job_service()
        if job_service is None:
            logger.warning(
                "Job service is not configured; job endpoints will be disabled"
            )
            return None

        self._job_service = job_service
        if self._job_manager is None:
            self._job_manager = self._build_job_manager()
        if self._job_scheduler is None:
            self._job_scheduler = self._build_job_scheduler()

        self._job_routes = JobRoutes(
            job_service,
            manager=self._job_manager,
            scheduler=self._job_scheduler,
            message_bus=self._message_bus,
        )
        return self._job_routes

    def _require_job_routes(self) -> JobRoutes:
        routes = self._ensure_job_routes()
        if routes is None:
            raise RuntimeError("Job service is not configured")
        return routes

    @staticmethod
    def _coerce_context(context: Any) -> RequestContext:
        if isinstance(context, RequestContext):
            return context
        if isinstance(context, Mapping):
            tenant_id = context.get("tenant_id")
            if tenant_id is None:
                raise ValueError("Context mapping must include 'tenant_id'")
            roles = context.get("roles") or ()
            metadata = context.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                metadata = None
            if isinstance(roles, str):
                roles_tuple = (roles,)
            elif isinstance(roles, Iterable):
                roles_tuple = tuple(roles)
            else:
                roles_tuple = ()
            return RequestContext(
                tenant_id=str(tenant_id),
                user_id=context.get("user_id"),
                session_id=context.get("session_id"),
                roles=roles_tuple,
                metadata=metadata,
            )
        raise TypeError("Unsupported request context type")

    def _resolve_request_context(self, context: Any = None) -> RequestContext:
        if context is not None:
            return self._coerce_context(context)

        tenant_id = "default"
        manager = self._config_manager
        if manager is not None:
            for source_name in ("config", "yaml_config"):
                source = getattr(manager, source_name, None)
                if not isinstance(source, Mapping):
                    continue
                candidate = source.get("tenant_id")
                if isinstance(candidate, str):
                    token = candidate.strip()
                    if token:
                        tenant_id = token
                        break

        return RequestContext(tenant_id=tenant_id)

    @staticmethod
    def _resolve_skill_identifier(*candidates: Any) -> str:
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, str):
                token = candidate.strip()
            else:
                token = str(candidate).strip()
            if token:
                return token
        raise ValueError("Skill name is required")

    def _find_skill_view(
        self,
        skill_name: str,
        persona: Optional[str] = None,
    ) -> SkillCapabilityView:
        registry = get_capability_registry(config_manager=self._config_manager)
        views = registry.query_skills()

        target_name = skill_name.strip().lower()
        persona_token = persona.strip().lower() if isinstance(persona, str) else None

        shared_candidate: Optional[SkillCapabilityView] = None
        fallback_candidate: Optional[SkillCapabilityView] = None

        for view in views:
            manifest = view.manifest
            manifest_name = getattr(manifest, "name", "")
            name_token = str(manifest_name).strip().lower()
            if name_token != target_name:
                continue

            manifest_persona = getattr(manifest, "persona", None)
            manifest_persona_token = (
                str(manifest_persona).strip().lower() if manifest_persona else None
            )

            if persona_token is None:
                if manifest_persona_token is None and shared_candidate is None:
                    shared_candidate = view
                elif fallback_candidate is None:
                    fallback_candidate = view
                continue

            if manifest_persona_token == persona_token:
                return view

            if manifest_persona_token is None and shared_candidate is None:
                shared_candidate = view

        if persona_token is None:
            if shared_candidate is not None:
                return shared_candidate
            if fallback_candidate is not None:
                return fallback_candidate

        if shared_candidate is not None:
            return shared_candidate

        raise KeyError(f"Skill '{skill_name}' could not be located")

    def _load_persisted_skill_metadata(
        self, skill_name: str
    ) -> Dict[str, Optional[str]]:
        metadata_getter = getattr(self._config_manager, "get_skill_metadata", None)
        default_payload = {"review_status": None, "tester_notes": None}

        if not callable(metadata_getter):
            return dict(default_payload)

        try:
            metadata = metadata_getter(skill_name)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning("Failed to load metadata for skill '%s': %s", skill_name, exc)
            return dict(default_payload)

        if not isinstance(metadata, Mapping):
            return dict(default_payload)

        review_status = metadata.get("review_status")
        if isinstance(review_status, str):
            review_status = review_status.strip() or None
        else:
            review_status = None

        tester_notes = metadata.get("tester_notes")
        if isinstance(tester_notes, str):
            tester_notes = tester_notes.strip() or None
        else:
            tester_notes = None

        return {
            "review_status": review_status,
            "tester_notes": tester_notes,
        }

    @staticmethod
    def _execute_skill_preview(
        skill: Any,
        *,
        context: Any,
        tool_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        timeout_seconds: Optional[float] = None,
        budget_ms: Optional[float] = None,
    ) -> Any:
        from ATLAS.SkillManager import use_skill

        async def _runner() -> Any:
            return await use_skill(
                skill,
                context=context,
                tool_inputs=tool_inputs,
                timeout_seconds=timeout_seconds,
                budget_ms=budget_ms,
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_runner())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # pragma: no cover - shutdown best effort
                pass
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    @staticmethod
    def _coerce_pagination_limit(
        value: Optional[Any],
        *,
        default: int = 20,
        maximum: int = 200,
    ) -> int:
        if value is None:
            return default
        try:
            limit_value = int(value)
        except (TypeError, ValueError):
            return default
        if limit_value <= 0:
            return 0
        return min(limit_value, maximum)

    @staticmethod
    def _coerce_pagination_offset(value: Optional[Any]) -> int:
        if value is None:
            return 0
        try:
            offset_value = int(value)
        except (TypeError, ValueError):
            return 0
        if offset_value < 0:
            return 0
        return offset_value

    @staticmethod
    def _format_audit_timestamp(raw: str) -> str:
        parsed = parse_persona_timestamp(str(raw or ""))
        if parsed is None:
            return str(raw or "")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _serialize_tool_change_entry(entry: PersonaAuditEntry) -> Dict[str, Any]:
        timestamp = AtlasServer._format_audit_timestamp(entry.timestamp)
        persona_name = entry.persona_name.strip() if entry.persona_name else ""
        author = entry.username.strip() if entry.username else ""

        old_tools = [tool for tool in entry.old_tools]
        new_tools = [tool for tool in entry.new_tools]

        old_lookup = set(old_tools)
        added = [tool for tool in new_tools if tool not in old_lookup]
        new_lookup = set(new_tools)
        removed = [tool for tool in old_tools if tool not in new_lookup]

        summary = "Tool configuration updated."
        if entry.rationale:
            rationale_text = entry.rationale.strip()
        else:
            rationale_text = ""

        return {
            "persona": persona_name or None,
            "author": author or "unknown",
            "timestamp": timestamp,
            "summary": summary,
            "action": "updated",
            "added": added,
            "removed": removed,
            "previous_tools": old_tools,
            "current_tools": new_tools,
            "rationale": rationale_text,
        }

    @staticmethod
    def _serialize_skill_change_entry(entry: SkillAuditEntry) -> Dict[str, Any]:
        timestamp = AtlasServer._format_audit_timestamp(entry.timestamp)
        persona_name = entry.persona_name.strip() if entry.persona_name else ""
        author = entry.username.strip() if entry.username else ""
        skill_name = entry.skill_name.strip() if entry.skill_name else ""
        summary = entry.summary.strip() if entry.summary else "Skill metadata updated."

        status_before = entry.review_status_before.strip()
        status_after = entry.review_status_after.strip()
        notes_before = entry.tester_notes_before
        notes_after = entry.tester_notes_after

        status_changed = (
            status_before.casefold() != status_after.casefold()
            if status_before and status_after
            else status_before != status_after
        )
        notes_changed = notes_before != notes_after

        details = {
            "review_status": {
                "before": status_before,
                "after": status_after,
                "changed": status_changed,
            },
            "tester_notes": {
                "before": notes_before,
                "after": notes_after,
                "changed": notes_changed,
            },
        }

        return {
            "persona": persona_name or None,
            "skill": skill_name or None,
            "author": author or "unknown",
            "timestamp": timestamp,
            "summary": summary,
            "action": "updated",
            "details": details,
        }

    def get_tools(
        self,
        *,
        capability: Optional[Any] = None,
        safety_level: Optional[Any] = None,
        persona: Optional[Any] = None,
        provider: Optional[Any] = None,
        version: Optional[Any] = None,
        min_success_rate: Optional[Any] = None,
        include_provider_health: bool = True,
    ) -> Dict[str, Any]:
        """Return merged tool metadata, optionally filtered."""

        registry = get_capability_registry(config_manager=self._config_manager)
        capability_tokens = _normalize_filters(capability)
        persona_tokens = _normalize_filters(persona)
        provider_tokens = _normalize_filters(provider)
        safety_tokens = _normalize_filters(safety_level)
        version_constraint = None
        if version is not None:
            version_text = str(version).strip()
            version_constraint = version_text or None
        success_threshold = _parse_success_rate(min_success_rate)

        views = registry.query_tools(
            persona_filters=persona_tokens,
            capability_filters=capability_tokens,
            provider_filters=provider_tokens,
            version_constraint=version_constraint,
            min_success_rate=success_threshold,
        )
        filtered = _filter_entries(
            views,
            capability_tokens=capability_tokens,
            safety_tokens=safety_tokens,
            persona_tokens=persona_tokens,
        )

        serialized: List[Dict[str, Any]] = []
        manifest_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in filtered:
            payload = _serialize_entry(
                entry, include_provider_health=include_provider_health
            )
            serialized.append(payload)

            name = payload.get("name") if isinstance(payload, Mapping) else None
            if isinstance(name, str):
                manifest_lookup[name] = {"auth": payload.get("auth")}

        snapshot: Dict[str, Dict[str, Any]] = {}
        getter = getattr(self._config_manager, "get_tool_config_snapshot", None)
        if callable(getter):
            try:
                snapshot_candidate = getter(
                    manifest_lookup=manifest_lookup if manifest_lookup else None
                )
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.warning("Failed to load tool configuration snapshot: %s", exc)
            else:
                if isinstance(snapshot_candidate, Mapping):
                    snapshot = dict(snapshot_candidate)

        for entry in serialized:
            name = entry.get("name")
            config_record: Mapping[str, Any]
            if isinstance(name, str):
                config_candidate = snapshot.get(name, {})
                config_record = (
                    dict(config_candidate)
                    if isinstance(config_candidate, Mapping)
                    else {}
                )
            else:
                config_record = {}
            entry["settings"] = deepcopy(
                config_record.get("settings", {})
                if isinstance(config_record, Mapping)
                else {}
            )
            entry["credentials"] = deepcopy(
                config_record.get("credentials", {})
                if isinstance(config_record, Mapping)
                else {}
            )

        return {
            "count": len(serialized),
            "tools": serialized,
        }

    def get_persona_metrics(
        self,
        persona_name: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
        metric_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return aggregated persona metrics for the requested persona."""

        if not persona_name:
            raise ValueError("Persona name is required for analytics")

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 20

        limit_value = max(1, min(limit_value, 200))

        payload = get_persona_metrics(
            persona_name,
            start=start,
            end=end,
            limit_recent=limit_value,
            config_manager=self._config_manager,
        )

        persona_anomalies = []
        if isinstance(payload, Mapping):
            source = payload.get("recent_anomalies")
            if isinstance(source, list):
                persona_anomalies = [
                    dict(item)
                    for item in source
                    if isinstance(item, Mapping)
                ]
        else:
            payload = {}

        category = (metric_type or "tool").strip().lower()
        if category == "skill":
            skill_metrics = dict(payload.get("skills") or {})
            skill_source = skill_metrics.get("anomalies")
            skill_anomalies = [
                dict(item)
                for item in persona_anomalies
                if (item.get("category") or "").lower() == "skill"
            ]
            if not skill_anomalies and isinstance(skill_source, list):
                skill_anomalies = [
                    dict(item) for item in skill_source if isinstance(item, Mapping)
                ]
            skill_metrics.setdefault("category", "skill")
            skill_metrics.setdefault("totals", {"calls": 0, "success": 0, "failure": 0})
            skill_metrics.setdefault("success_rate", 0.0)
            skill_metrics.setdefault("average_latency_ms", 0.0)
            skill_metrics.setdefault("totals_by_skill", [])
            skill_metrics.setdefault("recent", [])
            skill_metrics["anomalies"] = skill_anomalies
            skill_metrics["recent_anomalies"] = skill_anomalies
            skill_metrics["persona"] = payload.get("persona", persona_name)
            skill_metrics["window"] = payload.get("window", {"start": None, "end": None})
            return skill_metrics

        if isinstance(payload, dict):
            payload.setdefault("anomalies", [])
            payload.setdefault("recent_anomalies", persona_anomalies)
        else:
            payload = {
                "anomalies": [],
                "recent_anomalies": [],
            }

        payload.setdefault("category", "tool")
        if "skills" not in payload:
            payload["skills"] = {
                "category": "skill",
                "totals": {"calls": 0, "success": 0, "failure": 0},
                "success_rate": 0.0,
                "average_latency_ms": 0.0,
                "totals_by_skill": [],
                "recent": [],
            }
        return payload

    def get_job_metrics(
        self,
        persona_name: str,
        *,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return aggregated job lifecycle analytics for ``persona_name``."""

        if not persona_name:
            raise ValueError("Persona name is required for job analytics")

        try:
            limit_value = int(limit) if limit is not None else 50
        except (TypeError, ValueError):
            limit_value = 50

        limit_value = max(1, min(limit_value, 200))

        return get_job_lifecycle_metrics(
            persona=persona_name,
            tenant_id=tenant_id,
            limit_recent=limit_value,
            config_manager=self._config_manager,
        )

    def get_persona_comparison_summary(
        self,
        *,
        category: Optional[str] = None,
        personas: Optional[Iterable[str]] = None,
        search: Optional[str] = None,
        recent: Optional[Any] = None,
        page: Optional[Any] = None,
        page_size: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return persona comparison analytics with ranking highlights."""

        category_value = str(category or "tool").strip().lower()

        try:
            recent_limit = int(recent) if recent is not None else 5
        except (TypeError, ValueError):
            recent_limit = 5

        try:
            page_number = int(page) if page is not None else 1
        except (TypeError, ValueError):
            page_number = 1

        try:
            size_value = int(page_size) if page_size is not None else 25
        except (TypeError, ValueError):
            size_value = 25

        return get_persona_comparison_summary(
            category=category_value,
            personas=personas,
            search=search,
            limit_recent=recent_limit,
            page=page_number,
            page_size=size_value,
            config_manager=self._config_manager,
        )

    def get_skills(
        self,
        *,
        persona: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return serialized skill metadata using the manifest loader."""

        registry = get_capability_registry(config_manager=self._config_manager)
        persona_tokens = _normalize_filters(persona)
        views = list(registry.query_skills(persona_filters=persona_tokens))

        serialized: List[Dict[str, Any]] = []
        manifest_lookup: Dict[str, Dict[str, Any]] = {}

        for view in views:
            payload = _serialize_skill(view)
            serialized.append(payload)

            name = payload.get("name")
            manifest_entry: Dict[str, Any]
            if isinstance(name, str):
                manifest_entry = manifest_lookup.setdefault(name, {})
            else:
                continue

            candidate = getattr(view.manifest, "auth", None)
            if isinstance(candidate, Mapping):
                manifest_entry["auth"] = candidate

        snapshot: Dict[str, Dict[str, Any]] = {}
        getter = getattr(self._config_manager, "get_skill_config_snapshot", None)
        if callable(getter):
            try:
                snapshot_candidate = getter(
                    manifest_lookup=manifest_lookup if manifest_lookup else None,
                    skill_names=[entry.get("name", "") for entry in serialized],
                )
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.warning("Failed to load skill configuration snapshot: %s", exc)
            else:
                if isinstance(snapshot_candidate, Mapping):
                    snapshot = dict(snapshot_candidate)

        for entry in serialized:
            name = entry.get("name")
            config_record: Mapping[str, Any]
            if isinstance(name, str):
                config_candidate = snapshot.get(name, {})
                config_record = (
                    dict(config_candidate)
                    if isinstance(config_candidate, Mapping)
                    else {}
                )
            else:
                config_record = {}

            entry["settings"] = deepcopy(
                config_record.get("settings", {})
                if isinstance(config_record, Mapping)
                else {}
            )
            entry["credentials"] = deepcopy(
                config_record.get("credentials", {})
                if isinstance(config_record, Mapping)
                else {}
            )

        return {
            "count": len(serialized),
            "skills": serialized,
        }

    def get_skill_details(
        self,
        skill_identifier: Optional[str] = None,
        *,
        name: Optional[str] = None,
        skill: Optional[str] = None,
        skill_name: Optional[str] = None,
        persona: Optional[Any] = None,
        context: Any = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return manifest metadata and persisted review state for a single skill."""

        payload_map = payload if isinstance(payload, Mapping) else {}
        persona_value = persona
        if persona_value is None and isinstance(payload_map, Mapping):
            candidate = payload_map.get("persona")
            if isinstance(candidate, str):
                persona_value = candidate

        identifier = self._resolve_skill_identifier(
            skill_identifier,
            name,
            skill,
            skill_name,
            payload_map.get("name") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill_name") if isinstance(payload_map, Mapping) else None,
        )

        persona_token = str(persona_value).strip() if isinstance(persona_value, str) else None

        try:
            view = self._find_skill_view(identifier, persona_token)
        except KeyError as exc:
            raise KeyError(str(exc))

        request_context = self._resolve_request_context(context)
        metadata_snapshot = self._load_persisted_skill_metadata(identifier)

        instruction_prompt = getattr(view.manifest, "instruction_prompt", None)
        if isinstance(instruction_prompt, str):
            prompt_text = instruction_prompt
        else:
            prompt_text = None

        response: Dict[str, Any] = {
            "success": True,
            "skill": _serialize_skill(view),
            "metadata": metadata_snapshot,
            "tenant_id": request_context.tenant_id,
        }
        if prompt_text is not None:
            response["instruction_prompt"] = prompt_text
        return response

    def validate_skill(
        self,
        skill_identifier: Optional[str] = None,
        *,
        name: Optional[str] = None,
        skill: Optional[str] = None,
        skill_name: Optional[str] = None,
        persona: Optional[Any] = None,
        context: Any = None,
        payload: Optional[Mapping[str, Any]] = None,
        tool_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        timeout_seconds: Optional[Any] = None,
        budget_ms: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute a dry-run validation of a skill manifest using ``use_skill``."""

        payload_map = payload if isinstance(payload, Mapping) else {}
        persona_value = persona
        if persona_value is None:
            candidate = payload_map.get("persona") if isinstance(payload_map, Mapping) else None
            if isinstance(candidate, str):
                persona_value = candidate

        identifier = self._resolve_skill_identifier(
            skill_identifier,
            name,
            skill,
            skill_name,
            payload_map.get("name") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill_name") if isinstance(payload_map, Mapping) else None,
        )

        context_payload = context or (
            payload_map.get("context") if isinstance(payload_map, Mapping) else None
        )
        request_context = self._resolve_request_context(context_payload)

        persona_token = str(persona_value).strip() if isinstance(persona_value, str) else None
        try:
            view = self._find_skill_view(identifier, persona_token)
        except KeyError as exc:
            return {"success": False, "error": str(exc)}

        manifest_persona = getattr(view.manifest, "persona", None)
        persona_scope = persona_token or (
            str(manifest_persona).strip() if isinstance(manifest_persona, str) else None
        )

        persona_payload: Any = None
        if persona_scope:
            try:
                persona_payload = load_persona_definition(
                    persona_scope,
                    config_manager=self._config_manager,
                )
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.warning(
                    "Failed to load persona '%s' for skill '%s': %s",
                    persona_scope,
                    identifier,
                    exc,
                )
                persona_payload = persona_scope
            else:
                if persona_payload is None:
                    persona_payload = persona_scope
        else:
            persona_payload = manifest_persona

        conversation_id = payload_map.get("conversation_id") if isinstance(payload_map, Mapping) else None
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            conversation_id = f"skill-validation:{identifier}"

        conversation_history: List[Mapping[str, Any]] = []
        history_payload = (
            payload_map.get("conversation_history") if isinstance(payload_map, Mapping) else None
        )
        if isinstance(history_payload, Iterable) and not isinstance(
            history_payload, (str, bytes, bytearray)
        ):
            for entry in history_payload:
                if isinstance(entry, Mapping):
                    conversation_history.append(dict(entry))

        user_block = None
        raw_user = payload_map.get("user") if isinstance(payload_map, Mapping) else None
        if isinstance(raw_user, Mapping):
            user_block = dict(raw_user)
        elif request_context.user_id:
            user_block = {"id": request_context.user_id}
        if user_block is not None and request_context.roles:
            user_block.setdefault("roles", list(request_context.roles))

        metadata_block: Dict[str, Any] = {}
        raw_metadata = payload_map.get("metadata") if isinstance(payload_map, Mapping) else None
        if isinstance(raw_metadata, Mapping):
            metadata_block = dict(raw_metadata)
        metadata_block.setdefault("tenant_id", request_context.tenant_id)
        if request_context.user_id:
            metadata_block.setdefault("user_id", request_context.user_id)

        if tool_inputs is None and isinstance(payload_map, Mapping):
            raw_inputs = payload_map.get("tool_inputs")
        else:
            raw_inputs = tool_inputs

        sanitized_inputs: Optional[Dict[str, Mapping[str, Any]]] = None
        if isinstance(raw_inputs, Mapping):
            sanitized_inputs = {}
            for key, value in raw_inputs.items():
                if not isinstance(value, Mapping):
                    continue
                sanitized_inputs[str(key)] = dict(value)

        if timeout_seconds is None and isinstance(payload_map, Mapping):
            timeout_seconds = payload_map.get("timeout_seconds")
        if budget_ms is None and isinstance(payload_map, Mapping):
            budget_ms = payload_map.get("budget_ms")

        try:
            timeout_value = float(timeout_seconds) if timeout_seconds is not None else None
        except (TypeError, ValueError):
            timeout_value = None
        try:
            budget_value = float(budget_ms) if budget_ms is not None else None
        except (TypeError, ValueError):
            budget_value = None

        from ATLAS.SkillManager import SkillExecutionContext

        execution_context = SkillExecutionContext(
            conversation_id=str(conversation_id),
            conversation_history=list(conversation_history),
            persona=persona_payload,
            user=user_block,
            metadata=metadata_block,
        )

        try:
            result = self._execute_skill_preview(
                view.manifest,
                context=execution_context,
                tool_inputs=sanitized_inputs,
                timeout_seconds=timeout_value,
                budget_ms=budget_value,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "Skill validation for '%s' failed: %s",
                identifier,
                exc,
                exc_info=True,
            )
            message = str(exc) or "Skill validation failed."
            return {"success": False, "error": message}

        result_payload = {
            "skill": result.skill_name,
            "tool_results": dict(result.tool_results),
            "metadata": dict(result.metadata),
            "version": result.version,
            "required_capabilities": list(result.required_capabilities),
        }

        return {
            "success": True,
            "result": result_payload,
            "skill": _serialize_skill(view),
            "metadata": self._load_persisted_skill_metadata(identifier),
        }

    def run_skill_test(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compatibility wrapper aliasing :meth:`validate_skill`."""

        return self.validate_skill(*args, **kwargs)

    def set_skill_metadata(
        self,
        skill_identifier: Optional[str] = None,
        *,
        name: Optional[str] = None,
        skill: Optional[str] = None,
        skill_name: Optional[str] = None,
        persona: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        context: Any = None,
    ) -> Dict[str, Any]:
        """Persist tester notes or review status changes for a skill."""

        payload_map = payload if isinstance(payload, Mapping) else {}

        identifier = self._resolve_skill_identifier(
            skill_identifier,
            name,
            skill,
            skill_name,
            payload_map.get("name") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill") if isinstance(payload_map, Mapping) else None,
            payload_map.get("skill_name") if isinstance(payload_map, Mapping) else None,
        )

        if metadata is None:
            candidate_metadata = payload_map.get("metadata") if isinstance(payload_map, Mapping) else None
            if isinstance(candidate_metadata, Mapping):
                metadata = candidate_metadata

        if metadata is None:
            raise ValueError("Metadata payload is required when updating skill metadata.")
        if not isinstance(metadata, Mapping):
            raise TypeError("Skill metadata payload must be a mapping.")

        persona_value = persona
        if persona_value is None:
            candidate = payload_map.get("persona") if isinstance(payload_map, Mapping) else None
            if isinstance(candidate, str):
                persona_value = candidate

        context_payload = context or (
            payload_map.get("context") if isinstance(payload_map, Mapping) else None
        )
        request_context = self._resolve_request_context(context_payload)

        persona_token = str(persona_value).strip() if isinstance(persona_value, str) else None
        try:
            view = self._find_skill_view(identifier, persona_token)
        except KeyError as exc:
            return {"success": False, "error": str(exc)}

        manifest_persona = getattr(view.manifest, "persona", None)
        persona_for_logging = persona_token or (
            str(manifest_persona).strip() if isinstance(manifest_persona, str) else "shared"
        )
        if not persona_for_logging:
            persona_for_logging = "shared"

        metadata_payload: Dict[str, Any] = {}
        if "review_status" in metadata:
            metadata_payload["review_status"] = metadata.get("review_status")
        if "tester_notes" in metadata:
            metadata_payload["tester_notes"] = metadata.get("tester_notes")

        previous_metadata = self._load_persisted_skill_metadata(identifier)

        setter = getattr(self._config_manager, "set_skill_metadata", None)
        if not callable(setter):
            raise RuntimeError("Skill metadata persistence is not configured")

        try:
            setter(identifier, metadata_payload)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "Failed to persist metadata for skill '%s': %s",
                identifier,
                exc,
                exc_info=True,
            )
            message = str(exc) or "Unable to persist metadata."
            return {"success": False, "error": message}

        persisted_metadata = self._load_persisted_skill_metadata(identifier)

        audit_entry = None
        try:
            audit_entry = get_skill_audit_logger().record_change(
                persona_for_logging,
                identifier,
                old_review_status=previous_metadata.get("review_status"),
                new_review_status=persisted_metadata.get("review_status"),
                old_tester_notes=previous_metadata.get("tester_notes"),
                new_tester_notes=persisted_metadata.get("tester_notes"),
                username=request_context.user_id,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "Failed to record skill audit entry for '%s': %s",
                identifier,
                exc,
                exc_info=True,
            )
            audit_entry = None

        response: Dict[str, Any] = {
            "success": True,
            "metadata": persisted_metadata,
            "skill": _serialize_skill(view),
        }
        if audit_entry is not None:
            response["audit"] = asdict(audit_entry)
        return response

    def get_task_catalog(
        self,
        *,
        persona: Optional[Any] = None,
        tags: Optional[Any] = None,
        required_skills: Optional[Any] = None,
        required_tools: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return manifest-backed task metadata for UI catalogs."""

        registry = get_capability_registry(config_manager=self._config_manager)
        persona_tokens = _normalize_filters(persona)
        tag_tokens = _normalize_filters(tags)
        skill_tokens = _normalize_filters(required_skills)
        tool_tokens = _normalize_filters(required_tools)

        views = registry.query_tasks(
            persona_filters=persona_tokens or None,
            required_skill_filters=skill_tokens or None,
            required_tool_filters=tool_tokens or None,
            tag_filters=tag_tokens or None,
        )

        serialized: List[Dict[str, Any]] = []
        for view in sorted(
            views,
            key=lambda item: (
                (getattr(item.manifest, "persona", None) or ""),
                item.manifest.name.lower(),
            ),
        ):
            manifest = view.manifest
            escalation = manifest.escalation_policy
            escalation_payload: Optional[Dict[str, Any]]
            if isinstance(escalation, Mapping):
                escalation_payload = dict(escalation)
            else:
                escalation_payload = None

            serialized.append(
                {
                    "name": manifest.name,
                    "persona": manifest.persona,
                    "summary": manifest.summary,
                    "description": manifest.description,
                    "required_skills": list(view.required_skills),
                    "required_tools": list(view.required_tools),
                    "acceptance_criteria": list(manifest.acceptance_criteria),
                    "escalation_policy": escalation_payload,
                    "tags": list(view.tags),
                    "priority": manifest.priority,
                    "source": manifest.source,
                }
            )

        return {"count": len(serialized), "tasks": serialized}

    # ------------------------------------------------------------------
    # Blackboard endpoints
    # ------------------------------------------------------------------
    def get_blackboard_entries(
        self,
        scope_type: str,
        scope_id: str,
        *,
        summary: bool = False,
        category: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return entries or a summary for the requested blackboard scope."""

        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        if summary:
            return client.summary()

        category_value = None
        if category is not None:
            category_value = str(category).strip() or None
        entries = client.list_entries(category=category_value)
        return {
            "scope_id": client.scope_id,
            "scope_type": client.scope_type,
            "entries": entries,
            "count": len(entries),
        }

    def create_blackboard_entry(
        self,
        scope_type: str,
        scope_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        """Persist a new blackboard entry and return the serialized record."""

        category = payload.get("category")
        title = payload.get("title")
        content = payload.get("content")
        if not category:
            raise ValueError("category is required")
        if not title or not content:
            raise ValueError("title and content are required")

        payload_map = dict(payload)
        metadata_payload: Dict[str, Any] | None = None
        raw_metadata = payload_map.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata_payload = dict(raw_metadata)
        request_context = None
        if context is not None:
            request_context = self._resolve_request_context(context)
        if request_context is not None:
            metadata_payload = metadata_payload or {}
            metadata_payload.setdefault("tenant_id", request_context.tenant_id)
        if metadata_payload:
            payload_map["metadata"] = metadata_payload
        else:
            payload_map.pop("metadata", None)

        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        entry = client.publish(
            category,
            str(title),
            str(content),
            author=payload_map.get("author"),
            tags=payload_map.get("tags"),
            metadata=payload_map.get("metadata"),
        )

        return {"success": True, "entry": entry}

    def update_blackboard_entry(
        self,
        scope_type: str,
        scope_id: str,
        entry_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        """Update an existing blackboard entry."""

        payload_map = dict(payload)
        metadata_payload = None
        raw_metadata = payload_map.get("metadata")
        if isinstance(raw_metadata, Mapping):
            metadata_payload = dict(raw_metadata)
        request_context = None
        if context is not None:
            request_context = self._resolve_request_context(context)
        if request_context is not None and metadata_payload is not None:
            metadata_payload.setdefault("tenant_id", request_context.tenant_id)
            payload_map["metadata"] = metadata_payload
        elif metadata_payload is None:
            payload_map.pop("metadata", None)

        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        entry = client.update_entry(
            entry_id,
            title=payload_map.get("title"),
            content=payload_map.get("content"),
            tags=payload_map.get("tags"),
            metadata=payload_map.get("metadata"),
        )
        if entry is None:
            raise KeyError(f"Unknown blackboard entry: {entry_id}")
        return {"success": True, "entry": entry}

    def delete_blackboard_entry(
        self,
        scope_type: str,
        scope_id: str,
        entry_id: str,
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        """Delete the requested blackboard entry."""

        if context is not None:
            self._resolve_request_context(context)
        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        success = client.delete_entry(entry_id)
        return {"success": bool(success)}

    def stream_blackboard_events(self, scope_type: str, scope_id: str):
        """Return an asynchronous iterator of blackboard events."""

        return stream_blackboard(scope_id, scope_type=scope_type)

    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]:
        """Return the current review status for ``persona_name``."""

        if not persona_name:
            raise ValueError("Persona name is required for review status")

        status = compute_review_status(
            persona_name,
            audit_logger=get_persona_audit_logger(),
            review_logger=get_persona_review_logger(),
            review_queue=get_persona_review_queue(),
        )

        payload = asdict(status)
        payload["success"] = True
        return payload

    def attest_persona_review(
        self,
        persona_name: str,
        *,
        reviewer: str,
        expires_at: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a persona review attestation for ``persona_name``."""

        if not persona_name:
            raise ValueError("Persona name is required for review attestation")

        reviewer_name = reviewer.strip() if reviewer else ""
        if not reviewer_name:
            reviewer_name = "unknown"

        validity_days = (
            int(expires_in_days)
            if expires_in_days is not None
            else REVIEW_INTERVAL_DAYS
        )
        validity_days = max(1, validity_days)
        validity = timedelta(days=validity_days)

        parsed_expires = (
            parse_persona_timestamp(expires_at)
            if expires_at
            else None
        )

        review_logger = get_persona_review_logger()
        queue = get_persona_review_queue()

        attestation = review_logger.record_attestation(
            persona_name,
            reviewer=reviewer_name,
            expires_at=parsed_expires,
            notes=notes,
            validity=validity,
        )

        now = datetime.now(timezone.utc)
        queue.mark_completed(persona_name, timestamp=now)

        status = compute_review_status(
            persona_name,
            audit_logger=get_persona_audit_logger(),
            review_logger=review_logger,
            review_queue=queue,
            now=now,
            interval_days=validity_days,
        )

        return {
            "success": True,
            "attestation": asdict(attestation),
            "status": asdict(status),
        }

    def list_tool_changes(
        self,
        *,
        persona: Optional[Any] = None,
        limit: Optional[Any] = None,
        offset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return normalized persona tool change history."""

        persona_filter = None
        if persona is not None:
            persona_text = str(persona).strip()
            persona_filter = persona_text or None
        limit_value = self._coerce_pagination_limit(limit)
        offset_value = self._coerce_pagination_offset(offset)

        entries, total = get_persona_audit_logger().get_history(
            persona_name=persona_filter,
            offset=offset_value,
            limit=limit_value,
        )

        serialized = [self._serialize_tool_change_entry(entry) for entry in entries]

        return {
            "changes": serialized,
            "count": len(serialized),
            "total": total,
            "offset": offset_value,
            "limit": limit_value,
        }

    def list_skill_changes(
        self,
        *,
        persona: Optional[Any] = None,
        skill: Optional[Any] = None,
        limit: Optional[Any] = None,
        offset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return normalized skill review change history."""

        persona_filter = None
        if persona is not None:
            persona_text = str(persona).strip()
            persona_filter = persona_text or None

        skill_filter = None
        if skill is not None:
            skill_text = str(skill).strip()
            skill_filter = skill_text or None
        limit_value = self._coerce_pagination_limit(limit)
        offset_value = self._coerce_pagination_offset(offset)

        entries, total = get_skill_audit_logger().get_history(
            persona_name=persona_filter,
            skill_name=skill_filter,
            offset=offset_value,
            limit=limit_value,
        )

        serialized = [self._serialize_skill_change_entry(entry) for entry in entries]

        return {
            "changes": serialized,
            "count": len(serialized),
            "total": total,
            "offset": offset_value,
            "limit": limit_value,
        }

    def handle_request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Optional[Mapping[str, Any]] = None,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        """Simple request dispatcher supporting persona metadata endpoints."""

        method_upper = method.upper()
        query = query or {}

        if method_upper == "GET":
            if path == "/conversations":
                if context is None:
                    raise ConversationAuthorizationError(
                        "A tenant scoped context is required"
                    )
                return self.list_conversations(query, context=context)
            if path.startswith("/skills/") and path != "/skills/changes":
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) == 2:
                    skill_name = components[1]
                    return self.get_skill_details(
                        skill_name,
                        persona=query.get("persona"),
                        context=context,
                    )
                elif len(components) > 2:
                    raise ValueError(f"Unsupported path: {path}")
            if path.startswith("/personas/") and path.endswith("/analytics"):
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) != 3:
                    raise ValueError(f"Unsupported path: {path}")
                persona_name = components[1]
                start = _parse_query_timestamp(query.get("start"))
                end = _parse_query_timestamp(query.get("end"))
                limit = query.get("limit")
                metric_type = query.get("type")
                return self.get_persona_metrics(
                    persona_name,
                    start=start,
                    end=end,
                    limit=limit,
                    metric_type=metric_type,
                )
            if path == "/personas/analytics/comparison":
                persona_filters = _normalize_filters(query.get("persona"))
                category = query.get("type") or query.get("category")
                search = query.get("search")
                return self.get_persona_comparison_summary(
                    category=category,
                    personas=persona_filters or None,
                    search=search,
                    recent=query.get("recent"),
                    page=query.get("page"),
                    page_size=query.get("page_size"),
                )
            if path.startswith("/personas/") and path.endswith("/review"):
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) != 3:
                    raise ValueError(f"Unsupported path: {path}")
                persona_name = components[1]
                return self.get_persona_review_status(persona_name)
            if path == "/tools/changes":
                return self.list_tool_changes(
                    persona=query.get("persona"),
                    limit=query.get("limit"),
                    offset=query.get("offset"),
                )
            if path == "/skills/changes":
                return self.list_skill_changes(
                    persona=query.get("persona"),
                    skill=query.get("skill"),
                    limit=query.get("limit"),
                    offset=query.get("offset"),
                )
            if path == "/skills":
                return self.get_skills(persona=query.get("persona"))
            if path.startswith("/blackboard/"):
                scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
                if entry_id:
                    client = get_blackboard().client_for(scope_id, scope_type=scope_type)
                    entry = client.get_entry(entry_id)
                    if entry is None:
                        raise KeyError(f"Unknown blackboard entry: {entry_id}")
                    return {"entry": entry}
                return self.get_blackboard_entries(
                    scope_type,
                    scope_id,
                    summary=_as_bool(query.get("summary")),
                    category=query.get("category"),
                )
            if path != "/tools":
                raise ValueError(f"Unsupported path: {path}")
            include_health_value = (
                _as_bool(query.get("include_provider_health"))
                if "include_provider_health" in query
                else True
            )
            return self.get_tools(
                capability=query.get("capability"),
                safety_level=query.get("safety_level"),
                persona=query.get("persona"),
                provider=query.get("provider"),
                version=query.get("version"),
                min_success_rate=query.get("min_success_rate"),
                include_provider_health=include_health_value,
            )

        if method_upper == "POST":
            return self._handle_post(path, query, context=context)

        if method_upper == "PATCH":
            return self._handle_patch(path, query, context=context)

        if method_upper == "DELETE":
            return self._handle_delete(path, query, context=context)

        raise ValueError(f"Unsupported method: {method}")

    def _handle_post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        if path.startswith("/conversations/") and path.endswith("/reset"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            if context is None:
                raise ConversationAuthorizationError(
                    "A tenant scoped context is required"
                )
            conversation_id = components[1]
            return self.reset_conversation(conversation_id, context=context)

        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if entry_id:
                raise ValueError("POST requests should not target a specific entry")
            return self.create_blackboard_entry(
                scope_type,
                scope_id,
                payload,
                context=context,
            )

        if path.startswith("/personas/") and path.endswith("/tools"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            return self.update_persona_tools(
                persona_name,
                tools=payload.get("tools"),
                rationale=str(payload.get("rationale") or "Server route persona update"),
            )

        if path.startswith("/personas/") and path.endswith("/skills"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            return self.update_persona_skills(
                persona_name,
                skills=payload.get("skills"),
                rationale=str(payload.get("rationale") or "Server route persona update"),
            )

        if path.startswith("/personas/") and path.endswith("/export"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            return self.export_persona_bundle(
                persona_name,
                signing_key=str(payload.get("signing_key") or ""),
            )

        if path == "/personas/import":
            return self.import_persona_bundle(
                bundle_base64=str(payload.get("bundle") or ""),
                signing_key=str(payload.get("signing_key") or ""),
                rationale=str(payload.get("rationale") or "Imported via server route"),
            )

        if path.startswith("/personas/") and path.endswith("/review"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            expires_at = payload.get("expires_at")
            expires_in = payload.get("expires_in_days")
            reviewer = payload.get("reviewer")
            notes = payload.get("notes")
            return self.attest_persona_review(
                persona_name,
                reviewer=str(reviewer or ""),
                expires_at=str(expires_at) if expires_at else None,
                expires_in_days=int(expires_in) if expires_in is not None else None,
                notes=str(notes) if notes is not None else None,
            )

        if path.startswith("/skills/") and (
            path.endswith("/validate") or path.endswith("/test")
        ):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            skill_name = components[1]
            persona_value = payload.get("persona") if isinstance(payload, Mapping) else None
            context_payload = payload.get("context") if isinstance(payload, Mapping) else None
            return self.validate_skill(
                skill_name,
                persona=persona_value,
                payload=payload,
                context=context_payload,
            )

        raise ValueError(f"Unsupported path: {path}")

    def _handle_patch(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if not entry_id:
                raise ValueError("PATCH requests must target a specific entry")
            return self.update_blackboard_entry(
                scope_type,
                scope_id,
                entry_id,
                payload,
                context=context,
            )
        if path.startswith("/skills/") and path.endswith("/metadata"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            skill_name = components[1]
            persona_value = payload.get("persona") if isinstance(payload, Mapping) else None
            context_payload = payload.get("context") if isinstance(payload, Mapping) else None
            metadata_payload = payload.get("metadata") if isinstance(payload, Mapping) else None
            return self.set_skill_metadata(
                skill_name,
                persona=persona_value,
                metadata=metadata_payload,
                payload=payload,
                context=context_payload,
            )
        raise ValueError(f"Unsupported path: {path}")

    def _handle_delete(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        context: Any | None = None,
    ) -> Dict[str, Any]:
        if path.startswith("/conversations/"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 2:
                raise ValueError(f"Unsupported path: {path}")
            if context is None:
                raise ConversationAuthorizationError(
                    "A tenant scoped context is required"
                )
            conversation_id = components[1]
            return self.delete_conversation(conversation_id, context=context)

        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if not entry_id:
                raise ValueError("DELETE requests must target a specific entry")
            return self.delete_blackboard_entry(
                scope_type,
                scope_id,
                entry_id,
                context=context,
            )
        raise ValueError(f"Unsupported path: {path}")

    @staticmethod
    def _parse_blackboard_path(path: str) -> tuple[str, str, Optional[str]]:
        components = [part for part in path.strip("/").split("/") if part]
        if len(components) < 3 or components[0] != "blackboard":
            raise ValueError(f"Unsupported path: {path}")
        scope_type = components[1]
        scope_id = components[2]
        entry_id = components[3] if len(components) > 3 else None
        return scope_type, scope_id, entry_id

    def update_persona_tools(
        self,
        persona_name: str,
        *,
        tools: Optional[Any],
        rationale: str = "Server route persona update",
    ) -> Dict[str, Any]:
        """Update the allowed tools for a persona via server APIs."""

        if not persona_name:
            raise ValueError("Persona name is required")

        persona = load_persona_definition(
            persona_name,
            config_manager=self._config_manager,
        )
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' could not be loaded.")

        metadata_order, metadata_lookup = load_tool_metadata(
            config_manager=self._config_manager
        )
        normalised_tools = normalize_allowed_tools(
            self._normalise_tool_payload(tools),
            metadata_order=metadata_order,
        )

        persona_for_validation = dict(persona)
        persona_for_validation["allowed_tools"] = normalised_tools

        known_tools: set[str] = {str(name) for name in metadata_order}
        known_tools.update(str(name) for name in metadata_lookup.keys())
        existing_tools = persona.get("allowed_tools") or []
        known_tools.update(str(name) for name in existing_tools if str(name))

        skill_order, skill_lookup = load_skill_catalog(
            config_manager=self._config_manager
        )
        known_skills: set[str] = {str(name) for name in skill_order}
        known_skills.update(str(name) for name in skill_lookup.keys())
        existing_skills = persona.get("allowed_skills") or []
        known_skills.update(str(name) for name in existing_skills if str(name))

        try:
            _validate_persona_payload(
                {"persona": [persona_for_validation]},
                persona_name=persona_name,
                tool_ids=known_tools,
                skill_ids=known_skills,
                skill_catalog=skill_lookup,
                config_manager=self._config_manager,
            )
        except PersonaValidationError as exc:
            message = str(exc)
            logger.warning(
                "Rejected persona tool update for '%s': %s",
                persona_name,
                message,
            )
            return {"success": False, "error": message, "errors": [message]}

        persona["allowed_tools"] = normalised_tools

        persist_persona_definition(
            persona_name,
            persona,
            config_manager=self._config_manager,
            rationale=rationale,
        )

        return {
            "success": True,
            "persona": {
                "name": persona.get("name", persona_name),
                "allowed_tools": normalised_tools,
            },
        }

    @staticmethod
    def _normalise_tool_payload(raw: Optional[Any]) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, Mapping):
            return list(raw.values())
        if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
            return list(raw)
        return [str(raw)]

    def update_persona_skills(
        self,
        persona_name: str,
        *,
        skills: Optional[Any],
        rationale: str = "Server route persona update",
    ) -> Dict[str, Any]:
        """Update the allowed skills for a persona via server APIs."""

        if not persona_name:
            raise ValueError("Persona name is required")

        persona = load_persona_definition(
            persona_name,
            config_manager=self._config_manager,
        )
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' could not be loaded.")

        skill_order, skill_lookup = load_skill_catalog(
            config_manager=self._config_manager
        )
        normalized_skills = normalize_allowed_skills(
            self._normalise_skill_payload(skills),
            metadata_order=skill_order,
        )

        persona_for_validation = dict(persona)
        persona_for_validation["allowed_skills"] = normalized_skills

        known_skills: set[str] = {str(name) for name in skill_order}
        known_skills.update(str(name) for name in skill_lookup.keys())
        existing_skills = persona.get('allowed_skills') or []
        known_skills.update(str(name) for name in existing_skills if str(name))

        tool_order, tool_lookup = load_tool_metadata(
            config_manager=self._config_manager
        )
        known_tools: set[str] = {str(name) for name in tool_order}
        known_tools.update(str(name) for name in tool_lookup.keys())
        existing_tools = persona.get('allowed_tools') or []
        known_tools.update(str(name) for name in existing_tools if str(name))

        try:
            _validate_persona_payload(
                {"persona": [persona_for_validation]},
                persona_name=persona_name,
                tool_ids=known_tools,
                skill_ids=known_skills,
                skill_catalog=skill_lookup,
                config_manager=self._config_manager,
            )
        except PersonaValidationError as exc:
            message = str(exc)
            logger.warning(
                "Rejected persona skill update for '%s': %s",
                persona_name,
                message,
            )
            return {"success": False, "error": message, "errors": [message]}

        persona["allowed_skills"] = normalized_skills

        persist_persona_definition(
            persona_name,
            persona,
            config_manager=self._config_manager,
            rationale=rationale,
        )

        return {
            "success": True,
            "persona": {
                "name": persona.get("name", persona_name),
                "allowed_skills": normalized_skills,
            },
        }

    @staticmethod
    def _normalise_skill_payload(raw: Optional[Any]) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, Mapping):
            return list(raw.values())
        if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
            return list(raw)
        return [str(raw)]

    def export_persona_bundle(
        self,
        persona_name: str,
        *,
        signing_key: str,
    ) -> Dict[str, Any]:
        if not persona_name:
            raise ValueError("Persona name is required for export")

        try:
            bundle_bytes, persona = export_persona_bundle_bytes(
                persona_name,
                signing_key=signing_key,
                config_manager=self._config_manager,
            )
        except PersonaBundleError as exc:
            return {"success": False, "error": str(exc)}

        encoded = base64.b64encode(bundle_bytes).decode("ascii")
        return {
            "success": True,
            "persona": persona,
            "bundle": encoded,
        }

    def import_persona_bundle(
        self,
        *,
        bundle_base64: str,
        signing_key: str,
        rationale: str = "Imported via server route",
    ) -> Dict[str, Any]:
        if not bundle_base64:
            raise ValueError("Bundle payload is required for import")

        try:
            bundle_bytes = base64.b64decode(bundle_base64)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Bundle payload is not valid base64 data") from exc

        try:
            result = import_persona_bundle_bytes(
                bundle_bytes,
                signing_key=signing_key,
                config_manager=self._config_manager,
                rationale=rationale,
            )
        except PersonaBundleError as exc:
            return {"success": False, "error": str(exc)}

        result.setdefault("success", True)
        return result

    # -- conversation routes -------------------------------------------------

    def run_conversation_retention(self, *, context: Any) -> Dict[str, Any]:
        """Trigger conversation-store retention policies on demand."""

        repository = self._conversation_repository
        if repository is None:
            repository = self._build_conversation_repository()
            if repository is None:
                raise RuntimeError("Conversation store repository is not configured")
            self._conversation_repository = repository

        request_context = self._coerce_context(context)
        normalized_roles = {role.lower() for role in request_context.roles}
        if not normalized_roles.intersection({"admin", "system"}):
            raise ConversationAuthorizationError(
                "Administrative role required to trigger retention policies"
            )

        return repository.run_retention(now=datetime.now(timezone.utc))

    def list_conversations(
        self,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``GET /conversations`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.list_conversations(params, context=request_context)

    def reset_conversation(
        self,
        conversation_id: str,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``POST /conversations/{id}/reset`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.reset_conversation(conversation_id, context=request_context)

    def delete_conversation(
        self,
        conversation_id: str,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``DELETE /conversations/{id}`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.delete_conversation(conversation_id, context=request_context)

    def create_message(
        self,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``POST /messages`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.create_message(payload, context=request_context)

    def update_message(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``PATCH /messages/{id}`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.update_message(message_id, payload, context=request_context)

    def delete_message(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``DELETE /messages/{id}`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.delete_message(message_id, payload, context=request_context)

    def list_messages(
        self,
        conversation_id: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``GET /conversations/{id}/messages`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.list_messages(conversation_id, params, context=request_context)

    def search_conversations(
        self,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        """Handle ``POST /conversations/search`` requests."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.search_conversations(payload, context=request_context)

    def stream_conversation_events(
        self,
        conversation_id: str,
        *,
        context: Any,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream conversation message events via Server-Sent Events."""

        routes = self._get_conversation_routes()
        request_context = self._coerce_context(context)
        return routes.stream_message_events(
            conversation_id,
            context=request_context,
            after=after,
        )

    # -- job routes -------------------------------------------------------

    def create_job(
        self,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.create_job(payload, context=request_context)

    def update_job(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.update_job(job_id, payload, context=request_context)

    def transition_job(
        self,
        job_id: str,
        target_status: Any,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.transition_job(
            job_id,
            target_status,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def pause_job_schedule(
        self,
        job_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.pause_schedule(
            job_id,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def resume_job_schedule(
        self,
        job_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.resume_schedule(
            job_id,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def rerun_job(
        self,
        job_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.rerun_job(
            job_id,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def run_job_now(
        self,
        job_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.run_now(
            job_id,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def delete_job(
        self,
        job_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        return self.transition_job(
            job_id,
            target_status="cancelled",
            context=context,
            expected_updated_at=expected_updated_at,
        )

    def get_job(
        self,
        job_id: str,
        *,
        context: Any,
        include_schedule: bool = False,
        include_runs: bool = False,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.get_job(
            job_id,
            context=request_context,
            include_schedule=include_schedule,
            include_runs=include_runs,
            include_events=include_events,
        )

    def list_jobs(
        self,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._ensure_job_routes()
        request_context = self._coerce_context(context)
        if routes is None:
            return {"items": []}
        return routes.list_jobs(params, context=request_context)

    def list_job_tasks(
        self,
        job_id: str,
        *,
        context: Any,
    ) -> list[Dict[str, Any]]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.list_linked_tasks(job_id, context=request_context)

    def link_job_task(
        self,
        job_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.link_task(job_id, payload, context=request_context)

    def unlink_job_task(
        self,
        job_id: str,
        *,
        context: Any,
        link_id: Any | None = None,
        task_id: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.unlink_task(
            job_id,
            context=request_context,
            link_id=link_id,
            task_id=task_id,
        )

    def stream_job_events(
        self,
        job_id: str,
        *,
        context: Any,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        routes = self._require_job_routes()
        request_context = self._coerce_context(context)
        return routes.stream_job_events(job_id, context=request_context, after=after)

    # -- task routes -------------------------------------------------------

    def create_task(
        self,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.create_task(payload, context=request_context)

    def update_task(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.update_task(task_id, payload, context=request_context)

    def transition_task(
        self,
        task_id: str,
        target_status: Any,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.transition_task(
            task_id,
            target_status,
            context=request_context,
            expected_updated_at=expected_updated_at,
        )

    def delete_task(
        self,
        task_id: str,
        *,
        context: Any,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        return self.transition_task(
            task_id,
            target_status="cancelled",
            context=context,
            expected_updated_at=expected_updated_at,
        )

    def get_task(
        self,
        task_id: str,
        *,
        context: Any,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.get_task(task_id, context=request_context, include_events=include_events)

    def list_tasks(
        self,
        params: Optional[Mapping[str, Any]] = None,
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._ensure_task_routes()
        request_context = self._coerce_context(context)
        if routes is None:
            return {"items": []}
        return routes.list_tasks(params, context=request_context)

    def search_tasks(
        self,
        payload: Mapping[str, Any],
        *,
        context: Any,
    ) -> Dict[str, Any]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.search_tasks(payload, context=request_context)

    def stream_task_events(
        self,
        task_id: str,
        *,
        context: Any,
        after: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        routes = self._require_task_routes()
        request_context = self._coerce_context(context)
        return routes.stream_task_events(task_id, context=request_context, after=after)


def _filter_entries(
    entries: Iterable[ToolCapabilityView],
    *,
    capability_tokens: List[str],
    safety_tokens: List[str],
    persona_tokens: List[str],
) -> List[ToolCapabilityView]:
    filtered: List[ToolCapabilityView] = []
    for view in entries:
        manifest = view.manifest
        if capability_tokens and not _capabilities_match(manifest, capability_tokens):
            continue
        if safety_tokens and not _safety_matches(manifest, safety_tokens):
            continue
        if persona_tokens and not _persona_matches(manifest, persona_tokens):
            continue
        filtered.append(view)
    return filtered


def _capabilities_match(entry: Any, tokens: List[str]) -> bool:
    capabilities = {cap.lower() for cap in getattr(entry, "capabilities", [])}
    return all(token in capabilities for token in tokens)


def _safety_matches(entry: Any, tokens: List[str]) -> bool:
    safety = str(getattr(entry, "safety_level", "") or "").lower()
    return bool(safety) and safety in tokens


def _persona_matches(entry: Any, tokens: List[str]) -> bool:
    return persona_matches_filter(getattr(entry, "persona", None), tokens)


def _normalize_filters(values: Optional[Any]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    elif isinstance(values, Mapping):
        values = list(values.values())
    elif not isinstance(values, Iterable):
        values = [values]

    tokens: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip().lower()
        else:
            text = str(value).strip().lower()
        if text:
            tokens.append(text)
    return tokens


def _parse_success_rate(raw_value: Optional[Any]) -> Optional[float]:
    if raw_value is None:
        return None
    try:
        value = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        if value <= 100.0:
            value /= 100.0
        else:
            value = 1.0
    return max(0.0, min(value, 1.0))


def _serialize_health(
    health: Mapping[str, Any], *, include_provider_health: bool = True
) -> Dict[str, Any]:
    tool_metrics = health.get("tool") if isinstance(health, Mapping) else None
    providers = health.get("providers") if isinstance(health, Mapping) else None
    last_invocation = (
        health.get("last_invocation") if isinstance(health, Mapping) else None
    )

    serialized_providers: Dict[str, Any] = {}
    if include_provider_health and isinstance(providers, Mapping):
        for name, payload in providers.items():
            if not isinstance(payload, Mapping):
                continue
            metrics = payload.get("metrics")
            router = payload.get("router")
            last_call = payload.get("last_call")
            serialized_providers[str(name)] = {
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
                "router": dict(router) if isinstance(router, Mapping) else {},
                "last_call": dict(last_call) if isinstance(last_call, Mapping) else {},
            }

    serialized_last_invocation = (
        dict(last_invocation) if isinstance(last_invocation, Mapping) else {}
    )

    payload = {
        "tool": dict(tool_metrics) if isinstance(tool_metrics, Mapping) else {},
        "providers": serialized_providers,
    }
    if serialized_last_invocation or include_provider_health:
        payload["last_invocation"] = serialized_last_invocation
    return payload


def _serialize_entry(
    view: ToolCapabilityView, *, include_provider_health: bool
) -> Dict[str, Any]:
    entry = view.manifest
    payload = {
        "name": entry.name,
        "persona": entry.persona,
        "description": entry.description,
        "version": entry.version,
        "capabilities": entry.capabilities,
        "auth": entry.auth,
        "auth_required": entry.auth_required,
        "safety_level": entry.safety_level,
        "requires_consent": entry.requires_consent,
        "allow_parallel": entry.allow_parallel,
        "idempotency_key": entry.idempotency_key,
        "default_timeout": entry.default_timeout,
        "side_effects": entry.side_effects,
        "cost_per_call": entry.cost_per_call,
        "cost_unit": entry.cost_unit,
        "persona_allowlist": entry.persona_allowlist,
        "providers": entry.providers,
        "source": entry.source,
        "capability_tags": list(view.capability_tags),
        "auth_scopes": list(view.auth_scopes),
        "health": _serialize_health(
            view.health, include_provider_health=include_provider_health
        ),
    }
    return payload


def _serialize_skill(view: SkillCapabilityView) -> Dict[str, Any]:
    entry = view.manifest
    return {
        "name": entry.name,
        "persona": entry.persona,
        "version": entry.version,
        "instruction_prompt": entry.instruction_prompt,
        "required_tools": entry.required_tools,
        "required_capabilities": list(view.required_capabilities),
        "safety_notes": entry.safety_notes,
        "summary": entry.summary,
        "category": entry.category,
        "capability_tags": list(view.capability_tags),
        "source": entry.source,
    }


atlas_server = AtlasServer()

__all__ = ["AtlasServer", "atlas_server"]
