"""Lightweight server routing helpers for tool metadata and analytics."""

from __future__ import annotations

import base64
import binascii
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
from modules.analytics.persona_metrics import get_persona_metrics
from modules.conversation_store import ConversationStoreRepository
from modules.logging.audit import (
    get_persona_audit_logger,
    get_persona_review_logger,
    get_persona_review_queue,
    parse_persona_timestamp,
)
from modules.logging.logger import setup_logger
from modules.orchestration.blackboard import get_blackboard, stream_blackboard
from modules.orchestration.capability_registry import (
    ToolCapabilityView,
    SkillCapabilityView,
    get_capability_registry,
)
from modules.orchestration.job_manager import JobManager
from modules.orchestration.job_scheduler import JobScheduler
from modules.orchestration.message_bus import MessageBus

if TYPE_CHECKING:
    from modules.job_store.service import JobService
    from modules.task_store.service import TaskService
from modules.persona_review import REVIEW_INTERVAL_DAYS, compute_review_status
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
        getter = getattr(self._config_manager, "get_conversation_store_session_factory", None)
        if not callable(getter):
            return None
        session_factory = getter()
        if session_factory is None:
            return None
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

    def get_tools(
        self,
        *,
        capability: Optional[Any] = None,
        safety_level: Optional[Any] = None,
        persona: Optional[Any] = None,
        provider: Optional[Any] = None,
        version: Optional[Any] = None,
        min_success_rate: Optional[Any] = None,
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
        return {
            "count": len(filtered),
            "tools": [_serialize_entry(entry) for entry in filtered],
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

        category = (metric_type or "tool").strip().lower()
        if category == "skill":
            skill_metrics = dict(payload.get("skills") or {})
            skill_metrics.setdefault("category", "skill")
            skill_metrics.setdefault("totals", {"calls": 0, "success": 0, "failure": 0})
            skill_metrics.setdefault("success_rate", 0.0)
            skill_metrics.setdefault("average_latency_ms", 0.0)
            skill_metrics.setdefault("totals_by_skill", [])
            skill_metrics.setdefault("recent", [])
            skill_metrics["persona"] = payload.get("persona", persona_name)
            skill_metrics["window"] = payload.get("window", {"start": None, "end": None})
            return skill_metrics

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

    def get_skills(
        self,
        *,
        persona: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return serialized skill metadata using the manifest loader."""

        registry = get_capability_registry(config_manager=self._config_manager)
        persona_tokens = _normalize_filters(persona)
        entries = registry.query_skills(persona_filters=persona_tokens)

        return {
            "count": len(entries),
            "skills": [_serialize_skill(entry) for entry in entries],
        }

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
    ) -> Dict[str, Any]:
        """Persist a new blackboard entry and return the serialized record."""

        category = payload.get("category")
        title = payload.get("title")
        content = payload.get("content")
        if not category:
            raise ValueError("category is required")
        if not title or not content:
            raise ValueError("title and content are required")

        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        entry = client.publish(
            category,
            str(title),
            str(content),
            author=payload.get("author"),
            tags=payload.get("tags"),
            metadata=payload.get("metadata"),
        )

        return {"success": True, "entry": entry}

    def update_blackboard_entry(
        self,
        scope_type: str,
        scope_id: str,
        entry_id: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing blackboard entry."""

        client = get_blackboard().client_for(scope_id, scope_type=scope_type)
        entry = client.update_entry(
            entry_id,
            title=payload.get("title"),
            content=payload.get("content"),
            tags=payload.get("tags"),
            metadata=payload.get("metadata"),
        )
        if entry is None:
            raise KeyError(f"Unknown blackboard entry: {entry_id}")
        return {"success": True, "entry": entry}

    def delete_blackboard_entry(
        self,
        scope_type: str,
        scope_id: str,
        entry_id: str,
    ) -> Dict[str, Any]:
        """Delete the requested blackboard entry."""

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

    def handle_request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simple request dispatcher supporting persona metadata endpoints."""

        method_upper = method.upper()
        query = query or {}

        if method_upper == "GET":
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
            if path.startswith("/personas/") and path.endswith("/review"):
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) != 3:
                    raise ValueError(f"Unsupported path: {path}")
                persona_name = components[1]
                return self.get_persona_review_status(persona_name)
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
            return self.get_tools(
                capability=query.get("capability"),
                safety_level=query.get("safety_level"),
                persona=query.get("persona"),
            )

        if method_upper == "POST":
            return self._handle_post(path, query)

        if method_upper == "PATCH":
            return self._handle_patch(path, query)

        if method_upper == "DELETE":
            return self._handle_delete(path, query)

        raise ValueError(f"Unsupported method: {method}")

    def _handle_post(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if entry_id:
                raise ValueError("POST requests should not target a specific entry")
            return self.create_blackboard_entry(scope_type, scope_id, payload)

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

        raise ValueError(f"Unsupported path: {path}")

    def _handle_patch(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if not entry_id:
                raise ValueError("PATCH requests must target a specific entry")
            return self.update_blackboard_entry(scope_type, scope_id, entry_id, payload)
        raise ValueError(f"Unsupported path: {path}")

    def _handle_delete(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if path.startswith("/blackboard/"):
            scope_type, scope_id, entry_id = self._parse_blackboard_path(path)
            if not entry_id:
                raise ValueError("DELETE requests must target a specific entry")
            return self.delete_blackboard_entry(scope_type, scope_id, entry_id)
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
    persona_value = getattr(entry, "persona", None)
    persona_token = (persona_value or "shared").lower()
    shared_exclusions = {
        "-shared",
        "!shared",
        "no-shared",
        "without-shared",
        "shared=false",
        "shared:false",
    }
    exclude_shared = any(token in shared_exclusions for token in tokens)
    positive_tokens = [token for token in tokens if token not in shared_exclusions]
    if persona_token == "shared":
        return not exclude_shared
    if not positive_tokens:
        return True
    return persona_token in positive_tokens


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


def _serialize_health(health: Mapping[str, Any]) -> Dict[str, Any]:
    tool_metrics = health.get("tool") if isinstance(health, Mapping) else None
    providers = health.get("providers") if isinstance(health, Mapping) else None

    serialized_providers: Dict[str, Any] = {}
    if isinstance(providers, Mapping):
        for name, payload in providers.items():
            if not isinstance(payload, Mapping):
                continue
            metrics = payload.get("metrics")
            router = payload.get("router")
            serialized_providers[str(name)] = {
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
                "router": dict(router) if isinstance(router, Mapping) else {},
            }

    return {
        "tool": dict(tool_metrics) if isinstance(tool_metrics, Mapping) else {},
        "providers": serialized_providers,
    }


def _serialize_entry(view: ToolCapabilityView) -> Dict[str, Any]:
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
        "health": _serialize_health(view.health),
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
