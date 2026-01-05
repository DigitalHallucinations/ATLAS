"""Tooling service consolidating tool, skill, and job orchestration helpers."""

from __future__ import annotations

import copy
import inspect
import types
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional

from core import ToolManager as ToolManagerModule
from core.config import ConfigManager
from core.persona_manager import PersonaManager
from core.utils import normalize_sequence
from modules.Skills import load_skill_metadata
from modules.orchestration.capability_registry import get_capability_registry
from modules.orchestration.job_manager import JobManager
from modules.orchestration.job_scheduler import JobScheduler
from modules.orchestration.task_manager import TaskManager
from modules.Jobs.manifest_loader import load_job_metadata


class ToolingService:
    """Encapsulate tool/skill configuration and job orchestration helpers."""

    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        tool_manager_module: types.ModuleType,
        persona_manager: PersonaManager | None,
        message_bus: Any,
        logger: Any,
        tenant_id: str,
    ) -> None:
        self.config_manager = config_manager
        self.tool_manager_module = tool_manager_module
        self.persona_manager = persona_manager
        self.message_bus = message_bus
        self.logger = logger
        self.tenant_id = tenant_id

        self.task_manager: TaskManager | None = None
        self.job_manager: JobManager | None = None
        self.job_scheduler: JobScheduler | None = None
        self.job_repository = None
        self.job_service = None
        self._task_queue_service = None

    def set_persona_manager(self, persona_manager: PersonaManager | None) -> None:
        """Update the persona manager reference used by the service."""

        self.persona_manager = persona_manager

    @staticmethod
    def _serialize_tool_health(
        health: Mapping[str, Any], *, include_provider_health: bool = True
    ) -> Dict[str, Any]:
        """Return a JSON-safe view of tool health metrics."""

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

    def _serialize_tool_entry(
        self, view: Any, *, include_provider_health: bool
    ) -> Dict[str, Any]:
        """Return manifest metadata for a tool capability registry view."""

        entry = view.manifest
        return {
            "name": entry.name,
            "persona": entry.persona,
            "description": entry.description,
            "version": entry.version,
            "capabilities": list(entry.capabilities)
            if isinstance(entry.capabilities, list)
            else entry.capabilities,
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
            "health": self._serialize_tool_health(
                view.health, include_provider_health=include_provider_health
            ),
        }

    def _serialize_skill_entry(self, manifest: Any) -> Dict[str, Any]:
        """Return manifest metadata for a skill entry."""

        def _read(field: str, default: Any = None) -> Any:
            if isinstance(manifest, Mapping):
                return manifest.get(field, default)
            return getattr(manifest, field, default)

        sequence_normalizer = lambda candidate: list(
            normalize_sequence(
                candidate,
                allow_strings=True,
                accept_scalar=True,
                copy_items=True,
                coerce_mapping_values=False,
                filter_none=True,
            )
        )

        raw_collaboration = _read("collaboration")
        collaboration_block = (
            copy.deepcopy(raw_collaboration)
            if isinstance(raw_collaboration, Mapping)
            else None
        )

        raw_auth = _read("auth")
        auth_block = copy.deepcopy(raw_auth) if isinstance(raw_auth, Mapping) else None

        payload: Dict[str, Any] = {
            "name": _read("name", ""),
            "persona": _read("persona"),
            "version": _read("version"),
            "instruction_prompt": _read("instruction_prompt"),
            "required_tools": sequence_normalizer(_read("required_tools")),
            "required_capabilities": sequence_normalizer(
                _read("required_capabilities")
            ),
            "safety_notes": _read("safety_notes"),
            "summary": _read("summary"),
            "category": _read("category"),
            "capability_tags": sequence_normalizer(_read("capability_tags")),
            "source": _read("source"),
            "collaboration": collaboration_block,
            "auth": auth_block,
        }

        return payload

    def _refresh_tool_caches(self) -> None:
        """Refresh cached tool metadata after configuration changes."""

        registry = get_capability_registry(config_manager=self.config_manager)
        try:
            registry.refresh(force=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.debug("Failed to refresh capability registry: %s", exc)

        try:
            self.tool_manager_module.load_default_function_map(
                refresh=True,
                config_manager=self.config_manager,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.debug("Failed to refresh tool manager cache: %s", exc)

    def _refresh_skill_caches(self) -> None:
        """Refresh cached skill metadata after configuration changes."""

        registry = get_capability_registry(config_manager=self.config_manager)
        try:
            registry.refresh(force=True)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.debug("Failed to refresh capability registry for skills: %s", exc)

        manager = self.persona_manager
        if manager is not None:
            try:
                if hasattr(manager, "_skill_metadata_cache"):
                    manager._skill_metadata_cache = None
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.debug(
                    "Failed to reset persona skill metadata cache: %s", exc
                )

    def initialize_job_scheduling(self) -> None:
        """Bootstrap job orchestration services and register manifests."""

        if self.job_scheduler is not None:
            return

        if (
            hasattr(self.config_manager, "is_job_scheduling_enabled")
            and not self.config_manager.is_job_scheduling_enabled()
        ):
            manager_setter = getattr(self.config_manager, "set_job_manager", None)
            scheduler_setter = getattr(self.config_manager, "set_job_scheduler", None)
            if callable(manager_setter):
                manager_setter(None)
            if callable(scheduler_setter):
                scheduler_setter(None)
            self.logger.info(
                "Job scheduling disabled by configuration; skipping scheduler setup"
            )
            return

        repository = self.config_manager.get_job_repository()
        queue_service = self.config_manager.get_default_task_queue_service()
        manager_setter = getattr(self.config_manager, "set_job_manager", None)
        scheduler_setter = getattr(self.config_manager, "set_job_scheduler", None)

        if repository is None or queue_service is None:
            if callable(manager_setter):
                manager_setter(None)
            if callable(scheduler_setter):
                scheduler_setter(None)
            self.logger.debug(
                "Job scheduling unavailable (repository=%s, queue=%s)",
                bool(repository),
                bool(queue_service),
            )
            return

        runners = self._build_task_runners()
        task_manager = TaskManager(runners, message_bus=self.message_bus)
        job_manager = JobManager(task_manager, message_bus=self.message_bus)
        scheduler = JobScheduler(
            job_manager, queue_service, repository, tenant_id=self.tenant_id
        )

        if callable(manager_setter):
            manager_setter(job_manager)
        if callable(scheduler_setter):
            scheduler_setter(scheduler)

        executor = scheduler.build_executor()
        try:
            queue_service.set_executor(executor)
        except AttributeError:  # pragma: no cover - compatibility with legacy queue
            queue_service._executor = executor  # type: ignore[attr-defined]

        manifests = load_job_metadata(config_manager=self.config_manager)
        registered = 0
        for metadata in manifests:
            try:
                scheduler.register_manifest(metadata)
            except Exception as exc:  # pragma: no cover - defensive guard during bootstrap
                self.logger.warning(
                    "Failed to register job manifest '%s': %s",
                    metadata.name,
                    exc,
                )
            else:
                registered += 1

        self.logger.info("Registered %s job manifest(s) for scheduling.", registered)

        self.task_manager = task_manager
        self.job_manager = job_manager
        self.job_scheduler = scheduler
        self.job_repository = repository
        self.job_service = self.config_manager.get_job_service()
        self._task_queue_service = queue_service

    def _build_task_runners(self) -> Dict[str, Callable]:
        """Construct step runners backed by the tool function maps."""

        runners: Dict[str, Callable] = {}

        def _register(function_map: Mapping[str, Any] | None) -> None:
            if not isinstance(function_map, Mapping):
                return
            for name, entry in function_map.items():
                try:
                    callable_obj = self.tool_manager_module._resolve_function_callable(
                        entry
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.debug(
                        "Failed to resolve callable for tool '%s': %s", name, exc
                    )
                    continue
                if not callable(callable_obj):
                    continue
                runners[name] = self._wrap_tool_callable(callable_obj)

        try:
            shared_map = self.tool_manager_module.load_default_function_map(
                config_manager=self.config_manager
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.warning("Failed to load shared tool map: %s", exc)
            shared_map = {}

        _register(shared_map)

        manager = self.persona_manager
        persona_names = getattr(manager, "persona_names", []) if manager else []
        for persona_name in persona_names or []:
            persona_payload = {"name": persona_name}
            try:
                persona_map = self.tool_manager_module.load_function_map_from_current_persona(
                    persona_payload,
                    config_manager=self.config_manager,
                )
            except Exception as exc:  # pragma: no cover - persona-specific loaders may fail
                self.logger.debug(
                    "Failed to load tool map for persona '%s': %s", persona_name, exc
                )
                continue
            _register(persona_map)

        return runners

    def _wrap_tool_callable(self, func: Callable[..., Any]) -> Callable:
        """Adapt a tool callable for use as a task step runner."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
            signature = None

        async def _runner(step, context):
            inputs = dict(getattr(step, "inputs", {}) or {})
            call_kwargs = self._prepare_tool_kwargs(signature, inputs, step, context)
            try:
                if call_kwargs:
                    result = func(**call_kwargs)
                else:
                    result = func()
            except TypeError:
                result = func(**inputs)
            if inspect.isawaitable(result):
                result = await result
            return result

        return _runner

    @staticmethod
    def _prepare_tool_kwargs(
        signature: inspect.Signature | None,
        inputs: Dict[str, Any],
        step: Any,
        context: Any,
    ) -> Dict[str, Any]:
        """Return keyword arguments aligned to ``signature`` for a tool call."""

        if signature is None:
            return dict(inputs)

        prepared: Dict[str, Any] = {}
        accepts_kwargs = False

        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue
            if name in inputs:
                prepared[name] = inputs[name]
                continue
            if name in {"context", "task_context"}:
                prepared[name] = context
                continue
            if name in {"step", "plan_step"}:
                prepared[name] = step
                continue
            if name in {"shared_state", "state"}:
                prepared[name] = context.shared_state
                continue
            if name in {"results", "previous_results"}:
                prepared[name] = context.results
                continue
            if name == "inputs":
                prepared[name] = dict(inputs)

        if accepts_kwargs:
            for key, value in inputs.items():
                prepared.setdefault(key, value)

        if not prepared:
            return dict(inputs)

        return prepared

    def list_tools(self, *, include_provider_health: bool = True) -> List[Dict[str, Any]]:
        """Return merged tool metadata with persisted configuration state."""

        registry = get_capability_registry(config_manager=self.config_manager)
        views = registry.query_tools()

        manifest_lookup: Dict[str, Dict[str, Any]] = {}
        serialized_entries: List[Dict[str, Any]] = []

        for view in views:
            payload = self._serialize_tool_entry(
                view, include_provider_health=include_provider_health
            )
            serialized_entries.append(payload)
            manifest_lookup[payload["name"]] = {"auth": payload.get("auth")}

        snapshot = self.config_manager.get_tool_config_snapshot(
            manifest_lookup=manifest_lookup,
        )

        for entry in serialized_entries:
            tool_name = entry["name"]
            config_record = snapshot.get(tool_name, {})
            entry["settings"] = copy.deepcopy(config_record.get("settings", {}))
            entry["credentials"] = copy.deepcopy(
                config_record.get("credentials", {})
            )

        return serialized_entries

    def update_tool_settings(
        self, tool_name: str, settings: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist tool settings and refresh dependent caches."""

        updated = self.config_manager.set_tool_settings(tool_name, settings)
        self._refresh_tool_caches()
        return updated

    def update_tool_credentials(
        self,
        tool_name: str,
        credentials: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Persist tool credentials according to manifest metadata."""

        registry = get_capability_registry(config_manager=self.config_manager)
        manifest_lookup = registry.get_tool_metadata_lookup(
            persona=None,
            names=[tool_name],
        )
        manifest_payload = (
            manifest_lookup.get(tool_name) if isinstance(manifest_lookup, Mapping) else None
        )
        auth_block = None
        if isinstance(manifest_payload, Mapping):
            auth_candidate = manifest_payload.get("auth")
            if isinstance(auth_candidate, Mapping):
                auth_block = auth_candidate

        status = self.config_manager.set_tool_credentials(
            tool_name,
            credentials,
            manifest_auth=auth_block,
        )
        self._refresh_tool_caches()
        return status

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return merged skill metadata with persisted configuration state."""

        entries = load_skill_metadata(config_manager=self.config_manager)

        manifest_lookup: Dict[str, Dict[str, Any]] = {}
        serialized_entries: List[Dict[str, Any]] = []

        for manifest in entries:
            payload = self._serialize_skill_entry(manifest)
            name = payload.get("name")
            if not name:
                continue

            serialized_entries.append(payload)

            if isinstance(manifest, Mapping):
                candidate = manifest.get("auth")
            else:
                candidate = getattr(manifest, "auth", None)
            auth_block: Optional[Mapping[str, Any]] = (
                candidate if isinstance(candidate, Mapping) else None
            )

            manifest_entry = manifest_lookup.setdefault(name, {})
            if auth_block:
                manifest_entry["auth"] = auth_block

        snapshot = self.config_manager.get_skill_config_snapshot(
            manifest_lookup=manifest_lookup if manifest_lookup else None,
            skill_names=[entry.get("name", "") for entry in serialized_entries],
        )

        for entry in serialized_entries:
            name = entry.get("name")
            config_record = snapshot.get(name, {})
            entry["settings"] = copy.deepcopy(config_record.get("settings", {}))
            entry["credentials"] = copy.deepcopy(
                config_record.get("credentials", {})
            )

        return serialized_entries

    def update_skill_settings(
        self, skill_name: str, settings: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist skill settings and refresh dependent caches."""

        updated = self.config_manager.set_skill_settings(skill_name, settings)
        self._refresh_skill_caches()
        return updated

    def update_skill_credentials(
        self,
        skill_name: str,
        credentials: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Persist skill credentials according to manifest metadata."""

        manifest_auth: Optional[Mapping[str, Any]] = None
        for manifest in load_skill_metadata(config_manager=self.config_manager):
            if isinstance(manifest, Mapping):
                name = manifest.get("name")
                candidate = manifest.get("auth")
            else:
                name = getattr(manifest, "name", None)
                candidate = getattr(manifest, "auth", None)
            if name == skill_name and isinstance(candidate, Mapping):
                manifest_auth = candidate
                break

        status = self.config_manager.set_skill_credentials(
            skill_name,
            credentials,
            manifest_auth=manifest_auth,
        )
        self._refresh_skill_caches()
        return status

    def shutdown_jobs(self) -> None:
        """Tear down job scheduling resources if they were initialized."""

        if self._task_queue_service is not None:
            try:
                self._task_queue_service.shutdown(wait=False)
            except Exception:  # pragma: no cover - defensive guard around shutdown
                self.logger.debug(
                    "Task queue shutdown encountered an error", exc_info=True
                )
            self._task_queue_service = None

        self.job_scheduler = None
        self.job_manager = None
        self.task_manager = None
        self.job_repository = None
        self.job_service = None


__all__ = ["ToolingService"]
