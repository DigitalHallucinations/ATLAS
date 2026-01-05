"""Utility wrapper exposing capability registry summaries and lookups."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from core.utils import normalize_sequence
from modules.orchestration.capability_registry import get_capability_registry

_DEFAULT_CAPABILITY_TYPES = ("summary",)
_CAPABILITY_TYPE_ALIASES = {
    "summary": "summary",
    "summaries": "summary",
    "tool": "tools",
    "tools": "tools",
    "skill": "skills",
    "skills": "skills",
    "task": "tasks",
    "tasks": "tasks",
    "job": "jobs",
    "jobs": "jobs",
    "all": "all",
}


def registry_capability(
    *,
    persona: Optional[str] = None,
    persona_filters: Optional[Sequence[str]] = None,
    capability_types: Optional[Sequence[str]] = None,
    capability_filters: Optional[Sequence[str]] = None,
    provider_filters: Optional[Sequence[str]] = None,
    version_constraint: Optional[str] = None,
    min_success_rate: Optional[float] = None,
    task_required_skills: Optional[Sequence[str]] = None,
    task_required_tools: Optional[Sequence[str]] = None,
    task_tag_filters: Optional[Sequence[str]] = None,
    job_required_capabilities: Optional[Sequence[str]] = None,
    job_required_skills: Optional[Sequence[str]] = None,
    refresh: bool = False,
    force_refresh: bool = False,
    include_summary: Optional[bool] = None,
    config_manager: Optional[Any] = None,
) -> Mapping[str, Any]:
    """Return capability registry insights for orchestration tooling."""

    registry = get_capability_registry(config_manager=config_manager)
    refreshed = False
    if force_refresh:
        refreshed = bool(registry.refresh(force=True))
    elif refresh:
        refreshed = bool(registry.refresh(force=False))

    persona_tokens = _normalize_filter_tokens(persona_filters)
    if persona_tokens is None and persona is not None:
        persona_tokens = [persona]
    elif persona is not None and persona not in (persona_tokens or []):
        persona_tokens = (persona_tokens or []) + [persona]

    capability_tokens = _normalize_filter_tokens(capability_filters)
    provider_tokens = _normalize_filter_tokens(provider_filters)
    task_skill_tokens = _normalize_filter_tokens(task_required_skills)
    task_tool_tokens = _normalize_filter_tokens(task_required_tools)
    task_tag_tokens = _normalize_filter_tokens(task_tag_filters)
    job_capability_tokens = _normalize_filter_tokens(job_required_capabilities)
    job_skill_tokens = _normalize_filter_tokens(job_required_skills)

    requested_types = _normalize_capability_types(capability_types)
    if include_summary is False:
        requested_types = tuple(
            entry for entry in requested_types if entry != "summary"
        )
    elif include_summary is True and "summary" not in requested_types:
        requested_types = requested_types + ("summary",)

    response: MutableMapping[str, Any] = {
        "revision": registry.revision,
        "persona": persona,
        "requested": list(requested_types),
        "filters": {
            "persona": list(persona_tokens or []),
            "capability": list(capability_tokens or []),
            "provider": list(provider_tokens or []),
            "task_required_skills": list(task_skill_tokens or []),
            "task_required_tools": list(task_tool_tokens or []),
            "task_tags": list(task_tag_tokens or []),
            "job_required_capabilities": list(job_capability_tokens or []),
            "job_required_skills": list(job_skill_tokens or []),
            "version_constraint": version_constraint,
            "min_success_rate": min_success_rate,
        },
        "refreshed": refreshed,
        "force_refreshed": force_refresh,
    }

    if "summary" in requested_types:
        response["summary"] = _to_plain(
            registry.summary(persona=persona)
        )

    if "tools" in requested_types or "all" in requested_types:
        tool_views = registry.query_tools(
            persona_filters=persona_tokens,
            capability_filters=capability_tokens,
            provider_filters=provider_tokens,
            version_constraint=version_constraint,
            min_success_rate=min_success_rate,
        )
        response["tools"] = [
            {
                "manifest": _to_plain(view.manifest),
                "capability_tags": list(view.capability_tags),
                "auth_scopes": list(view.auth_scopes),
                "health": _to_plain(view.health),
            }
            for view in tool_views
        ]

    if "skills" in requested_types or "all" in requested_types:
        skill_views = registry.query_skills(
            persona_filters=persona_tokens,
            capability_filters=capability_tokens,
            version_constraint=version_constraint,
        )
        response["skills"] = [
            {
                "manifest": _to_plain(view.manifest),
                "capability_tags": list(view.capability_tags),
                "required_capabilities": list(view.required_capabilities),
            }
            for view in skill_views
        ]

    if "tasks" in requested_types or "all" in requested_types:
        task_views = registry.query_tasks(
            persona_filters=persona_tokens,
            required_skill_filters=task_skill_tokens,
            required_tool_filters=task_tool_tokens,
            tag_filters=task_tag_tokens,
        )
        response["tasks"] = [
            {
                "manifest": _to_plain(view.manifest),
                "required_skills": list(view.required_skills),
                "required_tools": list(view.required_tools),
                "tags": list(view.tags),
            }
            for view in task_views
        ]

    if "jobs" in requested_types or "all" in requested_types:
        job_views = registry.query_jobs(
            persona_filters=persona_tokens,
            required_capability_filters=job_capability_tokens,
            required_skill_filters=job_skill_tokens,
        )
        response["jobs"] = [
            {
                "manifest": _to_plain(view.manifest),
                "personas": list(view.personas),
                "required_skills": list(view.required_skills),
                "required_tools": list(view.required_tools),
                "required_capabilities": list(view.required_capabilities),
                "health": _to_plain(view.health),
            }
            for view in job_views
        ]

    return response

def _normalize_filter_tokens(values: Optional[Sequence[Any]]) -> Optional[list[str]]:
    tokens = normalize_sequence(
        values,
        transform=lambda item: str(item).strip(),
        filter_falsy=True,
        accept_scalar=False,
    )
    return list(tokens) or None


def _normalize_capability_types(
    capability_types: Optional[Sequence[Any]],
) -> tuple[str, ...]:
    if not capability_types:
        return _DEFAULT_CAPABILITY_TYPES

    resolved: list[str] = []
    for entry in capability_types:
        if entry is None:
            continue
        token = str(entry).strip().lower()
        if not token:
            continue
        alias = _CAPABILITY_TYPE_ALIASES.get(token)
        if alias is None:
            continue
        if alias == "all":
            for value in ("summary", "tools", "skills", "tasks", "jobs"):
                if value not in resolved:
                    resolved.append(value)
        elif alias not in resolved:
            resolved.append(alias)
    return tuple(resolved or _DEFAULT_CAPABILITY_TYPES)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "_asdict") and callable(getattr(value, "_asdict")):
        return _to_plain(value._asdict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: _to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


__all__ = ["registry_capability"]
