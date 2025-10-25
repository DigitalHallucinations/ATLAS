from __future__ import annotations

"""Utilities for producing execution plan snapshots without running skills."""

from typing import Any, Dict, Mapping, Sequence

from modules.orchestration.planner import Planner

__all__ = ["planner_decompose"]


def planner_decompose(
    *,
    skill_metadata: Mapping[str, Any] | None = None,
    available_tools: Mapping[str, Any] | Sequence[Any] | None = None,
    provided_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    persona_overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a serialized execution plan for the supplied skill metadata.

    Parameters
    ----------
    skill_metadata:
        Canonical skill manifest entry describing required tools and the
        declarative plan. When omitted an empty mapping is assumed.
    available_tools:
        Collection describing the tools accessible to the planner. Accepts a
        mapping of tool names to metadata objects or an iterable of either tool
        names or metadata dictionaries containing a ``"name"`` attribute. When
        not provided the tool registry is derived from the skill metadata's
        ``required_tools`` field.
    provided_inputs:
        Optional mapping of tool identifiers to keyword arguments that should be
        injected into the generated plan.
    persona_overrides:
        Optional shallow overrides applied to ``skill_metadata``. When supplied
        the overrides may either be the raw keys to merge or a mapping
        containing a ``"metadata"`` key.

    Returns
    -------
    Mapping[str, Any]
        Serializable payload containing plan nodes, edges, statuses, and the
        ordered step definitions.
    """

    metadata = _normalize_skill_metadata(skill_metadata)
    overrides = _normalize_persona_overrides(persona_overrides)
    if overrides:
        combined = dict(metadata)
        combined.update(overrides)
        metadata = combined

    tool_registry = _normalize_available_tools(available_tools, metadata)
    tool_inputs = _normalize_tool_inputs(provided_inputs)

    planner = Planner()
    plan = planner.build_plan(
        metadata,
        available_tools=tool_registry,
        provided_inputs=tool_inputs,
    )

    snapshot = plan.snapshot()
    statuses = {step_id: plan.status(step_id).value for step_id in plan.steps}

    ordered_steps = []
    for step_id in plan.topological_order():
        step = plan.steps[step_id]
        ordered_steps.append(
            {
                "id": step.identifier,
                "tool": step.tool_name,
                "dependencies": list(step.dependencies),
                "inputs": dict(step.inputs),
                "status": statuses[step_id],
            }
        )

    return {
        "nodes": snapshot["nodes"],
        "edges": snapshot["edges"],
        "statuses": statuses,
        "steps": ordered_steps,
    }


def _normalize_skill_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _normalize_persona_overrides(overrides: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(overrides, Mapping):
        return {}

    if "metadata" in overrides and isinstance(overrides["metadata"], Mapping):
        return dict(overrides["metadata"])
    if "skill_metadata" in overrides and isinstance(
        overrides["skill_metadata"], Mapping
    ):
        return dict(overrides["skill_metadata"])
    return dict(overrides)


def _normalize_tool_inputs(
    inputs: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, Mapping[str, Any]]:
    normalized: Dict[str, Mapping[str, Any]] = {}
    if not isinstance(inputs, Mapping):
        return normalized

    for name, entry in inputs.items():
        key = _coerce_name(name)
        if not key:
            continue
        if isinstance(entry, Mapping):
            normalized[key] = dict(entry)
    return normalized


def _normalize_available_tools(
    registry: Mapping[str, Any] | Sequence[Any] | None,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    if isinstance(registry, Mapping):
        for name, entry in registry.items():
            key = _coerce_name(_extract_tool_name(entry) or name)
            if not key:
                continue
            normalized[key] = entry
    elif isinstance(registry, Sequence) and not isinstance(
        registry, (str, bytes, bytearray)
    ):
        for entry in registry:
            if isinstance(entry, Mapping):
                name = _coerce_name(_extract_tool_name(entry))
                if not name:
                    continue
                normalized[name] = entry
            else:
                name = _coerce_name(entry)
                if not name:
                    continue
                normalized[name] = {}

    if not normalized:
        required = metadata.get("required_tools")
        if isinstance(required, Sequence) and not isinstance(
            required, (str, bytes, bytearray)
        ):
            for name in required:
                key = _coerce_name(name)
                if key and key not in normalized:
                    normalized[key] = {}

    return normalized


def _extract_tool_name(entry: Any) -> str | None:
    if isinstance(entry, Mapping):
        candidate = entry.get("name")
        if isinstance(candidate, str):
            return candidate
    return None


def _coerce_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
