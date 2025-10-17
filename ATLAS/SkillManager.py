"""Runtime orchestration utilities for executing skills.

This module coordinates skill execution by resolving tool dependencies,
emitting lifecycle events, and providing a lightweight execution context that
skills can use to share cached state between steps.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from modules.Tools.tool_event_system import event_system
from modules.logging.logger import setup_logger


logger = setup_logger(__name__)

SKILL_ACTIVITY_EVENT = "skill_activity"
_DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0
_DEFAULT_SKILL_RUNTIME_BUDGET_MS = 120_000.0


try:  # Lazy import to avoid expensive persona loading during module import in tests.
    from modules.Personas import load_persona_definition as _PERSONA_DEFINITION_LOADER
except Exception:  # pragma: no cover - defensive import guard
    _PERSONA_DEFINITION_LOADER = None  # type: ignore[assignment]


@dataclass(slots=True)
class SkillExecutionContext:
    """Snapshot of the state passed to skill executions.

    Attributes
    ----------
    conversation_id:
        Identifier for the active conversation.
    conversation_history:
        A history of messages that may be consulted by skills.
    persona:
        The persona configuration driving tool selection.
    user:
        Metadata about the active user.
    state:
        Mutable dictionary that skills can use to cache intermediate results
        between tool invocations.
    metadata:
        Optional additional data supplied by the caller.
    """

    conversation_id: str
    conversation_history: Iterable[Mapping[str, Any]]
    persona: Optional[Mapping[str, Any]] = None
    user: Optional[Mapping[str, Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def persona_identifier(self) -> Optional[str]:
        """Return the canonical identifier for the active persona."""

        persona = self.persona
        if isinstance(persona, str):
            candidate = persona.strip()
            return candidate or None

        if isinstance(persona, Mapping):
            for key in ("id", "identifier", "slug", "name"):
                value = persona.get(key)
                if isinstance(value, str):
                    candidate = value.strip()
                    if candidate:
                        return candidate

        identifier = self.metadata.get("persona_id") if isinstance(self.metadata, Mapping) else None
        if isinstance(identifier, str):
            candidate = identifier.strip()
            return candidate or None

        return None

    def build_dispatch_payload(self) -> Dict[str, Any]:
        """Return a sanitized payload for skill execution dispatch."""

        history_slice = list(self.conversation_history)

        return {
            "conversation_id": self.conversation_id,
            "conversation_history": history_slice,
            "persona_id": self.persona_identifier,
            "state": self.state,
        }


@dataclass(frozen=True)
class SkillRunResult:
    """Container for aggregated results returned by ``use_skill``."""

    skill_name: str
    tool_results: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SkillExecutionError(RuntimeError):
    """Raised when a skill fails to execute successfully."""

    def __init__(
        self,
        message: str,
        *,
        skill_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.skill_name = skill_name
        self.tool_name = tool_name
        self.cause = cause


def _normalize_skill_metadata(skill: Any) -> Mapping[str, Any]:
    if isinstance(skill, Mapping):
        metadata: Dict[str, Any] = dict(skill)
    else:
        metadata = {}
        for field_name in (
            "name",
            "instruction_prompt",
            "required_tools",
            "required_capabilities",
            "safety_notes",
            "version",
        ):
            value = getattr(skill, field_name, None)
            if value is not None:
                metadata[field_name] = value

    metadata.setdefault("name", "")

    metadata["required_tools"] = list(metadata.get("required_tools") or [])
    metadata["required_capabilities"] = list(
        metadata.get("required_capabilities") or []
    )

    for optional_field in (
        "instruction_prompt",
        "safety_notes",
        "version",
    ):
        if optional_field not in metadata:
            metadata[optional_field] = None

    return metadata


def _publish_event(event_type: str, payload: Mapping[str, Any]) -> None:
    event = {"type": event_type, **payload}
    event_system.publish(SKILL_ACTIVITY_EVENT, event)


def _resolve_callable(entry: Any) -> Callable[..., Any]:
    if isinstance(entry, Mapping):
        candidate = entry.get("callable")
        if callable(candidate):
            return candidate  # type: ignore[return-value]
    if callable(entry):
        return entry  # type: ignore[return-value]
    raise TypeError("Tool entry does not contain an executable callable")


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result  # pragma: no branch - single await path
    return result


async def _invoke_tool(
    tool_callable: Callable[..., Any],
    *,
    timeout: Optional[float],
    kwargs: Mapping[str, Any],
) -> Any:
    async def _call() -> Any:
        try:
            result = tool_callable(**dict(kwargs))
        except TypeError:
            # Retry without dict() conversion for callables expecting MappingProxyType
            result = tool_callable(**kwargs)
        return await _maybe_await(result)

    if timeout is None:
        return await _call()
    return await asyncio.wait_for(_call(), timeout)


def _extract_persona_identifier(persona: Any) -> Optional[str]:
    if isinstance(persona, str):
        candidate = persona.strip()
        return candidate or None

    if isinstance(persona, Mapping):
        for key in ("id", "identifier", "slug", "name"):
            value = persona.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate

    return None


def _ensure_persona_payload(persona: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(persona, Mapping) and persona.get("allowed_tools") is not None:
        if "name" not in persona:
            identifier = _extract_persona_identifier(persona)
            if identifier:
                enriched = dict(persona)
                enriched.setdefault("name", identifier)
                return enriched
        return persona

    identifier = _extract_persona_identifier(persona)
    if not identifier:
        return None

    loader = _PERSONA_DEFINITION_LOADER
    if callable(loader):
        try:
            loaded = loader(identifier, config_manager=None)
        except Exception:  # pragma: no cover - defensive guard around persona loading
            logger.warning(
                "Failed to load persona definition for '%s'", identifier, exc_info=True
            )
        else:
            if loaded:
                return copy.deepcopy(loaded)

    return {"name": identifier}


def _resolve_required_tools(
    *,
    persona: Optional[Mapping[str, Any]],
    tool_manager: Any,
) -> Mapping[str, Any]:
    try:
        return tool_manager.load_function_map_from_current_persona(
            persona or {},
            config_manager=None,
        )
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Tool manager does not expose persona resolution API") from exc


def _resolve_required_tool_specs(
    required_tools: Sequence[str],
    *,
    persona: Optional[Mapping[str, Any]],
    tool_manager: Any,
) -> Sequence[Mapping[str, Any]]:
    if not required_tools:
        return []

    loader = getattr(tool_manager, "load_functions_from_json", None)
    if not callable(loader) or not persona:
        return []

    try:
        payload = loader(persona, config_manager=None)
    except TypeError:
        payload = loader(persona)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Failed to resolve tool specifications from manifest.")
        return []

    entries: Iterable[Any]
    if isinstance(payload, Mapping):
        candidates = payload.get("functions") or payload.get("items")
        if isinstance(candidates, Iterable):
            entries = candidates
        else:
            entries = payload.values()
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        entries = payload
    else:
        return []

    lookup: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, Mapping):
            name = entry.get("name")
            if isinstance(name, str) and name:
                lookup[name] = entry

    resolved: list[Mapping[str, Any]] = []
    for tool_name in required_tools:
        spec = lookup.get(tool_name)
        if spec is not None:
            resolved.append(copy.deepcopy(spec))

    return resolved


async def use_skill(
    skill: Any,
    *,
    context: SkillExecutionContext,
    tool_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    tool_manager: Any = None,
    timeout_seconds: Optional[float] = None,
    budget_ms: Optional[float] = None,
) -> SkillRunResult:
    """Execute ``skill`` using the provided ``context``.

    Parameters
    ----------
    skill:
        Skill metadata object or mapping.
    context:
        ``SkillExecutionContext`` describing the conversation snapshot.
    tool_inputs:
        Optional mapping of tool identifiers to keyword argument dictionaries.
    tool_manager:
        Module exposing ``load_function_map_from_current_persona``. When not
        supplied the canonical ``ATLAS.ToolManager`` module is imported lazily.
    timeout_seconds:
        Per-tool execution timeout. Defaults to ``_DEFAULT_TOOL_TIMEOUT_SECONDS``.
    budget_ms:
        Total runtime budget for the skill. Defaults to
        ``_DEFAULT_SKILL_RUNTIME_BUDGET_MS``.
    """

    metadata = _normalize_skill_metadata(skill)
    skill_name = str(metadata.get("name") or "")
    required_tools = list(metadata.get("required_tools") or [])
    instruction_prompt = metadata.get("instruction_prompt")

    if not skill_name:
        raise SkillExecutionError("Skill metadata must include a name")

    tool_inputs = tool_inputs or {}
    timeout_seconds = (
        _DEFAULT_TOOL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    budget_ms = (
        _DEFAULT_SKILL_RUNTIME_BUDGET_MS if budget_ms is None else budget_ms
    )

    if tool_manager is None:
        try:
            import importlib

            tool_manager = importlib.import_module("ATLAS.ToolManager")
        except Exception as exc:  # pragma: no cover - defensive guard
            raise SkillExecutionError(
                "Unable to import tool manager", skill_name=skill_name, cause=exc
            ) from exc

    persona_identifier = context.persona_identifier
    persona_payload = _ensure_persona_payload(
        context.persona if context.persona is not None else persona_identifier
    )

    _publish_event(
        "skill_started",
        {
            "skill": skill_name,
            "conversation_id": context.conversation_id,
            "persona": persona_identifier,
            "user": context.user,
        },
    )

    if instruction_prompt:
        _publish_event(
            "skill_plan_generated",
            {
                "skill": skill_name,
                "prompt": instruction_prompt,
            },
        )

    available_tools = dict(
        _resolve_required_tools(persona=persona_payload, tool_manager=tool_manager) or {}
    )

    metadata["tool_specs"] = list(
        _resolve_required_tool_specs(
            required_tools,
            persona=persona_payload,
            tool_manager=tool_manager,
        )
    )

    missing_tools = [tool for tool in required_tools if tool not in available_tools]
    if missing_tools:
        message = f"Skill '{skill_name}' requires unknown tools: {', '.join(missing_tools)}"
        _publish_event(
            "skill_failed",
            {
                "skill": skill_name,
                "error": message,
                "missing_tools": missing_tools,
            },
        )
        raise SkillExecutionError(message, skill_name=skill_name)

    start_time = time.monotonic()
    results: Dict[str, Any] = {}

    for tool_name in required_tools:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        if budget_ms is not None and elapsed_ms >= budget_ms:
            message = (
                f"Skill '{skill_name}' exceeded runtime budget after "
                f"{elapsed_ms:.0f}ms"
            )
            _publish_event(
                "skill_failed",
                {
                    "skill": skill_name,
                    "error": message,
                    "tool": tool_name,
                },
            )
            raise SkillExecutionError(message, skill_name=skill_name, tool_name=tool_name)

        entry = available_tools[tool_name]
        tool_callable = _resolve_callable(entry)
        tool_kwargs = tool_inputs.get(tool_name, {})

        _publish_event(
            "tool_started",
            {"skill": skill_name, "tool": tool_name, "arguments": tool_kwargs},
        )

        try:
            result = await _invoke_tool(
                tool_callable,
                timeout=timeout_seconds,
                kwargs=tool_kwargs,
            )
        except asyncio.TimeoutError as exc:
            message = f"Tool '{tool_name}' timed out while executing skill '{skill_name}'"
            logger.warning(message)
            _publish_event(
                "tool_failed",
                {
                    "skill": skill_name,
                    "tool": tool_name,
                    "error": message,
                    "timeout_seconds": timeout_seconds,
                },
            )
            _publish_event(
                "skill_failed",
                {
                    "skill": skill_name,
                    "error": message,
                    "tool": tool_name,
                },
            )
            raise SkillExecutionError(
                message, skill_name=skill_name, tool_name=tool_name, cause=exc
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected errors
            message = f"Tool '{tool_name}' failed while executing skill '{skill_name}': {exc}"
            logger.exception(message)
            _publish_event(
                "tool_failed",
                {
                    "skill": skill_name,
                    "tool": tool_name,
                    "error": str(exc),
                },
            )
            _publish_event(
                "skill_failed",
                {
                    "skill": skill_name,
                    "error": str(exc),
                    "tool": tool_name,
                },
            )
            raise SkillExecutionError(
                message, skill_name=skill_name, tool_name=tool_name, cause=exc
            ) from exc

        results[tool_name] = result
        context.state.setdefault("tool_results", {})[tool_name] = result

        _publish_event(
            "tool_completed",
            {"skill": skill_name, "tool": tool_name, "result": result},
        )

    total_elapsed_ms = (time.monotonic() - start_time) * 1000.0
    _publish_event(
        "skill_completed",
        {
            "skill": skill_name,
            "elapsed_ms": total_elapsed_ms,
            "tool_results": results,
        },
    )

    return SkillRunResult(skill_name=skill_name, tool_results=results, metadata=metadata)

