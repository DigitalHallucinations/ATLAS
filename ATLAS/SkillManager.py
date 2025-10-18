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

from modules.Tools.tool_event_system import publish_bus_event
from modules.orchestration.blackboard import BlackboardClient, get_blackboard
from modules.orchestration.message_bus import MessagePriority, get_message_bus
from modules.orchestration.planner import Planner, PlanStep, PlanStepStatus
from modules.logging.logger import setup_logger


logger = setup_logger(__name__)

# Ensure the message bus is initialized outside of any running event loop to avoid
# nested ``asyncio.run`` calls during skill execution in tests.
get_message_bus()

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
    blackboard_client: Optional[BlackboardClient] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        project_id: Optional[str] = None
        if isinstance(self.metadata, Mapping):
            raw_project = self.metadata.get("project_id")
            if isinstance(raw_project, str) and raw_project.strip():
                project_id = raw_project.strip()

        if self.blackboard_client is None:
            scope_type = "project" if project_id else "conversation"
            scope_id = project_id or self.conversation_id
            self.blackboard_client = get_blackboard().client_for(
                scope_id,
                scope_type=scope_type,
            )

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

    @property
    def blackboard(self) -> BlackboardClient:
        """Convenience accessor returning the scoped blackboard client."""

        client = self.blackboard_client
        if client is None:
            scope_type = "conversation"
            scope_id = self.conversation_id
            if isinstance(self.metadata, Mapping):
                project_id = self.metadata.get("project_id")
                if isinstance(project_id, str) and project_id.strip():
                    scope_type = "project"
                    scope_id = project_id.strip()
            client = get_blackboard().client_for(scope_id, scope_type=scope_type)
            self.blackboard_client = client
        return client


@dataclass(frozen=True)
class SkillRunResult:
    """Container for aggregated results returned by ``use_skill``."""

    skill_name: str
    tool_results: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    required_capabilities: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # ``required_capabilities`` may be provided as any iterable. Normalize to a
        # tuple of non-empty strings to provide a stable consumer contract.
        normalized: list[str] = []
        for capability in self.required_capabilities:
            if capability is None:
                continue
            token = str(capability).strip()
            if token:
                normalized.append(token)
        object.__setattr__(self, "required_capabilities", tuple(normalized))

    @property
    def capability_tags(self) -> tuple[str, ...]:
        """Alias for compatibility with callers expecting capability metadata."""

        return self.required_capabilities


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


def _derive_correlation_id(event: Mapping[str, Any]) -> str:
    skill = event.get("skill")
    conversation_id = event.get("conversation_id")
    if conversation_id and skill:
        return f"{conversation_id}:{skill}"
    if skill:
        return str(skill)
    return event.get("type", "skill_event")


def _publish_event(
    event_type: str,
    payload: Mapping[str, Any],
    *,
    correlation_id: Optional[str] = None,
    tracing: Optional[Mapping[str, Any]] = None,
) -> None:
    event = {"type": event_type, **payload}
    base_trace = {
        "event_type": event_type,
        "skill": event.get("skill"),
        "conversation_id": event.get("conversation_id"),
        "persona": event.get("persona"),
    }
    trace_payload = dict(base_trace)
    if tracing:
        trace_payload.update({k: v for k, v in tracing.items() if v is not None})
    sanitized_trace = {k: v for k, v in trace_payload.items() if v is not None}
    publish_bus_event(
        SKILL_ACTIVITY_EVENT,
        dict(event),
        priority=MessagePriority.NORMAL,
        correlation_id=correlation_id or _derive_correlation_id(event),
        tracing=sanitized_trace or None,
        metadata={"component": "SkillManager"},
    )


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
    raw_capabilities = metadata.get("required_capabilities") or []
    capability_tags: list[str] = []
    for capability in raw_capabilities:
        if capability is None:
            continue
        token = str(capability).strip()
        if token:
            capability_tags.append(token)
    skill_version = metadata.get("version")
    if isinstance(skill_version, str):
        skill_version = skill_version.strip() or None
    elif skill_version is not None:
        skill_version = str(skill_version).strip() or None

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
            "skill_version": skill_version,
            "capability_tags": capability_tags,
        },
    )

    if instruction_prompt:
        _publish_event(
        "skill_plan_generated",
        {
            "skill": skill_name,
            "prompt": instruction_prompt,
            "skill_version": skill_version,
            "capability_tags": capability_tags,
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

    planner = Planner()
    plan = planner.build_plan(
        metadata,
        available_tools=available_tools,
        provided_inputs=tool_inputs,
    )

    start_time = time.monotonic()
    results: Dict[str, Any] = {}

    def _remaining_budget_ms() -> Optional[float]:
        if budget_ms is None:
            return None
        elapsed = (time.monotonic() - start_time) * 1000.0
        return max(budget_ms - elapsed, 0.0)

    def _emit_plan_event(
        event_type: str,
        *,
        step_id: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "skill": skill_name,
            "plan": plan.snapshot(),
            "remaining_budget_ms": _remaining_budget_ms(),
        }
        if step_id is not None:
            step = plan.steps[step_id]
            payload.update({"step": step_id, "tool": step.tool_name})
            reason = plan.cancellation_reason(step_id)
            if reason:
                payload["cancellation_reason"] = reason
        if extra:
            payload.update({k: v for k, v in extra.items() if v is not None})
        _publish_event(event_type, payload)

    _emit_plan_event("skill_plan_ready")

    running: Dict[str, asyncio.Task] = {}
    task_lookup: Dict[asyncio.Task, str] = {}
    pending_failure: Optional[SkillExecutionError] = None

    async def _run_step(step: PlanStep) -> Any:
        tool_name = step.tool_name
        entry = available_tools[tool_name]
        tool_callable = _resolve_callable(entry)
        base_inputs = tool_inputs.get(tool_name, {})
        kwargs: Dict[str, Any] = dict(base_inputs)
        if step.inputs:
            kwargs.update(step.inputs)

        _publish_event(
            "tool_started",
            {
                "skill": skill_name,
                "tool": tool_name,
                "arguments": kwargs,
                "remaining_budget_ms": _remaining_budget_ms(),
            },
        )

        try:
            result = await _invoke_tool(
                tool_callable,
                timeout=timeout_seconds,
                kwargs=kwargs,
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
                    "remaining_budget_ms": _remaining_budget_ms(),
                },
            )
            _publish_event(
                "skill_failed",
                {
                    "skill": skill_name,
                    "error": message,
                    "tool": tool_name,
                    "remaining_budget_ms": _remaining_budget_ms(),
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
                    "remaining_budget_ms": _remaining_budget_ms(),
                },
            )
            _publish_event(
                "skill_failed",
                {
                    "skill": skill_name,
                    "error": str(exc),
                    "tool": tool_name,
                    "remaining_budget_ms": _remaining_budget_ms(),
                },
            )
            raise SkillExecutionError(
                message, skill_name=skill_name, tool_name=tool_name, cause=exc
            ) from exc

        _publish_event(
            "tool_completed",
            {
                "skill": skill_name,
                "tool": tool_name,
                "result": result,
                "remaining_budget_ms": _remaining_budget_ms(),
            },
        )

        return result

    while plan.unfinished() or running:
        ready_steps = [
            step
            for step in plan.ready_steps()
            if step.identifier not in running and plan.status(step.identifier) is PlanStepStatus.PENDING
        ]

        if ready_steps:
            remaining = _remaining_budget_ms()
            if remaining is not None and remaining <= 0:
                message = (
                    f"Skill '{skill_name}' exceeded runtime budget after "
                    f"{(time.monotonic() - start_time) * 1000.0:.0f}ms"
                )
                _publish_event(
                    "skill_failed",
                    {
                        "skill": skill_name,
                        "error": message,
                        "remaining_budget_ms": remaining,
                    },
                )
                for step_identifier in plan.steps:
                    if plan.status(step_identifier) is PlanStepStatus.PENDING:
                        plan.mark_cancelled(step_identifier, "Skill budget exceeded")
                        _emit_plan_event(
                            "skill_plan_step_cancelled",
                            step_id=step_identifier,
                            extra={"cancellation_reason": "Skill budget exceeded"},
                        )
                pending_failure = SkillExecutionError(message, skill_name=skill_name)
                break

        for step in ready_steps:
            plan.mark_running(step.identifier)
            _emit_plan_event("skill_plan_step_started", step_id=step.identifier)
            task = asyncio.create_task(_run_step(step))
            running[step.identifier] = task
            task_lookup[task] = step.identifier

        if not running:
            break

        done, _pending = await asyncio.wait(
            list(running.values()), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            step_id = task_lookup.pop(task)
            running.pop(step_id, None)
            try:
                result = await task
            except asyncio.CancelledError:
                if plan.status(step_id) is not PlanStepStatus.CANCELLED:
                    plan.mark_cancelled(step_id, "Cancelled due to upstream failure")
                _emit_plan_event(
                    "skill_plan_step_cancelled",
                    step_id=step_id,
                )
                continue
            except SkillExecutionError as exc:
                cancelled = plan.mark_failed(step_id, str(exc))
                _emit_plan_event(
                    "skill_plan_step_failed",
                    step_id=step_id,
                    extra={"error": str(exc)},
                )
                for cancelled_step, reason in cancelled:
                    _emit_plan_event(
                        "skill_plan_step_cancelled",
                        step_id=cancelled_step,
                        extra={"cancellation_reason": reason},
                    )
                for other_task in list(running.values()):
                    other_task.cancel()
                pending_failure = exc
                break
            else:
                plan.mark_succeeded(step_id)
                step = plan.steps[step_id]
                results[step.tool_name] = result
                context.state.setdefault("tool_results", {})[step.tool_name] = result
                _emit_plan_event(
                    "skill_plan_step_succeeded",
                    step_id=step_id,
                    extra={"result": result},
                )

        if pending_failure is not None:
            break

    if running:
        await asyncio.gather(*running.values(), return_exceptions=True)

    if pending_failure is not None:
        raise pending_failure

    total_elapsed_ms = (time.monotonic() - start_time) * 1000.0
    _publish_event(
        "skill_completed",
        {
            "skill": skill_name,
            "elapsed_ms": total_elapsed_ms,
            "tool_results": results,
            "skill_version": skill_version,
            "capability_tags": capability_tags,
        },
    )

    return SkillRunResult(
        skill_name=skill_name,
        tool_results=results,
        metadata=metadata,
        version=skill_version,
        required_capabilities=tuple(capability_tags),
    )

