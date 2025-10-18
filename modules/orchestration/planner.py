"""Skill execution planning utilities.

This module converts high level skill instructions or explicit LLM generated
plans into executable dependency graphs.  The resulting plan exposes helper
APIs for orchestrators to reason about execution ordering, gating semantics,
and propagating cancellations when a dependency fails.

The planner intentionally keeps the runtime state lightweight – it only stores
enough information for callers to coordinate tool execution and emit telemetry
about the state of the directed acyclic graph (DAG).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence


class PlanStepStatus(str, Enum):
    """Enumerates the runtime state for a plan step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class PlanStep:
    """Represents a single node in a skill execution plan."""

    identifier: str
    tool_name: str
    dependencies: Sequence[str] = field(default_factory=tuple)
    inputs: Mapping[str, object] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Concrete dependency graph produced by :class:`Planner`."""

    steps: Dict[str, PlanStep]
    dependents: Dict[str, List[str]]

    def __post_init__(self) -> None:
        self._statuses: Dict[str, PlanStepStatus] = {
            step_id: PlanStepStatus.PENDING for step_id in self.steps
        }
        self._cancellation_reasons: Dict[str, str] = {}

    # -- graph helpers -------------------------------------------------
    def topological_order(self) -> List[str]:
        """Return a stable topological ordering of the plan."""

        indegree: Dict[str, int] = {step_id: 0 for step_id in self.steps}
        for step in self.steps.values():
            for dep in step.dependencies:
                indegree[step.identifier] += 1

        queue: deque[str] = deque(
            sorted(step_id for step_id, degree in indegree.items() if degree == 0)
        )
        ordered: List[str] = []
        local_dependents: Dict[str, List[str]] = self.dependents

        while queue:
            current = queue.popleft()
            ordered.append(current)
            for dependant in local_dependents.get(current, []):
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    queue.append(dependant)

        if len(ordered) != len(self.steps):  # pragma: no cover - defensive guard
            raise ValueError("Execution plan contains a cycle")

        return ordered

    # -- status helpers ------------------------------------------------
    def status(self, step_id: str) -> PlanStepStatus:
        return self._statuses[step_id]

    def cancellation_reason(self, step_id: str) -> Optional[str]:
        return self._cancellation_reasons.get(step_id)

    def mark_running(self, step_id: str) -> None:
        self._statuses[step_id] = PlanStepStatus.RUNNING

    def mark_succeeded(self, step_id: str) -> None:
        self._statuses[step_id] = PlanStepStatus.SUCCEEDED

    def mark_failed(self, step_id: str, reason: str) -> List[tuple[str, str]]:
        """Mark ``step_id`` as failed and propagate cancellations.

        Returns a list of ``(cancelled_step, reason)`` entries for steps that
        were cancelled due to the failure.
        """

        self._statuses[step_id] = PlanStepStatus.FAILED
        return self._cancel_descendants(step_id, reason)

    def _cancel_descendants(self, failed_step: str, reason: str) -> List[tuple[str, str]]:
        queue: deque[str] = deque(self.dependents.get(failed_step, []))
        cancelled: List[tuple[str, str]] = []
        while queue:
            candidate = queue.popleft()
            status = self._statuses.get(candidate)
            if status is None:
                continue
            if status is PlanStepStatus.PENDING:
                message = f"Blocked by '{failed_step}' failure: {reason}"
                self._statuses[candidate] = PlanStepStatus.CANCELLED
                self._cancellation_reasons[candidate] = message
                cancelled.append((candidate, message))
                queue.extend(self.dependents.get(candidate, []))
        return cancelled

    def mark_cancelled(self, step_id: str, reason: str) -> None:
        self._statuses[step_id] = PlanStepStatus.CANCELLED
        self._cancellation_reasons[step_id] = reason

    def ready_steps(self) -> List[PlanStep]:
        ready: List[PlanStep] = []
        for step_id in self.topological_order():
            if self._statuses[step_id] is not PlanStepStatus.PENDING:
                continue
            step = self.steps[step_id]
            dependencies = step.dependencies
            if not dependencies:
                ready.append(step)
                continue
            if all(self._statuses[dep] is PlanStepStatus.SUCCEEDED for dep in dependencies):
                ready.append(step)
        return ready

    def unfinished(self) -> bool:
        return any(
            status in (PlanStepStatus.PENDING, PlanStepStatus.RUNNING)
            for status in self._statuses.values()
        )

    def snapshot(self) -> Dict[str, object]:
        nodes = []
        for step in self.steps.values():
            nodes.append(
                {
                    "id": step.identifier,
                    "tool": step.tool_name,
                    "status": self._statuses[step.identifier].value,
                    "dependencies": list(step.dependencies),
                    "cancellation_reason": self._cancellation_reasons.get(step.identifier),
                }
            )

        edges = []
        for step in self.steps.values():
            for dependency in step.dependencies:
                edges.append({"from": dependency, "to": step.identifier})

        return {"nodes": nodes, "edges": edges}


class Planner:
    """Builds :class:`ExecutionPlan` instances for skills."""

    def build_plan(
        self,
        metadata: Mapping[str, object],
        *,
        available_tools: Mapping[str, object],
        provided_inputs: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> ExecutionPlan:
        provided_inputs = provided_inputs or {}

        raw_required = metadata.get("required_tools")
        required_tools: List[str] = []
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray)):
            required_tools = [str(name) for name in raw_required]

        raw_plan = metadata.get("plan") or metadata.get("execution_plan")
        step_entries: List[Mapping[str, object]] = []
        if isinstance(raw_plan, Mapping):
            candidates = raw_plan.get("steps")
            if isinstance(candidates, Sequence):
                step_entries = [entry for entry in candidates if isinstance(entry, Mapping)]
        elif isinstance(raw_plan, Sequence) and not isinstance(raw_plan, (str, bytes, bytearray)):
            step_entries = [entry for entry in raw_plan if isinstance(entry, Mapping)]

        steps: Dict[str, PlanStep] = {}
        tools_to_steps: Dict[str, str] = {}

        for entry in step_entries:
            tool_name = entry.get("tool") or entry.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            step_id = entry.get("id")
            if not isinstance(step_id, str) or not step_id:
                step_id = tool_name

            dependencies: Sequence[str]
            raw_dependencies = entry.get("after") or entry.get("depends_on") or []
            if isinstance(raw_dependencies, Sequence) and not isinstance(
                raw_dependencies, (str, bytes, bytearray)
            ):
                dependencies = [str(dep) for dep in raw_dependencies]
            elif isinstance(raw_dependencies, str) and raw_dependencies:
                dependencies = [raw_dependencies]
            else:
                dependencies = []

            inputs: Mapping[str, object]
            raw_inputs = entry.get("inputs")
            if isinstance(raw_inputs, Mapping):
                inputs = dict(raw_inputs)
            else:
                inputs = {}

            steps[step_id] = PlanStep(
                identifier=step_id,
                tool_name=tool_name,
                dependencies=tuple(dependencies),
                inputs=inputs,
            )
            tools_to_steps[tool_name] = step_id

        # Ensure every required tool has a corresponding step. Tools absent from
        # the plan are appended sequentially.
        last_step_id: Optional[str] = None
        if steps:
            ordered = self._topological_names(steps)
            last_step_id = ordered[-1] if ordered else None

        for tool_name in required_tools:
            if tool_name in tools_to_steps:
                last_step_id = tools_to_steps[tool_name]
                continue
            step_id = tool_name
            dependencies = (last_step_id,) if last_step_id else ()
            steps[step_id] = PlanStep(
                identifier=step_id,
                tool_name=tool_name,
                dependencies=dependencies,
                inputs=dict(provided_inputs.get(tool_name, {})),
            )
            tools_to_steps[tool_name] = step_id
            last_step_id = step_id

        # Merge tool inputs for steps already provided in the plan.
        for tool_name, step_id in tools_to_steps.items():
            if tool_name not in available_tools:
                continue
            if step_id not in steps:
                continue
            existing_inputs = dict(steps[step_id].inputs)
            provided = provided_inputs.get(tool_name)
            if isinstance(provided, Mapping):
                combined = dict(provided)
                combined.update(existing_inputs)
                steps[step_id] = PlanStep(
                    identifier=steps[step_id].identifier,
                    tool_name=steps[step_id].tool_name,
                    dependencies=steps[step_id].dependencies,
                    inputs=combined,
                )

        dependents: Dict[str, List[str]] = {step_id: [] for step_id in steps}
        normalized_steps: Dict[str, PlanStep] = {}
        for step in steps.values():
            normalized_dependencies: List[str] = []
            for dependency in step.dependencies:
                resolved_dependency = dependency
                if dependency not in steps and dependency in tools_to_steps:
                    resolved_dependency = tools_to_steps[dependency]
                if resolved_dependency not in steps:
                    raise ValueError(
                        f"Execution plan references unknown dependency '{dependency}'"
                    )
                normalized_dependencies.append(resolved_dependency)
                dependents.setdefault(resolved_dependency, []).append(step.identifier)
            normalized_steps[step.identifier] = PlanStep(
                identifier=step.identifier,
                tool_name=step.tool_name,
                dependencies=tuple(normalized_dependencies),
                inputs=step.inputs,
            )

        steps = normalized_steps

        # Validate for cycles.
        self._topological_names(steps)

        return ExecutionPlan(steps=steps, dependents=dependents)

    def _topological_names(self, steps: Mapping[str, PlanStep]) -> List[str]:
        indegree: MutableMapping[str, int] = {step_id: 0 for step_id in steps}
        adjacency: Dict[str, List[str]] = {step_id: [] for step_id in steps}
        for step in steps.values():
            for dependency in step.dependencies:
                if dependency not in steps:
                    continue
                indegree[step.identifier] += 1
                adjacency.setdefault(dependency, []).append(step.identifier)

        queue: deque[str] = deque(
            sorted(step_id for step_id, degree in indegree.items() if degree == 0)
        )
        ordered: List[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for dependant in adjacency.get(current, []):
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    queue.append(dependant)

        if len(ordered) != len(steps):
            raise ValueError("Execution plan contains a cycle")
        return ordered

