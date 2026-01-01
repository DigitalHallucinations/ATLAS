"""Asynchronous task orchestration utilities.

This module exposes a small task orchestrator that consumes task manifests,
converts them into :class:`~modules.orchestration.planner.ExecutionPlan`
instances using :class:`~modules.orchestration.planner.Planner` and then
executes each :class:`~modules.orchestration.planner.PlanStep` with retry and
failure propagation semantics.  State changes are published on the
``tasks.*`` topics of the shared :class:`~modules.orchestration.message_bus`
and, when available, important milestones are surfaced on the shared
blackboard so that collaborators receive timely updates.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from modules.orchestration.blackboard import BlackboardClient
from modules.orchestration.planner import ExecutionPlan, PlanStep, PlanStepStatus, Planner

from ATLAS.messaging import (
    AgentBus,
    AgentMessage,
    MessagePriority,
    get_agent_bus,
    TASK_CREATE,
    TASK_UPDATE,
    TASK_COMPLETE,
)

_LOGGER = logging.getLogger(__name__)

# Channel names for task events
TASK_CREATED_TOPIC = TASK_CREATE.name
TASK_UPDATED_TOPIC = TASK_UPDATE.name
TASK_COMPLETED_TOPIC = TASK_COMPLETE.name

StepRunner = Callable[[PlanStep, "TaskStepContext"], Awaitable[Any] | Any]


@dataclass(slots=True)
class TaskStepContext:
    """Context object passed to step runners.

    The context exposes the task identifier, manifest metadata and a shared
    state dictionary that runners may use to persist information between
    steps.  ``results`` provides a read-only mapping of completed step outputs
    which can be inspected when preparing inputs for the current step.
    """

    task_id: str
    manifest: Mapping[str, Any]
    step_id: str
    shared_state: MutableMapping[str, Any]
    results: Mapping[str, Any]


@dataclass
class _TaskRuntimeState:
    """Internal bookkeeping model tracking task execution state."""

    task_id: str
    manifest: Mapping[str, Any]
    plan: ExecutionPlan
    shared_state: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"

    def snapshot(self, *, current_step: Optional[str] = None) -> Dict[str, Any]:
        """Return a serialisable snapshot of the current state."""

        return {
            "task_id": self.task_id,
            "status": self.status,
            "current_step": current_step,
            "results": dict(self.results),
            "errors": dict(self.errors),
            "plan": self.plan.snapshot(),
        }

    def build_step_context(self, step: PlanStep) -> TaskStepContext:
        """Create the :class:`TaskStepContext` passed to runners."""

        return TaskStepContext(
            task_id=self.task_id,
            manifest=self.manifest,
            step_id=step.identifier,
            shared_state=self.shared_state,
            results=MappingProxyType(self.results),
        )


@dataclass(frozen=True)
class TaskRunResult:
    """Aggregated results returned after a task finishes executing."""

    task_id: str
    status: str
    results: Mapping[str, Any]
    errors: Mapping[str, str]
    plan: ExecutionPlan
    snapshot: Mapping[str, Any]


class TaskManager:
    """Execute task manifests by coordinating registered tool runners."""

    def __init__(
        self,
        tool_runners: Mapping[str, StepRunner],
        *,
        planner: Optional[Planner] = None,
        agent_bus: Optional[AgentBus] = None,
        max_attempts: int = 1,
        retry_delay: float = 0.1,
    ) -> None:
        self._tool_runners: Dict[str, StepRunner] = dict(tool_runners)
        self._planner = planner or Planner()
        self._bus = agent_bus or get_agent_bus()
        self._max_attempts = max(1, int(max_attempts))
        self._retry_delay = max(0.0, float(retry_delay))

    async def run_task(
        self,
        manifest: Mapping[str, Any],
        *,
        provided_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        blackboard_client: Optional[BlackboardClient] = None,
    ) -> TaskRunResult:
        """Execute the supplied *manifest* and return the aggregated result."""

        if not isinstance(manifest, Mapping):
            raise TypeError("Task manifests must be mapping objects")

        task_id = manifest.get("id")
        if not isinstance(task_id, str) or not task_id:
            task_id = uuid.uuid4().hex

        plan = self._planner.build_plan(
            manifest,
            available_tools=self._tool_runners,
            provided_inputs=provided_inputs,
        )

        state = _TaskRuntimeState(task_id=task_id, manifest=dict(manifest), plan=plan)

        await self._publish(TASK_CREATED_TOPIC, state.snapshot())
        self._publish_acceptance_criteria(state, manifest, blackboard_client)

        failed = False
        while plan.unfinished():
            ready_steps = plan.ready_steps()
            if not ready_steps:
                break

            for step in ready_steps:
                if plan.status(step.identifier) is not PlanStepStatus.PENDING:
                    continue

                success = await self._execute_step(state, step, blackboard_client)
                if not success:
                    failed = True
                    break

            if failed:
                break

        final_status = self._determine_final_status(plan)
        state.status = final_status
        final_snapshot = state.snapshot()
        await self._publish(TASK_COMPLETED_TOPIC, final_snapshot)

        return TaskRunResult(
            task_id=state.task_id,
            status=final_status,
            results=dict(state.results),
            errors=dict(state.errors),
            plan=plan,
            snapshot=final_snapshot,
        )

    async def _execute_step(
        self,
        state: _TaskRuntimeState,
        step: PlanStep,
        blackboard_client: Optional[BlackboardClient],
    ) -> bool:
        runner = self._tool_runners.get(step.tool_name)
        if runner is None:
            reason = f"No runner registered for tool '{step.tool_name}'"
            _LOGGER.error("%s", reason)
            state.errors[step.identifier] = reason
            state.status = "failed"
            cancellations = state.plan.mark_failed(step.identifier, reason)
            for cancelled_step, message in cancellations:
                state.errors.setdefault(cancelled_step, message)
            await self._publish(TASK_UPDATED_TOPIC, state.snapshot(current_step=step.identifier))
            self._publish_blocker(state, step.identifier, reason, cancellations, blackboard_client)
            return False

        state.plan.mark_running(step.identifier)
        state.status = "running"
        await self._publish(TASK_UPDATED_TOPIC, state.snapshot(current_step=step.identifier))

        attempt = 0
        while True:
            attempt += 1
            try:
                context = state.build_step_context(step)
                result = runner(step, context)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    result = await result  # type: ignore[assignment]
                state.results[step.identifier] = result
                state.plan.mark_succeeded(step.identifier)
                await self._publish(TASK_UPDATED_TOPIC, state.snapshot(current_step=step.identifier))
                self._publish_progress(state, step, blackboard_client)
                return True
            except Exception as exc:  # pylint: disable=broad-except
                _LOGGER.exception(
                    "Step '%s' for task '%s' failed (attempt %d/%d)",
                    step.identifier,
                    state.task_id,
                    attempt,
                    self._max_attempts,
                )
                if attempt >= self._max_attempts:
                    reason = str(exc) or exc.__class__.__name__
                    state.errors[step.identifier] = reason
                    cancellations = state.plan.mark_failed(step.identifier, reason)
                    for cancelled_step, message in cancellations:
                        state.errors.setdefault(cancelled_step, message)
                    await self._publish(TASK_UPDATED_TOPIC, state.snapshot(current_step=step.identifier))
                    self._publish_blocker(state, step.identifier, reason, cancellations, blackboard_client)
                    return False
                await asyncio.sleep(self._retry_delay)

    @staticmethod
    def _determine_final_status(plan: ExecutionPlan) -> str:
        statuses = [plan.status(step_id) for step_id in plan.steps]
        if all(status is PlanStepStatus.SUCCEEDED for status in statuses):
            return "succeeded"
        if any(status is PlanStepStatus.FAILED for status in statuses):
            return "failed"
        if any(status is PlanStepStatus.CANCELLED for status in statuses):
            return "cancelled"
        if any(status is PlanStepStatus.RUNNING for status in statuses):
            return "running"
        return "pending"

    async def _publish(self, topic: str, payload: Mapping[str, Any]) -> None:
        try:
            message = AgentMessage(
                channel=topic,
                payload=dict(payload),
                priority=MessagePriority.NORMAL,
                headers={"topic": topic, "task_id": str(payload.get("task_id", ""))},
            )
            await self._bus.publish(message)
        except Exception:  # pragma: no cover - message bus failures should not crash execution
            _LOGGER.exception("Failed to publish task event on topic '%s'", topic)

    def _publish_acceptance_criteria(
        self,
        state: _TaskRuntimeState,
        manifest: Mapping[str, Any],
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return

        raw_criteria = manifest.get("acceptance_criteria")
        if not isinstance(raw_criteria, Iterable) or isinstance(raw_criteria, (str, bytes, bytearray)):
            return

        for index, criterion in enumerate(raw_criteria, start=1):
            text = str(criterion).strip()
            if not text:
                continue
            try:
                blackboard_client.publish_hypothesis(
                    f"Acceptance criterion {index}",
                    text,
                    metadata={"task_id": state.task_id, "type": "acceptance_criterion"},
                )
            except Exception:  # pragma: no cover - collaboration aids should not interrupt execution
                _LOGGER.exception("Failed to publish acceptance criterion to blackboard")

    def _publish_progress(
        self,
        state: _TaskRuntimeState,
        step: PlanStep,
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return
        try:
            blackboard_client.publish_claim(
                f"Step completed: {step.identifier}",
                f"Tool '{step.tool_name}' completed for task {state.task_id}.",
                metadata={"task_id": state.task_id, "step_id": step.identifier},
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.exception("Failed to publish task progress to blackboard")

    def _publish_blocker(
        self,
        state: _TaskRuntimeState,
        failed_step: str,
        reason: str,
        cancellations: Iterable[tuple[str, str]],
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return

        try:
            blackboard_client.publish_claim(
                f"Blocker: {failed_step}",
                reason,
                metadata={"task_id": state.task_id, "step_id": failed_step},
            )
            for cancelled_step, message in cancellations:
                blackboard_client.publish_claim(
                    f"Cancelled: {cancelled_step}",
                    message,
                    metadata={"task_id": state.task_id, "step_id": cancelled_step},
                )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.exception("Failed to publish blocker update to blackboard")
