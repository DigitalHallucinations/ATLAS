"""Asynchronous job orchestration utilities.

This module exposes :class:`JobManager`, a lightweight coordinator that loads
job manifests, builds execution DAGs, and invokes the task orchestrator for
each queued task.  Execution state is published on the ``jobs.*`` message-bus
topics and, when collaboration tooling is available, high-value updates are
surfaced on the shared blackboard so that teammates can follow progress in
real time.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from modules.Jobs import manifest_loader
from modules.Jobs.manifest_loader import JobMetadata
from modules.Tasks import manifest_loader as task_manifest_loader
from modules.orchestration.blackboard import BlackboardClient
from modules.orchestration.message_bus import MessageBus, MessagePriority, get_message_bus
from modules.orchestration.planner import ExecutionPlan, PlanStep, PlanStepStatus
from modules.orchestration.task_manager import TaskManager
from modules.orchestration.utils import normalize_persona_identifier

_LOGGER = logging.getLogger(__name__)

JOB_CREATED_TOPIC = "jobs.created"
JOB_UPDATED_TOPIC = "jobs.updated"
JOB_COMPLETED_TOPIC = "jobs.completed"


@dataclass
class JobRunResult:
    """Aggregated result returned after a job run completes."""

    job_id: str
    job_name: str
    status: str
    tasks: Mapping[str, Mapping[str, Any]]
    shared_artifacts: Mapping[str, Any]
    errors: Mapping[str, str]
    plan: ExecutionPlan
    snapshot: Mapping[str, Any]


@dataclass
class _JobRuntimeState:
    """Internal bookkeeping model tracking job execution state."""

    job_id: str
    job_name: str
    metadata: JobMetadata
    plan: ExecutionPlan
    nodes: Mapping[str, Mapping[str, Any]]
    shared_artifacts: Dict[str, Any] = field(default_factory=dict)
    task_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        for task_id in self.plan.steps:
            self.task_summaries.setdefault(
                task_id,
                {"status": "pending", "results": {}, "errors": {}},
            )

    def snapshot(self, *, current_task: Optional[str] = None) -> Dict[str, Any]:
        """Return a serialisable snapshot of the current state."""

        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status,
            "current_task": current_task,
            "shared_artifacts": {
                task_id: _clone_artifact(artifact)
                for task_id, artifact in self.shared_artifacts.items()
            },
            "errors": dict(self.errors),
            "tasks": {
                task_id: {
                    "status": summary.get("status", "pending"),
                    "results": _clone_artifact(summary.get("results")),
                    "errors": dict(summary.get("errors", {})),
                }
                for task_id, summary in self.task_summaries.items()
            },
            "plan": self.plan.snapshot(),
        }


def build_task_manifest_resolver(*, config_manager=None) -> Callable[[str, Optional[str]], Optional[Mapping[str, Any]]]:
    """Return a resolver that maps job step identifiers to task manifests."""

    manifest_cache: Dict[Optional[str], Dict[str, Mapping[str, Any]]] | None = None
    casefold_cache: Dict[Optional[str], Dict[str, Mapping[str, Any]]] | None = None

    def _ensure_cache() -> tuple[Dict[Optional[str], Dict[str, Mapping[str, Any]]], Dict[Optional[str], Dict[str, Mapping[str, Any]]]]:
        nonlocal manifest_cache, casefold_cache
        if manifest_cache is None or casefold_cache is None:
            manifest_cache = {}
            casefold_cache = {}
            entries = task_manifest_loader.load_task_metadata(config_manager=config_manager)
            for entry in entries:
                persona_key = normalize_persona_identifier(entry.persona)
                manifest = _manifest_from_metadata(entry)
                manifest_cache.setdefault(persona_key, {})[entry.name] = manifest
                casefold_cache.setdefault(persona_key, {})[entry.name.lower()] = manifest
        return manifest_cache, casefold_cache

    def _lookup(task_id: str, persona_key: Optional[str]) -> Optional[Mapping[str, Any]]:
        cache, fallback = _ensure_cache()
        explicit = cache.get(persona_key, {})
        manifest = explicit.get(task_id)
        if manifest is not None:
            return manifest
        lowered = task_id.lower()
        persona_fallback = fallback.get(persona_key, {})
        return persona_fallback.get(lowered)

    def _resolver(task_id: str, persona: Optional[str] = None) -> Optional[Mapping[str, Any]]:
        if not isinstance(task_id, str):
            return None
        normalized_id = task_id.strip()
        if not normalized_id:
            return None

        persona_key = normalize_persona_identifier(persona)
        manifest = _lookup(normalized_id, persona_key)
        if manifest is None and persona_key is not None:
            manifest = _lookup(normalized_id, None)
        if manifest is None:
            return None
        return dict(manifest)

    return _resolver


def _manifest_from_metadata(metadata: task_manifest_loader.TaskMetadata) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "name": metadata.name,
        "summary": metadata.summary,
        "description": metadata.description,
        "required_skills": list(metadata.required_skills),
        "required_tools": list(metadata.required_tools),
        "acceptance_criteria": list(metadata.acceptance_criteria),
        "escalation_policy": dict(metadata.escalation_policy),
    }
    if metadata.tags:
        manifest["tags"] = list(metadata.tags)
    if metadata.priority:
        manifest["priority"] = metadata.priority
    if metadata.persona:
        manifest["persona"] = metadata.persona
    if metadata.source:
        manifest["source"] = metadata.source
    return manifest


class JobManager:
    """Coordinate execution of job manifests across multiple tasks."""

    def __init__(
        self,
        task_manager: TaskManager,
        *,
        message_bus: Optional[MessageBus] = None,
        job_loader: Optional[Callable[[], Iterable[JobMetadata]]] = None,
        task_resolver: Optional[Callable[[str, Optional[str]], Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self._task_manager = task_manager
        self._bus = message_bus or get_message_bus()
        self._job_loader = job_loader or manifest_loader.load_job_metadata
        self._task_resolver = task_resolver or build_task_manifest_resolver()

    async def run_job(
        self,
        job_name: str,
        *,
        persona: Optional[str] = None,
        run_id: Optional[str] = None,
        task_manifests: Optional[Mapping[str, Mapping[str, Any]]] = None,
        blackboard_client: Optional[BlackboardClient] = None,
    ) -> JobRunResult:
        """Execute ``job_name`` and return an aggregated run result."""

        metadata = self._resolve_job(job_name, persona)
        if metadata is None:
            raise ValueError(f"Unknown job '{job_name}' for persona '{persona or 'default'}'")

        plan, nodes = self._build_plan(metadata)
        job_id = run_id or uuid.uuid4().hex
        state = _JobRuntimeState(
            job_id=job_id,
            job_name=metadata.name,
            metadata=metadata,
            plan=plan,
            nodes=nodes,
        )

        await self._publish(JOB_CREATED_TOPIC, state.snapshot())
        self._publish_acceptance_criteria(state, blackboard_client)
        self._publish_escalation_policy(state, blackboard_client)

        failed = False
        while plan.unfinished():
            ready = plan.ready_steps()
            if not ready:
                break

            for step in ready:
                if plan.status(step.identifier) is not PlanStepStatus.PENDING:
                    continue

                success = await self._execute_task(
                    state,
                    step,
                    task_manifests=task_manifests,
                    blackboard_client=blackboard_client,
                )
                if not success:
                    failed = True
                    break

            if failed:
                break

        final_status = self._determine_final_status(plan)
        state.status = final_status
        final_snapshot = state.snapshot()
        await self._publish(JOB_COMPLETED_TOPIC, final_snapshot)

        return JobRunResult(
            job_id=state.job_id,
            job_name=state.job_name,
            status=final_status,
            tasks={task_id: dict(summary) for task_id, summary in state.task_summaries.items()},
            shared_artifacts={
                task_id: _clone_artifact(artifact)
                for task_id, artifact in state.shared_artifacts.items()
            },
            errors=dict(state.errors),
            plan=plan,
            snapshot=final_snapshot,
        )

    def _resolve_job(self, job_name: str, persona: Optional[str]) -> Optional[JobMetadata]:
        persona_key = persona or None
        entries = list(self._job_loader())

        for entry in entries:
            if entry.name == job_name and entry.persona == persona_key:
                return entry

        for entry in entries:
            if entry.name == job_name and entry.persona is None:
                return entry

        return None

    def _build_plan(
        self, metadata: JobMetadata
    ) -> tuple[ExecutionPlan, Dict[str, Mapping[str, Any]]]:
        steps: Dict[str, PlanStep] = {}
        dependents: Dict[str, list[str]] = defaultdict(list)
        nodes: Dict[str, Mapping[str, Any]] = {}

        for node in metadata.task_graph:
            task_id = str(node.get("task", "")).strip()
            if not task_id:
                continue
            dependencies = tuple(node.get("depends_on", tuple()))
            steps[task_id] = PlanStep(
                identifier=task_id,
                tool_name=task_id,
                dependencies=dependencies,
                inputs=node.get("metadata", {}),
            )
            nodes[task_id] = dict(node)

        for step in steps.values():
            for dependency in step.dependencies:
                if dependency not in steps:
                    raise ValueError(
                        f"Job task '{step.identifier}' depends on unknown task '{dependency}'"
                    )
                dependents.setdefault(dependency, []).append(step.identifier)

        plan = ExecutionPlan(steps=steps, dependents=dependents)
        return plan, nodes

    async def _execute_task(
        self,
        state: _JobRuntimeState,
        step: PlanStep,
        *,
        task_manifests: Optional[Mapping[str, Mapping[str, Any]]],
        blackboard_client: Optional[BlackboardClient],
    ) -> bool:
        manifest = self._resolve_task_manifest(state, step.identifier, task_manifests)
        if manifest is None:
            reason = f"No manifest found for job task '{step.identifier}'"
            state.errors[step.identifier] = reason
            state.task_summaries[step.identifier]["status"] = "failed"
            state.task_summaries[step.identifier]["errors"] = {"job": reason}
            cancellations = state.plan.mark_failed(step.identifier, reason)
            self._record_cancellations(state, cancellations)
            await self._publish(
                JOB_UPDATED_TOPIC,
                state.snapshot(current_task=step.identifier),
            )
            self._publish_blocker(state, step.identifier, reason, cancellations, blackboard_client)
            return False

        manifest = dict(manifest)
        manifest.setdefault("id", manifest.get("id") or f"{state.job_id}:{step.identifier}")
        manifest.setdefault("name", manifest.get("name") or step.identifier)

        try:
            state.plan.mark_running(step.identifier)
            state.status = "running"
            await self._publish(
                JOB_UPDATED_TOPIC,
                state.snapshot(current_task=step.identifier),
            )

            result = await self._task_manager.run_task(
                manifest,
                blackboard_client=blackboard_client,
            )
        except Exception as exc:  # pylint: disable=broad-except
            reason = str(exc) or exc.__class__.__name__
            _LOGGER.exception(
                "Task '%s' for job '%s' raised an exception", step.identifier, state.job_name
            )
            state.errors[step.identifier] = reason
            state.task_summaries[step.identifier]["status"] = "failed"
            state.task_summaries[step.identifier]["errors"] = {"exception": reason}
            cancellations = state.plan.mark_failed(step.identifier, reason)
            self._record_cancellations(state, cancellations)
            await self._publish(
                JOB_UPDATED_TOPIC,
                state.snapshot(current_task=step.identifier),
            )
            self._publish_blocker(state, step.identifier, reason, cancellations, blackboard_client)
            return False

        status = getattr(result, "status", None)
        if status != "succeeded":
            reason = _failure_reason(result, step.identifier)
            state.errors[step.identifier] = reason
            summary = state.task_summaries[step.identifier]
            summary["status"] = status or "failed"
            summary["errors"] = dict(getattr(result, "errors", {}) or {"job": reason})
            cancellations = state.plan.mark_failed(step.identifier, reason)
            self._record_cancellations(state, cancellations)
            await self._publish(
                JOB_UPDATED_TOPIC,
                state.snapshot(current_task=step.identifier),
            )
            self._publish_blocker(state, step.identifier, reason, cancellations, blackboard_client)
            return False

        artifact = getattr(result, "results", {})
        state.shared_artifacts[step.identifier] = _clone_artifact(artifact)
        summary = state.task_summaries[step.identifier]
        summary["status"] = "succeeded"
        summary["results"] = _clone_artifact(artifact)
        summary["errors"] = dict(getattr(result, "errors", {}) or {})
        state.plan.mark_succeeded(step.identifier)
        await self._publish(
            JOB_UPDATED_TOPIC,
            state.snapshot(current_task=step.identifier),
        )
        return True

    def _resolve_task_manifest(
        self,
        state: _JobRuntimeState,
        task_id: str,
        task_manifests: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Optional[Mapping[str, Any]]:
        if task_manifests and task_id in task_manifests:
            manifest = task_manifests[task_id]
            if manifest is not None:
                return manifest

        node_manifest = state.nodes.get(task_id, {}).get("manifest")
        if isinstance(node_manifest, Mapping):
            return node_manifest

        if self._task_resolver:
            resolved = self._task_resolver(task_id, state.metadata.persona)
            if resolved is not None:
                return resolved

        return None

    def _record_cancellations(
        self,
        state: _JobRuntimeState,
        cancellations: Iterable[tuple[str, str]],
    ) -> None:
        for cancelled_step, message in cancellations:
            state.errors.setdefault(cancelled_step, message)
            summary = state.task_summaries.setdefault(
                cancelled_step,
                {"status": "cancelled", "results": {}, "errors": {}},
            )
            summary["status"] = "cancelled"
            summary_errors = dict(summary.get("errors", {}))
            summary_errors.setdefault("job", message)
            summary["errors"] = summary_errors

    @staticmethod
    def _determine_final_status(plan: ExecutionPlan) -> str:
        statuses = [plan.status(step_id) for step_id in plan.steps]
        if not statuses:
            return "succeeded"
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
            await self._bus.publish(
                topic,
                dict(payload),
                priority=MessagePriority.NORMAL,
                metadata={
                    "topic": topic,
                    "job_id": payload.get("job_id"),
                    "job_name": payload.get("job_name"),
                },
            )
        except Exception:  # pragma: no cover - defensive guard for bus failures
            _LOGGER.exception("Failed to publish job event on topic '%s'", topic)

    def _publish_acceptance_criteria(
        self,
        state: _JobRuntimeState,
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return

        for index, criterion in enumerate(state.metadata.acceptance_criteria, start=1):
            text = str(criterion).strip()
            if not text:
                continue
            try:
                blackboard_client.publish_hypothesis(
                    f"Job acceptance criterion {index}",
                    text,
                    metadata={"job_id": state.job_id, "type": "acceptance_criterion"},
                )
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.exception("Failed to publish job acceptance criterion to blackboard")

    def _publish_escalation_policy(
        self,
        state: _JobRuntimeState,
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return

        policy = state.metadata.escalation_policy or {}
        if not isinstance(policy, Mapping) or not any(str(policy.get(key, "")).strip() for key in policy):
            return

        lines = []
        level = str(policy.get("level", "")).strip()
        contact = str(policy.get("contact", "")).strip()
        if level:
            lines.append(f"Level: {level}")
        if contact:
            lines.append(f"Contact: {contact}")

        for key, value in policy.items():
            if key in {"level", "contact"}:
                continue
            if isinstance(value, (list, tuple)):
                rendered = ", ".join(str(item) for item in value if str(item).strip())
            else:
                rendered = str(value).strip()
            if rendered:
                lines.append(f"{key.replace('_', ' ').title()}: {rendered}")

        if not lines:
            return

        try:
            blackboard_client.publish_claim(
                "Escalation policy",
                "\n".join(lines),
                metadata={"job_id": state.job_id, "type": "escalation_policy"},
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.exception("Failed to publish job escalation policy to blackboard")

    def _publish_blocker(
        self,
        state: _JobRuntimeState,
        failed_task: str,
        reason: str,
        cancellations: Iterable[tuple[str, str]],
        blackboard_client: Optional[BlackboardClient],
    ) -> None:
        if blackboard_client is None:
            return

        try:
            blackboard_client.publish_claim(
                f"Job blocker: {failed_task}",
                reason,
                metadata={"job_id": state.job_id, "task_id": failed_task},
            )
            for cancelled_step, message in cancellations:
                blackboard_client.publish_claim(
                    f"Job cancelled: {cancelled_step}",
                    message,
                    metadata={"job_id": state.job_id, "task_id": cancelled_step},
                )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.exception("Failed to publish job blocker update to blackboard")


def _clone_artifact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_artifact(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clone_artifact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_artifact(item) for item in value)
    return value


def _failure_reason(result: Any, task_id: str) -> str:
    errors = getattr(result, "errors", None)
    if isinstance(errors, Mapping):
        for _, message in errors.items():
            text = str(message).strip()
            if text:
                return text
    status = getattr(result, "status", None)
    if isinstance(status, str) and status:
        return f"Task '{task_id}' returned status '{status}'"
    return f"Task '{task_id}' failed"


__all__ = [
    "JOB_COMPLETED_TOPIC",
    "JOB_CREATED_TOPIC",
    "JOB_UPDATED_TOPIC",
    "build_task_manifest_resolver",
    "JobManager",
    "JobRunResult",
]

