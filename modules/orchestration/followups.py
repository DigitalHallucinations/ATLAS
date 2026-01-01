"""Handle follow-up events emitted from conversation summaries."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence

from modules.orchestration.job_manager import JobManager, build_task_manifest_resolver
from modules.orchestration.task_manager import TaskManager

from ATLAS.messaging import (
    AgentBus,
    AgentMessage,
    Subscription,
    get_agent_bus,
    FOLLOWUP,
)


class FollowUpOrchestrator:
    """Dispatch follow-up actions into the task and job managers."""

    TOPIC = FOLLOWUP.name

    def __init__(
        self,
        *,
        task_manager: TaskManager | None = None,
        job_manager: JobManager | None = None,
        agent_bus: AgentBus | None = None,
        task_resolver: Callable[[str, str | None], Mapping[str, Any] | None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._task_manager = task_manager
        self._job_manager = job_manager
        self._bus = agent_bus or get_agent_bus()
        self._task_resolver = task_resolver or build_task_manifest_resolver()
        self._logger = logger or logging.getLogger(__name__)
        self._subscription: Subscription | None = None
        self._pending: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------
    # Lifecycle management

    @property
    def is_running(self) -> bool:
        return self._subscription is not None

    async def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = await self._bus.subscribe(self.TOPIC, self._handle_message)

    async def stop(self) -> None:
        if self._subscription is not None:
            await self._subscription.cancel()
            self._subscription = None
        for task in list(self._pending):
            task.cancel()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Event handling

    async def process_event(self, payload: Mapping[str, Any], *, wait: bool = False) -> None:
        """Handle a follow-up payload, optionally awaiting completion."""

        if not isinstance(payload, Mapping):
            return
        followups = payload.get("followups")
        if not isinstance(followups, Sequence):
            return

        coroutines = [self._dispatch_followup(payload, followup) for followup in followups if isinstance(followup, Mapping)]
        if not coroutines:
            return

        if wait:
            await asyncio.gather(*coroutines)
        else:
            for coroutine in coroutines:
                self._track_task(coroutine)

    async def _handle_message(self, message: AgentMessage) -> None:
        payload = message.payload
        if not isinstance(payload, Mapping):
            self._logger.debug("Ignoring follow-up event without mapping payload: %s", payload)
            return
        await self.process_event(payload, wait=False)

    # ------------------------------------------------------------------
    # Internal helpers

    async def _dispatch_followup(self, event_payload: Mapping[str, Any], followup: Mapping[str, Any]) -> None:
        await asyncio.gather(
            self._maybe_run_task(event_payload, followup),
            self._maybe_schedule_job(event_payload, followup),
        )

    async def _maybe_run_task(self, event_payload: Mapping[str, Any], followup: Mapping[str, Any]) -> None:
        if self._task_manager is None:
            return
        task_spec = followup.get("task")
        if not isinstance(task_spec, Mapping):
            return

        manifest_name = str(task_spec.get("manifest") or task_spec.get("name") or "").strip()
        if not manifest_name:
            self._logger.debug(
                "Follow-up %s missing manifest reference; skipping task dispatch",
                followup.get("id"),
            )
            return

        persona = task_spec.get("persona") or event_payload.get("persona")
        try:
            manifest = self._task_resolver(manifest_name, persona)
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception("Task resolver failed for follow-up manifest %s", manifest_name)
            return

        if manifest is None:
            self._logger.debug("No manifest found for follow-up %s (%s)", followup.get("id"), manifest_name)
            return

        manifest_payload = self._augment_manifest(dict(manifest), event_payload, followup, task_spec)
        inputs = task_spec.get("inputs") if isinstance(task_spec.get("inputs"), Mapping) else None

        await self._task_manager.run_task(manifest_payload, provided_inputs=inputs)

    async def _maybe_schedule_job(self, event_payload: Mapping[str, Any], followup: Mapping[str, Any]) -> None:
        if self._job_manager is None:
            return

        escalation = followup.get("escalation")
        if not isinstance(escalation, Mapping):
            return

        job_name = str(escalation.get("job") or escalation.get("name") or "").strip()
        if not job_name:
            self._logger.debug("Follow-up %s missing job identifier; skipping escalation", followup.get("id"))
            return

        persona = escalation.get("persona") or event_payload.get("persona")
        delay_minutes = escalation.get("delay_minutes")
        try:
            delay_seconds = float(delay_minutes) * 60 if delay_minutes is not None else 0.0
        except (TypeError, ValueError):
            delay_seconds = 0.0

        async def _runner() -> None:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            await self._job_manager.run_job(job_name, persona=persona)

        if delay_seconds > 0:
            self._track_task(_runner())
        else:
            await _runner()

    def _augment_manifest(
        self,
        manifest: MutableMapping[str, Any],
        event_payload: Mapping[str, Any],
        followup: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> MutableMapping[str, Any]:
        manifest = dict(manifest)
        manifest.setdefault("id", f"{followup.get('id')}-task")
        tags = list(manifest.get("tags", []))
        if "followup" not in {str(tag).lower() for tag in tags}:
            tags.append("followup")
        manifest["tags"] = tags

        metadata = dict(manifest.get("metadata", {}))
        metadata.setdefault("source", "conversation_followup")
        metadata["followup_id"] = followup.get("id")
        metadata["conversation_id"] = event_payload.get("conversation_id")
        metadata["tenant_id"] = event_payload.get("tenant_id")
        metadata["followup_kind"] = followup.get("kind")
        manifest["metadata"] = metadata

        priority = spec.get("priority")
        if isinstance(priority, str) and priority.strip():
            manifest["priority"] = priority.strip()

        return manifest

    def _track_task(self, coroutine: Awaitable[Any]) -> None:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coroutine)
        self._pending.add(task)

        def _cleanup(completed: asyncio.Task[Any]) -> None:
            self._pending.discard(completed)
            if completed.cancelled():
                return
            exc = completed.exception()
            if exc is not None:
                self._logger.error("Follow-up orchestration task failed: %s", exc, exc_info=True)

        task.add_done_callback(_cleanup)


__all__ = ["FollowUpOrchestrator"]
