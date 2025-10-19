"""Provider that routes task queue requests to the default service."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict

from modules.Tools.Base_Tools import task_queue as task_queue_module

from .base import ToolProvider
from .registry import tool_provider_registry


_CALL_DISPATCH: Dict[str, Callable[..., Any]] = {
    "task_queue_enqueue": task_queue_module.enqueue_task,
    "task_queue_schedule": task_queue_module.schedule_cron_task,
    "task_queue_cancel": task_queue_module.cancel_task,
    "task_queue_status": task_queue_module.get_task_status,
}


class TaskQueueDefaultProvider(ToolProvider):
    """Adapter around :mod:`modules.Tools.Base_Tools.task_queue`."""

    async def call(self, **kwargs: Any) -> Any:
        handler = _CALL_DISPATCH.get(self.tool_name)
        if handler is None:
            raise RuntimeError(f"Unsupported task queue operation '{self.tool_name}'.")
        return await asyncio.to_thread(handler, **kwargs)

    async def health_check(self) -> bool:
        try:
            task_queue_module.get_default_task_queue_service()
        except Exception:  # pragma: no cover - defensive guard
            return False
        return True


tool_provider_registry.register("task_queue_default", TaskQueueDefaultProvider)

__all__ = ["TaskQueueDefaultProvider"]

