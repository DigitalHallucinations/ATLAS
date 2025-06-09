import asyncio
from typing import List, Coroutine, Any

class ThreadOrchestrator:
    """Simple orchestrator for managing asyncio tasks."""

    def __init__(self):
        self.tasks: List[asyncio.Task] = []

    async def start_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def stop_all(self) -> None:
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.tasks.clear()
