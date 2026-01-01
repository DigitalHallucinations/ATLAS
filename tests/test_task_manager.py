import asyncio
from collections import defaultdict

from ATLAS.messaging import AgentBus, AgentMessage
from modules.orchestration.blackboard import BlackboardStore, configure_blackboard
from modules.orchestration.task_manager import (
    TASK_COMPLETED_TOPIC,
    TASK_CREATED_TOPIC,
    TASK_UPDATED_TOPIC,
    TaskManager,
)


def test_task_manager_executes_ready_steps():
    async def _run() -> None:
        execution_order = []
        agent_bus = AgentBus()
        await agent_bus.start()
        try:
            store = BlackboardStore()
            configure_blackboard(store)
            store._publish_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

            async def record_runner(step, context):
                execution_order.append(step.identifier)
                return {"tool": step.tool_name, "inputs": dict(step.inputs)}

            manager = TaskManager(
                {"alpha": record_runner, "beta": record_runner, "gamma": record_runner},
                agent_bus=agent_bus,
            )

            manifest = {
                "id": "task-1",
                "name": "Demo task",
                "plan": {
                    "steps": [
                        {"id": "collect", "tool": "alpha"},
                        {"id": "analyze", "tool": "beta", "after": ["collect"]},
                        {"id": "summarize", "tool": "gamma", "after": ["analyze"]},
                    ]
                },
                "acceptance_criteria": [
                    "Collect data",
                    "Analyze it",
                    "Summarize findings",
                ],
            }

            client = store.client_for("task-1")
            result = await manager.run_task(manifest, blackboard_client=client)

            assert execution_order == ["collect", "analyze", "summarize"]
            assert result.status == "succeeded"
            assert set(result.results.keys()) == {"collect", "analyze", "summarize"}

            hypotheses = [entry for entry in client.list_entries(category="hypothesis")]
            assert len(hypotheses) == 3
        finally:
            await agent_bus.stop()

    asyncio.run(_run())


def test_task_manager_failure_cancels_dependents():
    async def _run() -> None:
        agent_bus = AgentBus()
        await agent_bus.start()
        try:
            store = BlackboardStore()
            configure_blackboard(store)
            store._publish_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

            async def success_runner(step, _context):
                return {"step": step.identifier}

            async def failing_runner(step, _context):
                raise RuntimeError(f"{step.tool_name} failure")

            manager = TaskManager(
                {"alpha": success_runner, "beta": failing_runner, "gamma": success_runner},
                agent_bus=agent_bus,
            )

            manifest = {
                "id": "task-2",
                "name": "Failure propagation",
                "plan": {
                    "steps": [
                        {"id": "prepare", "tool": "alpha"},
                        {"id": "execute", "tool": "beta", "after": ["prepare"]},
                        {"id": "report", "tool": "gamma", "after": ["execute"]},
                    ]
                },
                "acceptance_criteria": ["Should reach report"],
            }

            client = store.client_for("task-2")
            result = await manager.run_task(manifest, blackboard_client=client)

            assert result.status == "failed"
            assert result.plan.status("prepare").value == "succeeded"
            assert result.plan.status("execute").value == "failed"
            assert result.plan.status("report").value == "cancelled"

            claims = client.list_entries(category="claim")
            titles = {entry["title"] for entry in claims}
            assert any(title.startswith("Blocker: execute") for title in titles)
            assert any(title.startswith("Cancelled: report") for title in titles)
        finally:
            await agent_bus.stop()

    asyncio.run(_run())


def test_task_manager_publishes_bus_events():
    async def _run() -> None:
        agent_bus = AgentBus()
        await agent_bus.start()
        subscriptions = []
        try:
            events = defaultdict(list)
            finished = asyncio.Event()
            store = BlackboardStore()
            configure_blackboard(store)
            store._publish_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

            async def _subscribe(topic: str):
                async def handler(message: AgentMessage):
                    events[topic].append(message.payload)
                    if topic == TASK_COMPLETED_TOPIC:
                        finished.set()

                sub = await agent_bus.subscribe(topic, handler)
                subscriptions.append(sub)
                return sub

            await _subscribe(TASK_CREATED_TOPIC)
            await _subscribe(TASK_UPDATED_TOPIC)
            await _subscribe(TASK_COMPLETED_TOPIC)

            async def runner(step, _context):
                return {"id": step.identifier}

            manager = TaskManager({"alpha": runner}, agent_bus=agent_bus)

            manifest = {
                "id": "task-3",
                "name": "Event emission",
                "plan": {"steps": [{"id": "only", "tool": "alpha"}]},
            }

            await manager.run_task(manifest)
            await asyncio.wait_for(finished.wait(), timeout=2.0)

            assert len(events[TASK_CREATED_TOPIC]) == 1
            assert events[TASK_CREATED_TOPIC][0]["task_id"] == "task-3"
            assert events[TASK_COMPLETED_TOPIC][-1]["status"] == "succeeded"
            assert any(payload["status"] == "running" for payload in events[TASK_UPDATED_TOPIC])
        finally:
            for sub in subscriptions:
                await sub.cancel()
            await agent_bus.stop()

    asyncio.run(_run())
