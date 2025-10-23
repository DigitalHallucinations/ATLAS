import asyncio
from collections import defaultdict
from types import MappingProxyType, SimpleNamespace

from modules.Jobs.manifest_loader import JobMetadata
from modules.orchestration.blackboard import BlackboardStore, configure_blackboard
from modules.orchestration.job_manager import (
    JOB_COMPLETED_TOPIC,
    JOB_CREATED_TOPIC,
    JOB_UPDATED_TOPIC,
    JobManager,
)
from modules.orchestration.message_bus import InMemoryQueueBackend, MessageBus


class StubTaskManager:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    async def run_task(self, manifest, *, blackboard_client=None):
        task_id = manifest["id"]
        self.calls.append(task_id)
        outcome = self.outcomes.get(task_id, {})
        if "exception" in outcome:
            raise outcome["exception"]
        status = outcome.get("status", "succeeded")
        return SimpleNamespace(
            task_id=task_id,
            status=status,
            results=outcome.get("results", {"artifact": task_id}),
            errors=outcome.get("errors", {}),
        )


def _job_metadata(task_graph, *, acceptance=None, escalation=None) -> JobMetadata:
    return JobMetadata(
        name="DemoJob",
        summary="",
        description="",
        personas=tuple(),
        required_skills=tuple(),
        required_tools=tuple(),
        task_graph=tuple(task_graph),
        recurrence=MappingProxyType({}),
        acceptance_criteria=tuple(acceptance or ()),
        escalation_policy=MappingProxyType(escalation or {}),
        persona=None,
        source="tests/job.json",
    )


def test_job_manager_executes_tasks_with_dependencies():
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        message_bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop)
        try:
            store = BlackboardStore()
            configure_blackboard(store)
            store._publish_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

            events = defaultdict(list)
            finished = asyncio.Event()

            def _subscribe(topic: str):
                async def handler(message):
                    events[topic].append(message.payload)
                    if topic == JOB_COMPLETED_TOPIC:
                        finished.set()

                return message_bus.subscribe(topic, handler)

            created_sub = _subscribe(JOB_CREATED_TOPIC)
            updated_sub = _subscribe(JOB_UPDATED_TOPIC)
            completed_sub = _subscribe(JOB_COMPLETED_TOPIC)

            try:
                outcomes = {
                    "collect": {"results": {"data": [1, 2, 3]}},
                    "analyze": {"results": {"summary": "ok"}},
                    "report": {"results": {"status": "shared"}},
                }
                stub_manager = StubTaskManager(outcomes)

                job = _job_metadata(
                    (
                        MappingProxyType({"task": "collect"}),
                        MappingProxyType({"task": "analyze", "depends_on": ("collect",)}),
                        MappingProxyType({"task": "report", "depends_on": ("analyze",)}),
                    ),
                    acceptance=["Data collected", "Insights produced"],
                    escalation={"level": "tier1", "contact": "oncall@example.com"},
                )

                task_manifests = {
                    "collect": {"id": "collect", "name": "Collect", "plan": {}},
                    "analyze": {"id": "analyze", "name": "Analyze", "plan": {}},
                    "report": {"id": "report", "name": "Report", "plan": {}},
                }

                manager = JobManager(
                    stub_manager,
                    message_bus=message_bus,
                    job_loader=lambda: [job],
                )

                client = store.client_for("job-1", scope_type="project")
                result = await manager.run_job(
                    "DemoJob",
                    run_id="job-1",
                    task_manifests=task_manifests,
                    blackboard_client=client,
                )

                await asyncio.wait_for(finished.wait(), timeout=1.0)

                assert stub_manager.calls == ["collect", "analyze", "report"]
                assert result.status == "succeeded"
                assert set(result.shared_artifacts.keys()) == {"collect", "analyze", "report"}

                hypotheses = client.list_entries(category="hypothesis")
                assert len(hypotheses) == 2
                claims = client.list_entries(category="claim")
                assert any(entry["title"] == "Escalation policy" for entry in claims)

                assert len(events[JOB_CREATED_TOPIC]) == 1
                assert events[JOB_CREATED_TOPIC][0]["job_id"] == "job-1"
                assert events[JOB_COMPLETED_TOPIC][-1]["status"] == "succeeded"
                assert any(payload["status"] == "running" for payload in events[JOB_UPDATED_TOPIC])
            finally:
                created_sub.cancel()
                updated_sub.cancel()
                completed_sub.cancel()
        finally:
            await message_bus.close()

    asyncio.run(_run())


def test_job_manager_propagates_failures_and_cancellations():
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        message_bus = MessageBus(backend=InMemoryQueueBackend(), loop=loop)
        try:
            store = BlackboardStore()
            configure_blackboard(store)
            store._publish_event = lambda *args, **kwargs: None  # type: ignore[attr-defined]

            events = defaultdict(list)
            finished = asyncio.Event()

            def _subscribe(topic: str):
                async def handler(message):
                    events[topic].append(message.payload)
                    if topic == JOB_COMPLETED_TOPIC:
                        finished.set()

                return message_bus.subscribe(topic, handler)

            created_sub = _subscribe(JOB_CREATED_TOPIC)
            updated_sub = _subscribe(JOB_UPDATED_TOPIC)
            completed_sub = _subscribe(JOB_COMPLETED_TOPIC)

            try:
                outcomes = {
                    "gather": {"results": {"records": 10}},
                    "synthesize": {
                        "status": "failed",
                        "errors": {"message": "analysis failure"},
                    },
                }
                stub_manager = StubTaskManager(outcomes)

                job = _job_metadata(
                    (
                        MappingProxyType({"task": "gather"}),
                        MappingProxyType({"task": "synthesize", "depends_on": ("gather",)}),
                        MappingProxyType({"task": "deliver", "depends_on": ("synthesize",)}),
                    ),
                    acceptance=["Insights delivered"],
                    escalation={"level": "tier1", "contact": "duty@example.com"},
                )

                task_manifests = {
                    "gather": {"id": "gather", "name": "Gather", "plan": {}},
                    "synthesize": {"id": "synthesize", "name": "Synthesize", "plan": {}},
                    "deliver": {"id": "deliver", "name": "Deliver", "plan": {}},
                }

                manager = JobManager(
                    stub_manager,
                    message_bus=message_bus,
                    job_loader=lambda: [job],
                )

                client = store.client_for("job-2", scope_type="project")
                result = await manager.run_job(
                    "DemoJob",
                    run_id="job-2",
                    task_manifests=task_manifests,
                    blackboard_client=client,
                )

                await asyncio.wait_for(finished.wait(), timeout=1.0)

                assert stub_manager.calls == ["gather", "synthesize"]
                assert result.status == "failed"
                assert result.tasks["synthesize"]["status"] == "failed"
                assert result.tasks["deliver"]["status"] == "cancelled"

                claims = client.list_entries(category="claim")
                titles = {entry["title"] for entry in claims}
                assert any(title.startswith("Job blocker: synthesize") for title in titles)
                assert any(title.startswith("Job cancelled: deliver") for title in titles)

                assert len(events[JOB_CREATED_TOPIC]) == 1
                assert events[JOB_COMPLETED_TOPIC][-1]["status"] == "failed"
                assert any(payload["tasks"]["synthesize"]["status"] == "failed" for payload in events[JOB_UPDATED_TOPIC])
            finally:
                created_sub.cancel()
                updated_sub.cancel()
                completed_sub.cancel()
        finally:
            await message_bus.close()

    asyncio.run(_run())

