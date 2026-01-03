from __future__ import annotations

import asyncio

from modules.orchestration.followups import FollowUpOrchestrator


class _StubTaskManager:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object] | None]] = []

    async def run_task(
        self,
        manifest: dict[str, object],
        *,
        provided_inputs: dict[str, object] | None = None,
        blackboard_client=None,
    ) -> None:
        self.calls.append((manifest, provided_inputs))


class _StubJobManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def run_job(
        self,
        job_name: str,
        *,
        persona: str | None = None,
        run_id=None,
        task_manifests=None,
        blackboard_client=None,
    ) -> None:
        self.calls.append((job_name, persona))


def test_followup_orchestrator_dispatches_task_and_job():
    task_manager = _StubTaskManager()
    job_manager = _StubJobManager()

    def resolver(name: str, persona: str | None) -> dict[str, object] | None:
        return {
            "name": name,
            "summary": "",
            "description": "",
            "required_skills": (),
            "required_tools": (),
            "acceptance_criteria": (),
            "escalation_policy": {},
            "tags": [],
        }

    orchestrator = FollowUpOrchestrator(
        task_manager=task_manager,
        job_manager=job_manager,
        task_resolver=resolver,
    )

    payload = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-42",
        "persona": "knowledgecurator",
        "followups": [
            {
                "id": "outstanding-question::message-1",
                "kind": "question",
                "title": "Respond to outstanding question",
                "description": "Question awaiting response",
                "reasons": ["unanswered_question"],
                "source": {"type": "message", "role": "user", "index": 1},
                "task": {
                    "manifest": "ClarifyOutstandingQuestion",
                    "priority": "high",
                    "inputs": {"context": {"question": "Can you send the report?"}},
                },
                "escalation": {"job": "EscalateFollowUp", "persona": "atlasops"},
            }
        ],
    }

    asyncio.run(orchestrator.process_event(payload, wait=True))

    assert len(task_manager.calls) == 1
    manifest, inputs = task_manager.calls[0]
    assert manifest["name"] == "ClarifyOutstandingQuestion"
    assert manifest["priority"] == "high"
    assert "followup" in manifest["tags"]
    metadata = manifest["metadata"]
    assert metadata["followup_id"] == "outstanding-question::message-1"
    assert inputs == {"context": {"question": "Can you send the report?"}}

    # Job escalations should also be dispatched.
    assert job_manager.calls == [("EscalateFollowUp", "atlasops")]
