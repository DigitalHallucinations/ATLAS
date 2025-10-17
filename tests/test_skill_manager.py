import asyncio
import sys
import types
from typing import Any, Dict, List

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

import pytest

from ATLAS.SkillManager import (
    SKILL_ACTIVITY_EVENT,
    SkillExecutionContext,
    SkillExecutionError,
    SkillRunResult,
    use_skill,
)
from modules.Tools.tool_event_system import event_system


class _StubToolManager:
    def __init__(self, tool_entries: Dict[str, Any]):
        self._tool_entries = tool_entries
        self.calls: List[Any] = []

    def load_function_map_from_current_persona(self, persona, *, config_manager=None):
        self.calls.append((persona, config_manager))
        return dict(self._tool_entries)


def test_use_skill_executes_required_tools_and_emits_events():
    events: List[Dict[str, Any]] = []

    def _subscriber(payload):
        events.append(payload)

    event_system.subscribe(SKILL_ACTIVITY_EVENT, _subscriber)
    try:
        async def _runner():
            async def first_tool(message: str) -> str:
                await asyncio.sleep(0)
                return message.upper()

            async def second_tool(prefix: str, previous: str) -> str:
                await asyncio.sleep(0)
                return f"{prefix}:{previous}"

            tool_manager = _StubToolManager(
                {
                    "tool_one": {"callable": first_tool},
                    "tool_two": {"callable": second_tool},
                }
            )

            context = SkillExecutionContext(
                conversation_id="conv-123",
                conversation_history=[],
                persona={"name": "Analyst"},
                user={"id": "user-1"},
            )

            skill_metadata = {
                "name": "ExampleSkill",
                "instruction_prompt": "Summarize the conversation before notifying.",
                "required_tools": ["tool_one", "tool_two"],
            }

            result = await use_skill(
                skill_metadata,
                context=context,
                tool_inputs={
                    "tool_one": {"message": "hello"},
                    "tool_two": {"prefix": "result", "previous": "HELLO"},
                },
                tool_manager=tool_manager,
            )

            return result, tool_manager

        result, tool_manager = asyncio.run(_runner())

        assert isinstance(result, SkillRunResult)
        assert result.tool_results == {
            "tool_one": "HELLO",
            "tool_two": "result:HELLO",
        }

        assert tool_manager.calls == [({"name": "Analyst"}, None)]

        event_types = [entry["type"] for entry in events]
        assert "skill_started" in event_types
        assert "skill_plan_generated" in event_types
        assert event_types.count("tool_started") == 2
        assert event_types.count("tool_completed") == 2
        assert events[-1]["type"] == "skill_completed"
        assert events[-1]["tool_results"] == result.tool_results
    finally:
        event_system.unsubscribe(SKILL_ACTIVITY_EVENT, _subscriber)


def test_use_skill_missing_tool_raises():
    tool_manager = _StubToolManager({})
    context = SkillExecutionContext(
        conversation_id="conv-456",
        conversation_history=[],
    )

    skill_metadata = {
        "name": "BrokenSkill",
        "required_tools": ["nonexistent"],
    }

    async def _runner():
        return await use_skill(
            skill_metadata,
            context=context,
            tool_manager=tool_manager,
            budget_ms=10_000,
            timeout_seconds=1,
        )

    with pytest.raises(SkillExecutionError) as excinfo:
        asyncio.run(_runner())

    assert "requires unknown tools" in str(excinfo.value)


def test_use_skill_tool_failure_emits_event():
    events: List[Dict[str, Any]] = []

    def _subscriber(payload):
        events.append(payload)

    event_system.subscribe(SKILL_ACTIVITY_EVENT, _subscriber)
    try:
        async def _runner():
            async def failing_tool():
                raise RuntimeError("boom")

            tool_manager = _StubToolManager({"explode": {"callable": failing_tool}})
            context = SkillExecutionContext(
                conversation_id="conv-789",
                conversation_history=[],
            )

            skill_metadata = {
                "name": "FailureSkill",
                "required_tools": ["explode"],
            }

            await use_skill(skill_metadata, context=context, tool_manager=tool_manager)

        with pytest.raises(SkillExecutionError) as excinfo:
            asyncio.run(_runner())

        assert excinfo.value.tool_name == "explode"
        assert excinfo.value.skill_name == "FailureSkill"

        event_types = [entry["type"] for entry in events]
        assert "tool_failed" in event_types
        assert events[-1]["type"] == "skill_failed"
    finally:
        event_system.unsubscribe(SKILL_ACTIVITY_EVENT, _subscriber)

