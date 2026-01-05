import asyncio
import time
from typing import Any, Dict, List

import pytest

from core.SkillManager import (
    SKILL_ACTIVITY_EVENT,
    SkillExecutionContext,
    SkillExecutionError,
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

    def load_functions_from_json(self, persona, *, config_manager=None):
        return []


def test_parallel_execution_runs_steps_concurrently():
    events: List[Dict[str, Any]] = []

    def _subscriber(payload):
        events.append(payload)

    event_system.subscribe(SKILL_ACTIVITY_EVENT, _subscriber)
    try:
        async def _runner():
            async def first_tool():
                await asyncio.sleep(0.1)
                return "first"

            async def second_tool():
                await asyncio.sleep(0.1)
                return "second"

            tool_manager = _StubToolManager(
                {
                    "first": {"callable": first_tool},
                    "second": {"callable": second_tool},
                }
            )

            context = SkillExecutionContext(
                conversation_id="conv-parallel",
                conversation_history=[],
            )

            metadata = {
                "name": "ParallelSkill",
                "required_tools": ["first", "second"],
                "plan": {
                    "steps": [
                        {"id": "first", "tool": "first"},
                        {"id": "second", "tool": "second"},
                    ]
                },
            }

            started = time.perf_counter()
            result = await use_skill(
                metadata,
                context=context,
                tool_manager=tool_manager,
            )
            elapsed = time.perf_counter() - started
            return result, elapsed

        result, elapsed = asyncio.run(_runner())
        assert result.tool_results == {"first": "first", "second": "second"}
        assert elapsed < 0.18

        plan_events = [entry for entry in events if entry["type"].startswith("skill_plan_")]
        assert any(entry["type"] == "skill_plan_step_succeeded" for entry in plan_events)
    finally:
        event_system.unsubscribe(SKILL_ACTIVITY_EVENT, _subscriber)


def test_failed_gate_cancels_dependents_and_surfaces_reason():
    events: List[Dict[str, Any]] = []

    def _subscriber(payload):
        events.append(payload)

    event_system.subscribe(SKILL_ACTIVITY_EVENT, _subscriber)
    try:
        async def _runner():
            async def gate_tool():
                raise RuntimeError("gate failed")

            async def dependent_tool():
                return "should-not-run"

            tool_manager = _StubToolManager(
                {
                    "gate": {"callable": gate_tool},
                    "dependent": {"callable": dependent_tool},
                }
            )

            context = SkillExecutionContext(
                conversation_id="conv-fail",
                conversation_history=[],
            )

            metadata = {
                "name": "CancellationSkill",
                "required_tools": ["gate", "dependent"],
                "plan": {
                    "steps": [
                        {"id": "gate", "tool": "gate"},
                        {"id": "dependent", "tool": "dependent", "after": ["gate"]},
                    ]
                },
            }

            await use_skill(metadata, context=context, tool_manager=tool_manager)

        with pytest.raises(SkillExecutionError) as excinfo:
            asyncio.run(_runner())

        assert excinfo.value.tool_name == "gate"
        cancelled_events = [
            entry for entry in events if entry["type"] == "skill_plan_step_cancelled"
        ]
        assert cancelled_events
        cancellation_reason = cancelled_events[0].get("cancellation_reason")
        assert cancellation_reason and "gate" in cancellation_reason
    finally:
        event_system.unsubscribe(SKILL_ACTIVITY_EVENT, _subscriber)
