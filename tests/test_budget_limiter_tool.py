import asyncio

import pytest

from modules.Tools.Base_Tools.budget_limiter import BudgetLimiterTool, BudgetLimiterError
from modules.orchestration import budget_tracker


class _DummyConfigManager:
    def __init__(self, *, budget_ms=100.0):
        self._budget_ms = budget_ms

    def get_config(self, key, default=None):
        if key == "conversation":
            return {"max_tool_duration_ms": self._budget_ms}
        return default


def test_budget_limiter_enforces_budget_cycle():
    async def run_test():
        tool = BudgetLimiterTool(config_manager=_DummyConfigManager(budget_ms=100))
        await budget_tracker.reset_runtime()

        snapshot = await tool.run(operation="inspect", conversation_id="thread")
        assert snapshot["budget_ms"] == 100
        assert snapshot["consumed_ms"] == 0

        first = await tool.run(operation="reserve", conversation_id="thread", duration_ms=60)
        assert first["accepted"] is True
        assert pytest.approx(first["consumed_ms"], rel=1e-3) == 60
        assert first["budget_exceeded"] is False

        second = await tool.run(operation="reserve", conversation_id="thread", duration_ms=50)
        assert second["accepted"] is True
        assert pytest.approx(second["consumed_ms"], rel=1e-3) == 110
        assert second["budget_exceeded"] is True

        third = await tool.run(operation="reserve", conversation_id="thread", duration_ms=10)
        assert third["accepted"] is False
        assert third["reason"] == "budget_exhausted"
        assert pytest.approx(third["consumed_ms"], rel=1e-3) == 110

        release = await tool.run(operation="release", conversation_id="thread", duration_ms=25)
        assert release["accepted"] is True
        assert pytest.approx(release["consumed_ms"], rel=1e-3) == 85
        assert release["budget_exceeded"] is False

        reset = await tool.run(operation="reset", conversation_id="thread")
        assert reset["consumed_ms"] == 0
        assert reset["budget_exceeded"] is False

        await budget_tracker.reset_runtime()

    asyncio.run(run_test())


def test_budget_limiter_global_reset_and_validation():
    async def run_test():
        tool = BudgetLimiterTool(config_manager=_DummyConfigManager(budget_ms=0))
        await budget_tracker.reset_runtime()

        global_view = await tool.run(operation="inspect")
        assert global_view["snapshot"] == {}
        assert global_view["budget_ms"] is None

        with pytest.raises(BudgetLimiterError):
            await tool.run(operation="reserve", duration_ms=10)

        await tool.run(operation="reserve", conversation_id="chat", duration_ms=15)

        global_view = await tool.run(operation="inspect")
        assert pytest.approx(global_view["snapshot"]["chat"], rel=1e-3) == 15

        cleared = await tool.run(operation="reset")
        assert cleared["snapshot"] == {}

        await budget_tracker.reset_runtime()

    asyncio.run(run_test())
