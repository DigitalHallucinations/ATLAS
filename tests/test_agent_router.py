import pytest

from ATLAS.AgentRouter import AgentRouter, RouterDecision


class _StubConfig:
    def __init__(self, tool_defaults=None):
        self._tool_defaults = tool_defaults or {}

    def get_config(self, key, default=None):
        if key == "tool_defaults":
            return self._tool_defaults
        return default


@pytest.fixture
def router():
    return AgentRouter(config_manager=_StubConfig())


def _function_entry(cost, capabilities, unit="credits"):
    return {
        "metadata": {
            "cost_per_call": cost,
            "cost_unit": unit,
            "capabilities": capabilities,
        }
    }


def test_router_prefers_cheapest_tool(router):
    function_map = {
        "expensive_tool": _function_entry(0.8, ["search"]),
        "cheap_tool": _function_entry(0.1, ["search"]),
        "mid_tool": _function_entry(0.2, ["search"]),
        "irrelevant": _function_entry(0.0, ["math"]),
    }

    decision = router.select_tool("search", function_map, session_id="session-1")

    assert isinstance(decision, RouterDecision)
    assert decision.allowed is True
    assert decision.tool_name == "cheap_tool"
    assert abs(dict(decision.metadata)["cost_per_call"] - 0.1) < 1e-9


def test_router_respects_session_budget():
    config = _StubConfig(tool_defaults={"max_cost_per_session": 0.3})
    router = AgentRouter(config_manager=config)
    function_map = {
        "cheap_tool": _function_entry(0.2, ["search"], unit="USD"),
        "also_search": _function_entry(0.25, ["search"], unit="USD"),
    }

    first = router.select_tool("search", function_map, session_id="budgeted")
    assert first.allowed is True
    assert first.tool_name == "cheap_tool"

    second = router.select_tool("search", function_map, session_id="budgeted")
    assert second.allowed is False
    assert second.tool_name is None
    assert "budget" in (second.reason or "").lower()
    assert "usd" in (second.reason or "").lower()
