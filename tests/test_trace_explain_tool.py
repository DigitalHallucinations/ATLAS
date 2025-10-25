import importlib
from datetime import datetime, timezone


def test_trace_explain_merges_logs_and_traces(monkeypatch):
    module = importlib.import_module("modules.Tools.Base_Tools.trace_explain")

    base_time = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp()

    tool_entries = [
        {
            "tool_name": "search",
            "completed_at": "2024-05-01T00:10:00+00:00",
            "conversation_id": "conv-1",
        },
        {
            "tool_name": "notebook",
            "completed_at": "2024-05-01T02:00:00+00:00",
            "conversation_id": "conv-2",
        },
    ]

    def fake_activity_log(limit=None):
        assert limit is None or isinstance(limit, int)
        return [dict(entry) for entry in tool_entries]

    monkeypatch.setattr(
        module.ToolManagerModule,
        "get_tool_activity_log",
        fake_activity_log,
        raising=False,
    )

    negotiation_traces = [
        {
            "id": "trace-1",
            "completed_at": base_time + 300,
            "status": "success",
        },
        {
            "id": "trace-2",
            "completed_at": base_time + 5400,
            "status": "timeout",
        },
    ]

    class StubSession:
        conversation_id = "conv-1"

        def get_negotiation_history(self):
            return [dict(entry) for entry in negotiation_traces]

    context = {"chat_session": StubSession()}

    result = module.trace_explain(
        conversation_id="conv-1",
        trace_id="trace-1",
        start_at=datetime.fromtimestamp(base_time - 60, tz=timezone.utc).isoformat(),
        end_at=datetime.fromtimestamp(base_time + 3600, tz=timezone.utc).isoformat(),
        limit=5,
        context=context,
    )

    assert result["filters"]["conversation_id"] == "conv-1"
    assert result["filters"]["trace_id"] == "trace-1"
    assert result["counts"] == {"tool_activity": 1, "negotiation_traces": 1}
    assert result["pagination"]["total"] == 2

    events = result["events"]
    assert [event["type"] for event in events] == ["negotiation_trace", "tool_activity"]
    assert events[0]["data"]["id"] == "trace-1"
    assert events[0]["conversation_id"] == "conv-1"
    assert events[1]["data"]["tool_name"] == "search"
    assert events[1]["conversation_id"] == "conv-1"

    assert result["tool_activity"] == [events[1]["data"]]
    assert result["negotiation_traces"] == [events[0]["data"]]


def test_trace_explain_available_in_default_map(monkeypatch):
    aiohttp_mod = importlib.import_module("aiohttp")

    class _ClientTimeout:
        def __init__(self, *_, **__):
            pass

    monkeypatch.setattr(
        aiohttp_mod,
        "ClientTimeout",
        getattr(aiohttp_mod, "ClientTimeout", _ClientTimeout),
        raising=False,
    )

    maps_module = importlib.import_module("modules.Tools.tool_maps.maps")
    maps_module = importlib.reload(maps_module)

    assert "trace.explain" in maps_module.function_map
    assert maps_module.function_map["trace.explain"] is not None
