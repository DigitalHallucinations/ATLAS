import importlib
import json
import sys
import types
from pathlib import Path


aiohttp_module = sys.modules.get("aiohttp")
if aiohttp_module is None or not hasattr(aiohttp_module, "ClientTimeout"):
    class _FakeClientTimeout:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total") if kwargs else (args[0] if args else None)

    class _FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):  # pragma: no cover - defensive stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - defensive stub
            return False

    sys.modules["aiohttp"] = types.SimpleNamespace(
        ClientTimeout=_FakeClientTimeout,
        ClientSession=_FakeClientSession,
        ClientError=Exception,
    )


def test_eval_judge_runs_evaluators_and_publishes_event(monkeypatch):
    module = importlib.import_module("modules.Tools.Base_Tools.eval_judge")

    recorded: dict[str, object] = {}

    def fake_publish(event_name, payload, **kwargs):
        recorded["event"] = event_name
        recorded["payload"] = payload
        recorded["kwargs"] = kwargs
        return "corr-001"

    monkeypatch.setattr(module, "publish_bus_event", fake_publish)

    calls: list[tuple[str, dict[str, object]]] = []

    def evaluator_exact(**kwargs):
        calls.append(("exact", kwargs))
        return {"score": 1.0, "details": {"matches": 1.0}}

    def evaluator_llm(**kwargs):
        calls.append(("llm", kwargs))
        return {"score": 0.7, "details": {"explanation": "ok"}, "explanation": "ok"}

    original_exact = module._EVALUATOR_REGISTRY.get("exact")
    original_llm = module._EVALUATOR_REGISTRY.get("llm")
    module._EVALUATOR_REGISTRY["exact"] = evaluator_exact
    module._EVALUATOR_REGISTRY["llm"] = evaluator_llm

    analytics_calls: list[dict[str, object]] = []

    def analytics_hook(payload):
        analytics_calls.append(payload)

    try:
        result = module.eval_judge(
            candidates=["alpha"],
            references=["alpha"],
            rubrics=["accuracy"],
            evaluators=["exact", {"name": "llm", "options": {"model": "mock"}}],
            thresholds={"default": 0.6, "per_evaluator": {"exact": 0.9}},
            metadata={"conversation_id": "conv-1"},
            analytics=[analytics_hook],
        )
    finally:
        if original_exact is None:
            module._EVALUATOR_REGISTRY.pop("exact", None)
        else:
            module._EVALUATOR_REGISTRY["exact"] = original_exact
        if original_llm is None:
            module._EVALUATOR_REGISTRY.pop("llm", None)
        else:
            module._EVALUATOR_REGISTRY["llm"] = original_llm

    assert [name for name, _ in calls] == ["exact", "llm"]
    first_call_kwargs = calls[0][1]
    assert first_call_kwargs["candidates"][0]["output"] == "alpha"
    assert first_call_kwargs["references"][0]["output"] == "alpha"
    assert first_call_kwargs["rubrics"][0]["description"] == "accuracy"

    assert recorded["event"] == "atlas.evaluation.judged"
    payload = recorded["payload"]
    assert payload["passed"] is True
    assert payload["summary"]["passed"] == 2
    assert payload["results"][0]["threshold"] == 0.9
    assert payload["results"][1]["threshold"] == 0.6
    assert result["correlation_id"] == "corr-001"
    assert analytics_calls and analytics_calls[0]["summary"]["passed"] == 2


def test_eval_judge_manifest_entry_exposed():
    manifest_path = Path(__file__).resolve().parents[1] / "modules/Tools/tool_maps/functions.json"
    manifest = json.loads(manifest_path.read_text())

    entry = next((item for item in manifest if item["name"] == "eval.judge"), None)
    assert entry is not None

    params = entry["parameters"]
    assert set(params["required"]) == {"candidates", "evaluators"}
    assert "thresholds" in params["properties"]
    assert "evaluators" in params["properties"]
    assert "threshold" in entry["description"].lower()

    returns = entry["returns"]
    assert "correlation_id" in returns["properties"]


def test_eval_judge_function_map_exposes_tool():
    module = importlib.import_module("modules.Tools.Base_Tools.eval_judge")
    from modules.Tools.tool_maps.maps import function_map

    assert function_map["eval.judge"] is module.eval_judge
