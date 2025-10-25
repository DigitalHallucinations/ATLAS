import importlib
import json
import sys
import types
from pathlib import Path


def _ensure_aiohttp_stub():
    module = sys.modules.get("aiohttp")
    if module is not None and hasattr(module, "ClientTimeout"):
        return

    class _FakeClientTimeout:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total") if kwargs else (args[0] if args else None)

    class _FakeClientSession:
        def __init__(self, *args, **kwargs):
            return

        async def __aenter__(self):  # pragma: no cover - defensive stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - defensive stub
            return False

    sys.modules["aiohttp"] = types.SimpleNamespace(
        ClientTimeout=_FakeClientTimeout,
        ClientSession=_FakeClientSession,
        ClientError=Exception,
    )


_ensure_aiohttp_stub()


def _load_module():
    return importlib.import_module("modules.Tools.Base_Tools.eval_regression")


def test_eval_regression_computes_diff_and_emits_event(monkeypatch):
    module = _load_module()

    recorded: dict[str, object] = {}

    def fake_publish(event_name, payload, **kwargs):
        recorded["event"] = event_name
        recorded["payload"] = payload
        recorded["kwargs"] = kwargs
        return "corr-123"

    monkeypatch.setattr(module, "publish_bus_event", fake_publish)

    store = {
        "baseline-1": {
            "id": "baseline-1",
            "content": "alpha\nbeta\n",
        }
    }
    candidate = {"id": "candidate-1", "content": "alpha\ngamma\n"}

    result = module.eval_regression(
        baseline="baseline-1",
        candidate=candidate,
        artifact_store=store,
        metrics={"thresholds": {"min_similarity": 0.65, "max_total_changes": 2}},
        diff={"context": 1},
        metadata={"task_id": "task-007"},
    )

    assert recorded["event"] == "atlas.evaluation.regression"
    payload = recorded["payload"]
    assert payload["baseline"]["id"] == "baseline-1"
    assert payload["candidate"]["id"] == "candidate-1"

    diff_text = payload["diff"]["text"]
    assert "-beta" in diff_text
    assert "+gamma" in diff_text

    metrics = payload["metrics"]
    assert metrics["added_lines"] == 1
    assert metrics["removed_lines"] == 1
    assert metrics["total_changes"] == 2

    thresholds = payload["summary"]["thresholds"]
    assert thresholds["min_similarity"]["expected"] == 0.65
    assert thresholds["min_similarity"]["passed"] is True
    assert thresholds["max_total_changes"]["expected"] == 2
    assert thresholds["max_total_changes"]["passed"] is True

    assert result["correlation_id"] == "corr-123"
    assert result["metrics"] == metrics
    assert result["diff"]["text"] == diff_text
    assert result["summary"]["regressed"] is False


def test_eval_regression_thresholds_flag_regressions(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(module, "publish_bus_event", lambda *args, **kwargs: "corr-999")

    baseline = {"id": "base", "content": "line1\nline2\n"}
    candidate = {"id": "cand", "content": "line1\nline3\n"}

    result = module.eval_regression(
        baseline=baseline,
        candidate=candidate,
        metrics={"thresholds": {"min_similarity": 0.99, "max_total_changes": 1}},
    )

    summary = result["summary"]
    assert summary["regressed"] is True

    thresholds = summary["thresholds"]
    assert thresholds["min_similarity"]["passed"] is False
    assert thresholds["max_total_changes"]["passed"] is False


def test_eval_regression_manifest_entry_exposed():
    manifest_path = Path(__file__).resolve().parents[1] / "modules/Tools/tool_maps/functions.json"
    manifest = json.loads(manifest_path.read_text())

    entry = next((item for item in manifest if item["name"] == "eval.regression"), None)
    assert entry is not None

    params = entry["parameters"]
    assert "baseline" in params["properties"]
    assert "candidate" in params["properties"]
    assert set(params["required"]) == {"baseline", "candidate"}

    returns = entry["returns"]
    assert "correlation_id" in returns["required"]
    assert "metrics" in returns["properties"]


def test_eval_regression_available_via_tool_manager(monkeypatch):
    import ATLAS.ToolManager as tool_manager

    shared_map = tool_manager.load_default_function_map(refresh=True)
    assert "eval.regression" in shared_map

    entry = shared_map["eval.regression"]
    assert isinstance(entry, dict)
    assert callable(entry.get("callable"))
    metadata = entry.get("metadata")
    assert metadata and metadata.get("version") == "1.0.0"
    assert "evaluation" in metadata.get("capabilities", [])
