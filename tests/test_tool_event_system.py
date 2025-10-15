import importlib
import logging
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_tool_manager_dependencies(monkeypatch):
    """Provide minimal dependencies required to import ToolManager."""
    for key in [
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROK_API_KEY",
        "XI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda stream: {}
        monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")

        def _load_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
            return True

        def _set_key(*_args, **_kwargs):  # pragma: no cover - stub helper
            return None

        def _find_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
            return ""

        dotenv_stub.load_dotenv = _load_dotenv
        dotenv_stub.set_key = _set_key
        dotenv_stub.find_dotenv = _find_dotenv
        monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)


@pytest.fixture()
def tool_manager():
    importlib.invalidate_caches()
    module = importlib.import_module("ATLAS.ToolManager")
    return importlib.reload(module)


def test_event_system_handles_subscriber_failures(caplog):
    from modules.Tools.tool_event_system import event_system

    event_name = "test_event_system_failure"
    deliveries = []

    def good_callback(payload):
        deliveries.append(payload)

    def bad_callback(_payload):
        raise RuntimeError("boom")

    try:
        event_system.subscribe(event_name, bad_callback)
        event_system.subscribe(event_name, good_callback)

        with caplog.at_level(logging.ERROR):
            for _ in range(3):
                event_system.publish(event_name, {"value": "data"})

        assert deliveries == [{"value": "data"}] * 3
        assert any(
            "Error while notifying subscriber" in record.getMessage()
            for record in caplog.records
        ), "Subscriber failures should be logged"

        event_system.publish(event_name, {"value": "data"})
        assert deliveries == [{"value": "data"}] * 4
    finally:
        event_system.unsubscribe(event_name, good_callback)
        event_system.unsubscribe(event_name, bad_callback)


def test_record_tool_activity_survives_subscriber_failure(caplog, tool_manager):
    from modules.Tools.tool_event_system import event_system

    initial_len = len(tool_manager.get_tool_activity_log())

    deliveries = []

    def good_callback(entry):
        deliveries.append(entry)

    def bad_callback(_entry):
        raise ValueError("broken subscriber")

    event_name = "tool_activity"

    try:
        event_system.subscribe(event_name, bad_callback)
        event_system.subscribe(event_name, good_callback)

        entry = {"tool": "echo", "arguments": {"value": "ping"}, "result": {"ok": True}}
        with caplog.at_level(logging.ERROR):
            tool_manager._record_tool_activity(entry)  # noqa: SLF001

        assert len(tool_manager.get_tool_activity_log()) == initial_len + 1
        assert deliveries
        assert deliveries[-1]["tool"] == "echo"
        assert any(
            "Error while notifying subscriber" in record.getMessage()
            for record in caplog.records
        ), "Subscriber failure should be logged"
    finally:
        event_system.unsubscribe(event_name, good_callback)
        event_system.unsubscribe(event_name, bad_callback)
