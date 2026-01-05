import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

log_event_module = importlib.import_module("modules.Tools.Base_Tools.log_event")
from core.messaging import MessagePriority


def test_log_event_publishes_to_message_bus_and_logger(monkeypatch):
    recorded: dict[str, object] = {}

    log_calls: list[tuple[int, str]] = []
    original_log = logging.Logger.log

    def capture_log(self, level, msg, *args, **kwargs):
        if self.name == "atlas.events":
            text = msg % args if args else msg
            log_calls.append((level, text))
        return original_log(self, level, msg, *args, **kwargs)

    def fake_publish(event_name, payload, **kwargs):
        recorded["event_name"] = event_name
        recorded["payload"] = payload
        recorded["kwargs"] = kwargs
        return "abc123"

    monkeypatch.setattr(logging.Logger, "log", capture_log)
    monkeypatch.setattr(log_event_module, "publish_bus_event", fake_publish)

    result = log_event_module.log_event(
        event_name="atlas.observability.test",
        severity="info",
        payload={"message": "ok", "value": 1},
        correlation_id="corr-1",
        parent_correlation_id="parent-1",
        logger="atlas.events",
        persistence={"metadata": {"persist": True}},
    )

    assert recorded["event_name"] == "atlas.observability.test"
    bus_payload = recorded["payload"]
    assert isinstance(bus_payload, dict)
    assert bus_payload["severity"] == "info"
    assert bus_payload["payload"] == {"message": "ok", "value": 1}
    assert bus_payload["parent_correlation_id"] == "parent-1"

    kwargs = recorded["kwargs"]
    assert kwargs["correlation_id"] == "corr-1"
    assert kwargs["priority"] == MessagePriority.NORMAL
    assert kwargs["metadata"] == {"persist": True}

    assert result["correlation_id"] == "abc123"
    assert any("atlas.observability.test" in entry for _level, entry in log_calls)


def test_log_event_redacts_sensitive_fields_and_validates(monkeypatch):
    captured: dict[str, object] = {}

    def fake_publish(event_name, payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return "cid"

    monkeypatch.setattr(log_event_module, "publish_bus_event", fake_publish)

    log_event_module.log_event(
        event_name="atlas.observability.secret",
        severity="error",
        payload={
            "password": "hunter2",
            "nested": {"token": "value", "safe": "ok"},
            "items": [{"apiKey": "secret"}],
        },
        persistence={"priority": "high", "emit_legacy": False},
    )

    sanitized = captured["payload"]["payload"]
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert sanitized["nested"]["safe"] == "ok"
    assert sanitized["items"][0]["apiKey"] == "[REDACTED]"

    kwargs = captured["kwargs"]
    assert kwargs["priority"] == MessagePriority.HIGH
    assert kwargs["emit_legacy"] is False

    with pytest.raises(TypeError):
        log_event_module.log_event(
            event_name="atlas.observability.invalid",
            severity="info",
            payload={"invalid": {1, 2, 3}},
        )


def test_log_event_manifest_entry_exposed():
    manifest_path = Path(__file__).resolve().parents[1] / "modules/Tools/tool_maps/functions.json"
    manifest = json.loads(manifest_path.read_text())

    entry = next((item for item in manifest if item["name"] == "log.event"), None)
    assert entry is not None, "log.event manifest entry should be published"

    assert entry["side_effects"] == "logging"
    schema = entry["parameters"]
    assert set(schema["required"]) == {"event_name", "severity"}

    properties = schema["properties"]
    assert "payload" in properties
    assert "persistence" in properties
    assert properties["persistence"]["additionalProperties"] is False
