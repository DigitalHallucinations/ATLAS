"""Tests for the ToolingService helper class."""

from __future__ import annotations

import logging
import types
from typing import Any, Dict

import pytest

from ATLAS.services.tooling import ToolingService


class DummyConfigManager:
    def __init__(self) -> None:
        self._tool_settings: Dict[str, Dict[str, Any]] = {}
        self.snapshot: Dict[str, Dict[str, Any]] = {}

    def set_tool_settings(self, name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(settings)
        self._tool_settings[name] = payload
        return payload

    def get_tool_config_snapshot(self, *, manifest_lookup=None) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self.snapshot.items()}

    def get_job_repository(self):  # pragma: no cover - not used in focused tests
        return None

    def get_default_task_queue_service(self):  # pragma: no cover - not used
        return None

    def get_job_service(self):  # pragma: no cover - not used
        return None

    def set_tool_credentials(self, *args, **kwargs):  # pragma: no cover - helper stub
        raise AssertionError("Unexpected call in focused tests")

    def set_skill_settings(self, *args, **kwargs):  # pragma: no cover - helper stub
        raise AssertionError("Unexpected call in focused tests")

    def set_skill_credentials(self, *args, **kwargs):  # pragma: no cover - helper stub
        raise AssertionError("Unexpected call in focused tests")

    def is_job_scheduling_enabled(self):  # pragma: no cover - helper stub
        return True


@pytest.fixture
def tooling_service(monkeypatch):
    config_manager = DummyConfigManager()

    calls: Dict[str, Any] = {}

    class DummyRegistry:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def refresh(self, *, force=False):
            self.calls.append(force)
            return True

        def query_tools(self):
            return []

    registry = DummyRegistry()
    monkeypatch.setattr(
        "ATLAS.services.tooling.get_capability_registry",
        lambda **_kwargs: registry,
    )

    tool_manager_module = types.SimpleNamespace(
        load_default_function_map=lambda **_kwargs: {},
        load_function_map_from_current_persona=lambda *_args, **_kwargs: {},
        _resolve_function_callable=lambda entry: entry,
    )

    service = ToolingService(
        config_manager=config_manager,
        tool_manager_module=tool_manager_module,
        persona_manager=None,
        message_bus=None,
        logger=logging.getLogger(__name__),
        tenant_id="tenant",
    )

    calls["registry"] = registry
    calls["config_manager"] = config_manager
    calls["tool_manager_module"] = tool_manager_module

    return service, calls


def test_update_tool_settings_triggers_cache_refresh(monkeypatch, tooling_service):
    service, calls = tooling_service
    config_manager: DummyConfigManager = calls["config_manager"]

    load_calls = []

    def fake_load_default_function_map(*, refresh=False, config_manager=None):
        load_calls.append({"refresh": refresh, "manager": config_manager})
        return {}

    monkeypatch.setattr(
        calls["tool_manager_module"],
        "load_default_function_map",
        fake_load_default_function_map,
    )

    registry = calls["registry"]

    updated = service.update_tool_settings("example", {"enabled": True})

    assert updated == {"enabled": True}
    assert registry.calls and registry.calls[-1] is True
    assert load_calls and load_calls[-1]["refresh"] is True
    assert load_calls[-1]["manager"] is config_manager


def test_list_tools_serializes_health_and_snapshot(monkeypatch, tooling_service):
    service, calls = tooling_service
    config_manager: DummyConfigManager = calls["config_manager"]

    class DummyManifest:
        name = "example"
        persona = None
        description = "Example tool"
        version = "1.0.0"
        capabilities = ["execute"]
        auth = {"required": True}
        auth_required = True
        safety_level = "low"
        requires_consent = False
        allow_parallel = True
        idempotency_key = None
        default_timeout = 30
        side_effects = None
        cost_per_call = None
        cost_unit = None
        persona_allowlist = None
        providers = {}
        source = "test"

    class DummyView:
        manifest = DummyManifest()
        capability_tags = ["alpha"]
        auth_scopes = ["scope:read"]
        health = {
            "tool": {"status": "ok"},
            "providers": {
                "primary": {
                    "metrics": {"latency": 1},
                    "router": {"requests": 5},
                }
            },
        }

    class DummyRegistry:
        def refresh(self, *, force=False):  # pragma: no cover - not triggered here
            return True

        def query_tools(self):
            return [DummyView()]

    monkeypatch.setattr(
        "ATLAS.services.tooling.get_capability_registry",
        lambda **_kwargs: DummyRegistry(),
    )

    config_manager.snapshot = {
        "example": {
            "settings": {"enabled": True},
            "credentials": {"TOKEN": {"configured": True, "hint": "••••"}},
        }
    }

    tools = service.list_tools()

    assert len(tools) == 1
    entry = tools[0]
    assert entry["name"] == "example"
    assert entry["health"]["tool"] == {"status": "ok"}
    assert entry["health"]["providers"]["primary"]["metrics"] == {"latency": 1}
    assert entry["settings"]["enabled"] is True
    assert entry["credentials"]["TOKEN"]["hint"] == "••••"
    assert entry["auth"]["required"] is True
