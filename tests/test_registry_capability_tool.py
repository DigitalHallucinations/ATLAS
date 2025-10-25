from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.Tools.Base_Tools.registry_capability import (
    registry_capability as capability_tool,
)


class _StubRegistry:
    def __init__(self) -> None:
        self.revision = 3
        self.refresh_calls: list[bool] = []
        self.summary_calls: list[str | None] = []
        self.tool_calls: list[dict[str, object]] = []

    def refresh(self, *, force: bool = False) -> bool:
        self.refresh_calls.append(force)
        return bool(force)

    def summary(self, *, persona: str | None = None):
        self.summary_calls.append(persona)
        return {"revision": self.revision, "persona": persona, "tools": []}

    def query_tools(
        self,
        *,
        persona_filters=None,
        capability_filters=None,
        provider_filters=None,
        version_constraint=None,
        min_success_rate=None,
    ):
        self.tool_calls.append(
            {
                "persona_filters": persona_filters,
                "capability_filters": capability_filters,
                "provider_filters": provider_filters,
                "version_constraint": version_constraint,
                "min_success_rate": min_success_rate,
            }
        )

        manifest = SimpleNamespace(
            name="alpha",
            persona=None,
            description="Demo tool",
            version="1.0.0",
            capabilities=["demo"],
            auth={"required": False},
            safety_level=None,
            requires_consent=None,
            allow_parallel=None,
            idempotency_key=None,
            default_timeout=None,
            side_effects=None,
            cost_per_call=None,
            cost_unit=None,
            persona_allowlist=[],
            requires_flags={},
            providers=[],
            source="tests",
        )
        view = SimpleNamespace(
            manifest=manifest,
            capability_tags=("demo",),
            auth_scopes=("scope",),
            health={"tool": {"total": 1}},
        )
        return [view]

    def query_skills(self, **_kwargs):  # pragma: no cover - unused in tests
        return []

    def query_tasks(self, **_kwargs):  # pragma: no cover - unused in tests
        return []

    def query_jobs(self, **_kwargs):  # pragma: no cover - unused in tests
        return []


def _install_stub(monkeypatch: pytest.MonkeyPatch, stub: _StubRegistry) -> None:
    module = importlib.import_module("modules.Tools.Base_Tools.registry_capability")
    monkeypatch.setattr(
        module, "get_capability_registry", lambda config_manager=None: stub
    )


def test_registry_capability_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubRegistry()
    _install_stub(monkeypatch, stub)

    payload = capability_tool(persona="Atlas", capability_types=["summary"])

    assert stub.summary_calls == ["Atlas"]
    assert payload["summary"]["persona"] == "Atlas"
    assert payload["revision"] == stub.revision


def test_registry_capability_filters_and_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubRegistry()
    _install_stub(monkeypatch, stub)

    payload = capability_tool(
        persona="Atlas",
        persona_filters=["atlas", "-shared"],
        capability_types=["tools"],
        capability_filters=["demo"],
        provider_filters=["primary"],
        version_constraint=">=1.0",
        min_success_rate=0.75,
        refresh=True,
    )

    assert stub.refresh_calls == [False]
    assert stub.tool_calls
    tool_call = stub.tool_calls[0]
    assert tool_call["persona_filters"] == ["atlas", "-shared", "Atlas"]
    assert tool_call["capability_filters"] == ["demo"]
    assert tool_call["provider_filters"] == ["primary"]
    assert tool_call["version_constraint"] == ">=1.0"
    assert tool_call["min_success_rate"] == 0.75

    assert payload["tools"]
    tool_entry = payload["tools"][0]
    assert tool_entry["manifest"]["name"] == "alpha"
    assert tool_entry["health"]["tool"]["total"] == 1


def test_registry_capability_force_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubRegistry()
    _install_stub(monkeypatch, stub)

    payload = capability_tool(force_refresh=True)

    assert stub.refresh_calls == [True]
    assert payload["refreshed"] is True


def test_registry_capability_visible_via_tool_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    if "yaml" not in sys.modules:
        yaml_stub = SimpleNamespace(
            safe_load=lambda *_args, **_kwargs: {},
            dump=lambda *_args, **_kwargs: None,
        )
        sys.modules["yaml"] = yaml_stub

    if "dotenv" not in sys.modules:
        dotenv_stub = SimpleNamespace(
            load_dotenv=lambda *_args, **_kwargs: None,
            set_key=lambda *_args, **_kwargs: None,
            find_dotenv=lambda *_args, **_kwargs: "",
        )
        sys.modules["dotenv"] = dotenv_stub

    if "aiohttp" not in sys.modules:
        aiohttp_stub = SimpleNamespace()
        sys.modules["aiohttp"] = aiohttp_stub
    else:
        aiohttp_stub = sys.modules["aiohttp"]

    class _ClientTimeout:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    class _ClientSession:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    class _ClientError(Exception):
        pass

    class _ClientResponse:  # pragma: no cover - simple stub
        pass

    monkeypatch.setattr(aiohttp_stub, "ClientTimeout", _ClientTimeout, raising=False)
    monkeypatch.setattr(aiohttp_stub, "ClientSession", _ClientSession, raising=False)
    monkeypatch.setattr(aiohttp_stub, "ClientError", _ClientError, raising=False)
    monkeypatch.setattr(aiohttp_stub, "ClientResponse", _ClientResponse, raising=False)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )
    tool_manager._default_function_map_cache = None

    function_map = tool_manager.load_default_function_map(refresh=True)

    assert "registry.capability" in function_map
