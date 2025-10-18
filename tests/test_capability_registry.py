import json
from pathlib import Path

import pytest

from modules.orchestration.capability_registry import CapabilityRegistry

import json
from pathlib import Path

import pytest

from modules.orchestration.capability_registry import CapabilityRegistry


class _DummyConfig:
    def __init__(self, root: Path) -> None:
        self._root = root

    def get_app_root(self) -> str:
        return str(self._root)


@pytest.fixture()
def capability_root(tmp_path: Path) -> Path:
    (tmp_path / "modules" / "Tools" / "tool_maps").mkdir(parents=True, exist_ok=True)
    (tmp_path / "modules" / "Skills").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_shared_manifests(root: Path, tools: list[dict], skills: list[dict]) -> None:
    tool_path = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    skill_path = root / "modules" / "Skills" / "skills.json"
    tool_path.write_text(json.dumps(tools), encoding="utf-8")
    skill_path.write_text(json.dumps(skills), encoding="utf-8")


def _minimal_tool(name: str) -> dict:
    return {
        "name": name,
        "description": "sample",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "version": "1.0.0",
        "side_effects": "none",
        "default_timeout": 30,
        "auth": {"required": False},
        "allow_parallel": True,
        "capabilities": ["sample"],
        "providers": [{"name": "primary"}],
    }


def _minimal_skill(name: str) -> dict:
    return {
        "name": name,
        "version": "1.0.0",
        "instruction_prompt": "run",
        "required_tools": [],
        "required_capabilities": ["sample"],
        "safety_notes": "",
        "summary": "",
        "category": "general",
        "capability_tags": ["sample"],
    }


def test_registry_filters_invalid_entries(capability_root: Path) -> None:
    tools = [_minimal_tool("valid_tool"), {"description": "missing name"}]
    skills = [_minimal_skill("valid_skill")]
    _write_shared_manifests(capability_root, tools, skills)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    assert registry.refresh(force=True)

    tools = registry.query_tools()
    assert len(tools) == 1
    assert tools[0].manifest.name == "valid_tool"


def test_registry_records_tool_metrics(capability_root: Path) -> None:
    tools = [_minimal_tool("measured_tool")]
    skills = [_minimal_skill("skill")]
    _write_shared_manifests(capability_root, tools, skills)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    registry.record_tool_execution(
        persona=None,
        tool_name="measured_tool",
        success=True,
        latency_ms=100,
    )
    registry.record_tool_execution(
        persona=None,
        tool_name="measured_tool",
        success=False,
        latency_ms=200,
    )

    view = registry.query_tools()[0]
    tool_health = view.health["tool"]
    assert tool_health["total"] == 2
    assert pytest.approx(tool_health["success_rate"], rel=1e-6) == 0.5
    assert pytest.approx(tool_health["average_latency_ms"], rel=1e-6) == 150.0


def test_registry_records_provider_metrics(capability_root: Path) -> None:
    tools = [_minimal_tool("provider_tool")]
    skills = [_minimal_skill("skill")]
    _write_shared_manifests(capability_root, tools, skills)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    summary = {
        "tool": "provider_tool",
        "selected": "primary",
        "success": True,
        "providers": {
            "primary": {
                "successes": 1,
                "failures": 0,
                "consecutive_failures": 0,
                "failure_rate": 0.0,
                "last_success": 123.0,
                "last_failure": None,
                "last_check": 123.0,
                "backoff_until": None,
            }
        },
    }
    registry.record_provider_metrics(persona=None, tool_name="provider_tool", summary=summary)

    view = registry.query_tools()[0]
    provider_health = view.health["providers"]["primary"]["metrics"]
    assert provider_health["total"] == 1
    assert provider_health["success"] == 1
    assert provider_health["failure"] == 0
