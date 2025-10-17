from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
import types

if "numpy" not in sys.modules:  # pragma: no cover - lightweight stub for pytest helpers
    numpy_stub = types.ModuleType("numpy")

    def _isscalar(obj):
        return isinstance(obj, (int, float, complex, bool, str))

    numpy_stub.isscalar = _isscalar
    sys.modules["numpy"] = numpy_stub

from modules.Personas import (
    build_tool_state,
    load_persona_definition,
    load_tool_metadata,
)


class _StubConfigManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


@pytest.fixture
def persona_fixture(tmp_path: Path) -> tuple[_StubConfigManager, Path]:
    root = tmp_path
    config = _StubConfigManager(root)

    tools_payload = [
        {
            "name": "alpha_tool",
            "description": "Alpha",
            "safety_level": "low",
            "cost_per_call": 0.1,
            "cost_unit": "USD",
        },
        {
            "name": "beta_tool",
            "description": "Beta",
            "safety_level": "medium",
            "cost_per_call": 0.0,
            "cost_unit": "USD",
        },
    ]
    _write_json(root / "modules" / "Tools" / "tool_maps" / "functions.json", tools_payload)

    return config, root


def test_load_persona_definition_populates_allowed_tools(persona_fixture: tuple[_StubConfigManager, Path]) -> None:
    config, root = persona_fixture
    persona_payload = {
        "persona": [
            {
                "name": "Atlas",
                "meaning": "",
                "content": {
                    "start_locked": "start",
                    "editable_content": "body",
                    "end_locked": "end",
                },
            }
        ]
    }
    persona_file = root / "modules" / "Personas" / "Atlas" / "Persona" / "Atlas.json"
    _write_json(persona_file, persona_payload)

    order, _lookup = load_tool_metadata(config_manager=config)
    persona = load_persona_definition("Atlas", config_manager=config, metadata_order=order)

    assert persona is not None
    assert persona["allowed_tools"] == order


def test_build_tool_state_merges_overrides(persona_fixture: tuple[_StubConfigManager, Path]) -> None:
    config, root = persona_fixture

    # extend metadata for reuse
    extra_payload = {
        "persona": [
            {
                "name": "Specialist",
                "meaning": "",
                "content": {
                    "start_locked": "",
                    "editable_content": "",
                    "end_locked": "",
                },
                "allowed_tools": ["beta_tool", "custom_tool"],
            }
        ]
    }
    persona_file = root / "modules" / "Personas" / "Specialist" / "Persona" / "Specialist.json"
    _write_json(persona_file, extra_payload)

    overrides_payload = [
        {
            "name": "custom_tool",
            "description": "Custom override",
            "safety_level": "high",
            "cost_per_call": 1.25,
            "cost_unit": "USD",
        }
    ]
    _write_json(
        root / "modules" / "Personas" / "Specialist" / "Toolbox" / "functions.json",
        overrides_payload,
    )

    order, lookup = load_tool_metadata(config_manager=config)
    persona = load_persona_definition("Specialist", config_manager=config, metadata_order=order)
    tool_state = build_tool_state(
        persona,
        config_manager=config,
        metadata_order=order,
        metadata_lookup=lookup,
    )

    available = tool_state["available"]
    names = [entry["name"] for entry in available]

    assert names[:2] == ["beta_tool", "custom_tool"]
    assert any(entry["name"] == "alpha_tool" for entry in available)

    lookup_entries = {entry["name"]: entry for entry in available}
    assert lookup_entries["beta_tool"]["enabled"] is True
    assert lookup_entries["alpha_tool"]["enabled"] is False

    custom_metadata = lookup_entries["custom_tool"]["metadata"]
    assert custom_metadata["description"] == "Custom override"
    assert custom_metadata["safety_level"] == "high"
    assert pytest.approx(custom_metadata["cost_per_call"], rel=0.01) == 1.25
