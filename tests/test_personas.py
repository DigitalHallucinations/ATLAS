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

if "jsonschema" not in sys.modules:  # pragma: no cover - lightweight stub for validators
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Validator:
        def __init__(self, _schema=None):
            self.schema = _schema

        def iter_errors(self, _payload):  # pragma: no cover - simple stub
            return []

    class _SchemaError(Exception):
        pass

    jsonschema_stub.Draft202012Validator = _Validator
    jsonschema_stub.Draft7Validator = _Validator
    jsonschema_stub.ValidationError = _SchemaError
    jsonschema_stub.exceptions = types.SimpleNamespace(SchemaError=_SchemaError)
    sys.modules["jsonschema"] = jsonschema_stub

from modules.Personas import (
    build_skill_state,
    build_tool_state,
    load_persona_definition,
    load_skill_catalog,
    load_tool_metadata,
    normalize_allowed_skills,
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

    shared_skills = [
        {
            "name": "shared_skill",
            "version": "1.0",
            "instruction_prompt": "Shared skill instructions.",
            "required_tools": ["alpha_tool"],
            "required_capabilities": ["analysis"],
            "safety_notes": "Review output for accuracy.",
        }
    ]
    _write_json(root / "modules" / "Skills" / "skills.json", shared_skills)

    specialist_skills = [
        {
            "name": "specialist_insight",
            "version": "1.0",
            "instruction_prompt": "Specialist-only insight generation.",
            "required_tools": ["beta_tool"],
            "required_capabilities": ["expertise"],
            "safety_notes": "Use responsibly.",
        },
        {
            "name": "shared_skill",
            "version": "2.0",
            "instruction_prompt": "Specialist-tuned shared skill.",
            "required_tools": ["beta_tool"],
            "required_capabilities": ["analysis"],
            "safety_notes": "Override for Specialist persona.",
        },
    ]
    _write_json(
        root / "modules" / "Personas" / "Specialist" / "Skills" / "skills.json",
        specialist_skills,
    )

    schema_src = Path(__file__).resolve().parents[1] / "modules" / "Personas" / "schema.json"
    schema_dest = root / "modules" / "Personas" / "schema.json"
    schema_dest.parent.mkdir(parents=True, exist_ok=True)
    schema_dest.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    return config, root


def test_load_persona_definition_defaults_to_no_tools(persona_fixture: tuple[_StubConfigManager, Path]) -> None:
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
    assert persona["allowed_tools"] == []
    assert persona["allowed_skills"] == []


def test_build_tool_state_surfaces_catalog_disabled_by_default(
    persona_fixture: tuple[_StubConfigManager, Path]
) -> None:
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

    order, lookup = load_tool_metadata(config_manager=config)
    persona = load_persona_definition("Atlas", config_manager=config, metadata_order=order)
    tool_state = build_tool_state(
        persona,
        config_manager=config,
        metadata_order=order,
        metadata_lookup=lookup,
    )

    assert tool_state["allowed"] == []
    assert {entry["name"] for entry in tool_state["available"]} == set(order)
    assert all(entry["enabled"] is False for entry in tool_state["available"])


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
                "allowed_skills": ["specialist_insight"],
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


def test_load_skill_catalog_includes_persona_variants(
    persona_fixture: tuple[_StubConfigManager, Path]
) -> None:
    config, _root = persona_fixture

    order, lookup = load_skill_catalog(config_manager=config)

    assert "shared_skill" in order
    catalog_entry = lookup["shared_skill"]

    shared_metadata = catalog_entry.get("shared")
    assert shared_metadata is not None
    assert shared_metadata["instruction_prompt"] == "Shared skill instructions."

    persona_variants = catalog_entry.get("persona_variants") or {}
    assert "specialist" in persona_variants
    assert (
        persona_variants["specialist"]["instruction_prompt"]
        == "Specialist-tuned shared skill."
    )


def test_build_skill_state_handles_persona_restrictions(persona_fixture: tuple[_StubConfigManager, Path]) -> None:
    config, root = persona_fixture

    persona_payload = {
        "persona": [
            {
                "name": "Atlas",
                "meaning": "",
                "content": {
                    "start_locked": "",
                    "editable_content": "",
                    "end_locked": "",
                },
            }
        ]
    }
    atlas_file = root / "modules" / "Personas" / "Atlas" / "Persona" / "Atlas.json"
    _write_json(atlas_file, persona_payload)

    tool_order, tool_lookup = load_tool_metadata(config_manager=config)
    persona = load_persona_definition(
        "Atlas",
        config_manager=config,
        metadata_order=tool_order,
        metadata_lookup=tool_lookup,
    )

    skill_order, skill_lookup = load_skill_catalog(config_manager=config)
    skill_state = build_skill_state(
        persona,
        config_manager=config,
        metadata_order=skill_order,
        metadata_lookup=skill_lookup,
    )

    assert skill_state["allowed"] == []
    available = skill_state["available"]

    shared_entries = [
        entry
        for entry in available
        if entry["name"] == "shared_skill" and entry["metadata"].get("persona") in (None, "")
    ]
    assert shared_entries
    assert shared_entries[0].get("disabled") in (False, None)
    assert shared_entries[0]["enabled"] is False
    assert (
        shared_entries[0]["metadata"]["instruction_prompt"]
        == "Shared skill instructions."
    )

    specialist_override_entries = [
        entry
        for entry in available
        if entry["name"] == "shared_skill" and entry["metadata"].get("persona") == "Specialist"
    ]
    assert specialist_override_entries
    assert specialist_override_entries[0].get("disabled", False) is True
    assert (
        specialist_override_entries[0]["metadata"]["instruction_prompt"]
        == "Specialist-tuned shared skill."
    )

    specialist_entries = [
        entry for entry in available if entry["name"] == "specialist_insight"
    ]
    assert specialist_entries
    assert specialist_entries[0].get("disabled", False) is True


def test_build_skill_state_enables_owned_skills(persona_fixture: tuple[_StubConfigManager, Path]) -> None:
    config, root = persona_fixture

    specialist_payload = {
        "persona": [
            {
                "name": "Specialist",
                "meaning": "",
                "content": {
                    "start_locked": "",
                    "editable_content": "",
                    "end_locked": "",
                },
                "allowed_skills": ["specialist_insight"],
            }
        ]
    }
    specialist_file = root / "modules" / "Personas" / "Specialist" / "Persona" / "Specialist.json"
    _write_json(specialist_file, specialist_payload)

    persona = load_persona_definition("Specialist", config_manager=config)
    assert persona is not None

    skill_order, skill_lookup = load_skill_catalog(config_manager=config)
    skill_state = build_skill_state(
        persona,
        config_manager=config,
        metadata_order=skill_order,
        metadata_lookup=skill_lookup,
    )

    assert normalize_allowed_skills(["specialist_insight"], metadata_order=skill_order) == [
        "specialist_insight"
    ]
    available = skill_state["available"]

    specialist_entries = [
        entry
        for entry in available
        if entry["name"] == "specialist_insight"
    ]
    assert specialist_entries
    assert specialist_entries[0]["enabled"] is True
    assert specialist_entries[0].get("disabled") in (False, None)

    override_entries = [
        entry
        for entry in available
        if entry["name"] == "shared_skill" and entry["metadata"].get("persona") == "Specialist"
    ]
    assert override_entries
    assert override_entries[0]["enabled"] is False  # not explicitly allowed
    assert override_entries[0].get("disabled") in (False, None)
    assert (
        override_entries[0]["metadata"]["instruction_prompt"]
        == "Specialist-tuned shared skill."
    )

    shared_entries = [
        entry
        for entry in available
        if entry["name"] == "shared_skill" and entry["metadata"].get("persona") in (None, "")
    ]
    assert shared_entries
    assert shared_entries[0].get("disabled") is True
    assert "override" in shared_entries[0].get("disabled_reason", "").lower()
    assert (
        shared_entries[0]["metadata"]["instruction_prompt"]
        == "Shared skill instructions."
    )
