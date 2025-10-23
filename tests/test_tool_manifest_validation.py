import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(
        load_dotenv=lambda *_args, **_kwargs: None,
        set_key=lambda *_args, **_kwargs: None,
        find_dotenv=lambda *_args, **_kwargs: "",
    )

if "jsonschema" in sys.modules and getattr(sys.modules["jsonschema"], "__file__", None) is None:
    sys.modules.pop("jsonschema", None)

jsonschema = importlib.import_module("jsonschema")

import ATLAS.ToolManager as tool_manager

importlib.reload(tool_manager)

tool_manager._tool_manifest_validator = None


def _manifest_entry(name: str, **overrides):
    entry = {
        "name": name,
        "description": "sample description",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "version": "1.0.0",
        "side_effects": "none",
        "default_timeout": 15,
        "auth": {"required": False},
        "allow_parallel": True,
        "cost_per_call": 0.25,
        "cost_unit": "credits",
        "capabilities": ["demo"],
        "idempotency_key": False,
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def persona_workspace(monkeypatch):
    persona_name = f"ValidationPersona_{uuid.uuid4().hex}"
    base_dir = Path("modules") / "Personas" / persona_name / "Toolbox"
    base_dir.mkdir(parents=True, exist_ok=True)

    functions_path = base_dir / "functions.json"
    functions_path.write_text(json.dumps([_manifest_entry("valid")]))

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    try:
        yield {
            "persona": {"name": persona_name},
            "functions_path": functions_path,
        }
    finally:
        tool_manager._function_payload_cache.pop(persona_name, None)
        tool_manager._function_map_cache.pop(persona_name, None)
        shutil.rmtree(base_dir.parent, ignore_errors=True)


def test_invalid_manifest_raises_structured_error(persona_workspace):
    tool_manager._function_payload_cache.clear()

    persona = persona_workspace["persona"]
    functions_path = persona_workspace["functions_path"]

    # Sanity check: valid manifest loads without error.
    loaded = tool_manager.load_functions_from_json(persona, refresh=True)
    assert isinstance(loaded, list)
    assert loaded[0]["name"] == "valid"

    invalid_entry = _manifest_entry("invalid")
    invalid_entry.pop("description")
    functions_path.write_text(json.dumps([invalid_entry]))
    os.utime(functions_path, None)

    with pytest.raises(tool_manager.ToolManifestValidationError) as exc_info:
        tool_manager.load_functions_from_json(persona, refresh=True)

    error = exc_info.value.errors
    assert error.get("persona") == persona["name"]
    assert "description" in error.get("message", "")
    assert exc_info.value.persona == persona["name"]


@pytest.mark.parametrize(
    "side_effect",
    [
        "none",
        "write",
        "network",
        "read_external_service",
        "filesystem",
        "compute",
        "system",
        "database",
    ],
)
def test_manifest_accepts_known_side_effect_categories(persona_workspace, side_effect):
    tool_manager._function_payload_cache.clear()

    persona = persona_workspace["persona"]
    functions_path = persona_workspace["functions_path"]

    entry = _manifest_entry("valid", side_effects=side_effect)
    functions_path.write_text(json.dumps([entry]))
    os.utime(functions_path, None)

    loaded = tool_manager.load_functions_from_json(persona, refresh=True)

    assert isinstance(loaded, list)
    assert loaded[0]["side_effects"] == side_effect


@pytest.mark.parametrize(
    "persona, tool",
    [
        ("ATLAS", "task_catalog_snapshot"),
        ("Cleverbot", "persona_backstory_sampler"),
        ("DocGenius", "generate_doc_outline"),
        ("Einstein", "relativity_scenario"),
        ("Nikola Tesla", "wireless_power_brief"),
        ("genius", "metaphor_palette"),
        ("ComplianceOfficer", "regulatory_gap_audit"),
        ("KnowledgeCurator", "knowledge_card_builder"),
        ("HealthCoach", "habit_stack_planner"),
        ("FitnessCoach", "microcycle_plan"),
        ("LanguageTutor", "dialogue_drill"),
        ("MathTutor", "stepwise_solution"),
    ],
)
def test_specialized_tools_present_in_manifest(persona: str, tool: str) -> None:
    tool_manager._function_payload_cache.pop(persona, None)
    entries = tool_manager.load_functions_from_json({"name": persona}, refresh=True)
    assert any(entry["name"] == tool for entry in entries)
