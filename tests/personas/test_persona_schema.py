from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from modules.Personas import (
    PersonaValidationError,
    load_persona_definition,
    load_tool_metadata,
)


def _persona_directories(root: Path) -> Iterable[Path]:
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "__pycache__":
            continue
        if not (entry / "Persona").exists():
            continue
        yield entry


def test_persona_definitions_validate() -> None:
    personas_root = Path("modules/Personas").resolve()
    order, lookup = load_tool_metadata()

    directories = list(_persona_directories(personas_root))
    assert directories, "Expected at least one persona directory"

    for directory in directories:
        persona_name = directory.name
        persona = load_persona_definition(
            persona_name,
            metadata_order=order,
            metadata_lookup=lookup,
        )
        assert persona is not None, f"Persona '{persona_name}' should load successfully"


def test_echo_persona_includes_mediation_stack() -> None:
    order, lookup = load_tool_metadata()
    persona = load_persona_definition(
        "Echo",
        metadata_order=order,
        metadata_lookup=lookup,
    )

    allowed_tools = set(persona["allowed_tools"])
    expected = {"context_tracker", "get_current_info", "tone_analyzer", "reflective_prompt", "memory_recall", "conflict_resolver"}
    assert expected.issubset(allowed_tools)

    skills = persona.get("allowed_skills", [])
    assert "EchoMediationCycle" in skills


class _StubConfigManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def test_invalid_persona_rejected(tmp_path: Path) -> None:
    config = _StubConfigManager(tmp_path)

    schema_src = Path("modules/Personas/schema.json").resolve()
    schema_dest = tmp_path / "modules" / "Personas" / "schema.json"
    schema_dest.parent.mkdir(parents=True, exist_ok=True)
    schema_dest.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    tools_payload = [
        {
            "name": "alpha_tool",
            "description": "Alpha",
        }
    ]
    tools_path = tmp_path / "modules" / "Tools" / "tool_maps" / "functions.json"
    tools_path.parent.mkdir(parents=True, exist_ok=True)
    tools_path.write_text(json.dumps(tools_payload, indent=2), encoding="utf-8")

    invalid_payload = {
        "persona": [
            {
                "name": "Invalid",
                "meaning": "",
                "content": {
                    "start_locked": "",
                    "editable_content": "",
                    "end_locked": "",
                },
                "allowed_tools": ["missing_tool"],
            }
        ]
    }
    persona_path = tmp_path / "modules" / "Personas" / "Invalid" / "Persona" / "Invalid.json"
    persona_path.parent.mkdir(parents=True, exist_ok=True)
    persona_path.write_text(json.dumps(invalid_payload, indent=2), encoding="utf-8")

    order, lookup = load_tool_metadata(config_manager=config)

    with pytest.raises(PersonaValidationError) as excinfo:
        load_persona_definition(
            "Invalid",
            config_manager=config,
            metadata_order=order,
            metadata_lookup=lookup,
        )

    assert "allowed_tools" in str(excinfo.value)
