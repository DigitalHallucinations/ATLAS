from __future__ import annotations

import json
import shutil
from pathlib import Path

from modules.store_common.package_bundles import (
    export_asset_package_bytes,
    import_asset_package_bytes,
)


class _StubConfigManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _copy_persona_schema(root: Path) -> None:
    schema_src = Path(__file__).resolve().parents[2] / "modules" / "Personas" / "schema.json"
    schema_dest = root / "modules" / "Personas" / "schema.json"
    schema_dest.parent.mkdir(parents=True, exist_ok=True)
    schema_dest.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")


def _setup_capability_metadata(root: Path) -> None:
    tool_manifest = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    tool_payload = [
        {
            "name": "alpha_tool",
            "description": "Alpha",
            "auth": {"required": False},
            "capabilities": ["demo"],
        }
    ]
    _write_json(tool_manifest, tool_payload)

    skill_manifest = root / "modules" / "Skills" / "skills.json"
    skill_payload = [
        {
            "name": "memory_skill",
            "version": "1.0",
            "instruction_prompt": "Remember context.",
            "required_tools": ["alpha_tool"],
            "required_capabilities": ["memory"],
            "safety_notes": "Verify stored data before use.",
            "summary": "Maintain important context for future tasks.",
            "category": "Memory",
            "capability_tags": ["memory"],
            "collaboration": {"enabled": False},
        }
    ]
    _write_json(skill_manifest, skill_payload)


def _create_persona(root: Path, name: str) -> Path:
    persona_payload = {
        "persona": [
            {
                "name": name,
                "meaning": "A helpful assistant.",
                "content": {
                    "start_locked": "Start",
                    "editable_content": "Editable",
                    "end_locked": "End",
                },
                "allowed_tools": ["alpha_tool"],
                "allowed_skills": ["memory_skill"],
            }
        ]
    }

    persona_dir = root / "modules" / "Personas" / name
    persona_file = persona_dir / "Persona" / f"{name}.json"
    _write_json(persona_file, persona_payload)

    memory_dir = persona_dir / "Memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "notes.txt").write_text("Remember context", encoding="utf-8")

    toolbox_dir = persona_dir / "Toolbox"
    toolbox_dir.mkdir(parents=True, exist_ok=True)
    (toolbox_dir / "tool.md").write_text("# Toolbox", encoding="utf-8")

    return persona_dir


def test_persona_asset_package_import_preserves_references(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    _copy_persona_schema(root)
    _setup_capability_metadata(root)
    persona_dir = _create_persona(root, "Explorer")

    bundle_bytes, exported_assets = export_asset_package_bytes(
        [
            {"type": "tool", "name": "alpha_tool"},
            {"type": "skill", "name": "memory_skill"},
            {"type": "persona", "name": "Explorer"},
        ],
        signing_key="secret",
        config_manager=config,
    )

    assert len(exported_assets) == 3
    persona_export_entry = next(
        entry for entry in exported_assets if entry.get("type") == "persona"
    )
    assert persona_export_entry["persona"] == "Explorer"
    assert persona_export_entry["persona_payload"]["name"] == "Explorer"

    tool_manifest = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    skill_manifest = root / "modules" / "Skills" / "skills.json"

    tool_manifest.unlink()
    skill_manifest.unlink()
    shutil.rmtree(persona_dir)

    import_result = import_asset_package_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Regression test",
    )

    assert import_result["success"] is True

    reloaded_tools = json.loads(tool_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "alpha_tool" for entry in reloaded_tools)

    reloaded_skills = json.loads(skill_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "memory_skill" for entry in reloaded_skills)

    persona_file = root / "modules" / "Personas" / "Explorer" / "Persona" / "Explorer.json"
    persona_payload = json.loads(persona_file.read_text(encoding="utf-8"))
    persona_entry = persona_payload["persona"][0]

    assert persona_entry["allowed_tools"] == ["alpha_tool"]
    assert persona_entry["allowed_skills"] == ["memory_skill"]

    persona_results = [entry for entry in import_result["results"] if entry["type"] == "persona"]
    assert persona_results, "Persona import result should be captured"
    assert persona_results[0]["result"].get("warnings") == []
    assert persona_results[0]["persona"] == "Explorer"
    assert persona_results[0]["persona_payload"]["name"] == "Explorer"
