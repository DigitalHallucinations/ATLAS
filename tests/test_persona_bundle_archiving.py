import json
import shutil
import sys
import types
from pathlib import Path

from modules.Personas import export_persona_bundle_bytes, import_persona_bundle_bytes


if "jsonschema" not in sys.modules:  # pragma: no cover - avoid optional dependency requirements
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Validator:
        def __init__(self, _schema=None):
            self.schema = _schema

        def iter_errors(self, _payload):  # pragma: no cover - simple stub
            return []

    class _SchemaError(Exception):
        pass

    jsonschema_stub.Draft202012Validator = _Validator
    jsonschema_stub.ValidationError = _SchemaError
    jsonschema_stub.exceptions = types.SimpleNamespace(SchemaError=_SchemaError)
    sys.modules["jsonschema"] = jsonschema_stub


class _StubConfigManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def _copy_persona_schema(root: Path) -> None:
    schema_src = Path(__file__).resolve().parents[1] / "modules" / "Personas" / "schema.json"
    schema_dest = root / "modules" / "Personas" / "schema.json"
    schema_dest.parent.mkdir(parents=True, exist_ok=True)
    schema_dest.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")


def _setup_metadata(root: Path) -> None:
    tools_payload = [
        {
            "name": "alpha_tool",
            "description": "Alpha",
            "safety_level": "low",
            "cost_per_call": 0.0,
            "cost_unit": "USD",
        }
    ]
    _write_json(root / "modules" / "Tools" / "tool_maps" / "functions.json", tools_payload)

    skills_payload = [
        {
            "name": "memory_skill",
            "version": "1.0",
            "instruction_prompt": "Remember important context.",
            "required_tools": ["alpha_tool"],
            "required_capabilities": ["memory"],
            "safety_notes": "Verify stored data before use.",
        }
    ]
    _write_json(root / "modules" / "Skills" / "skills.json", skills_payload)


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
    (memory_dir / "notes.txt").write_text("Remember to review context", encoding="utf-8")

    config_dir = persona_dir / "Config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text("mode: exploratory", encoding="utf-8")

    toolbox_dir = persona_dir / "Toolbox"
    toolbox_dir.mkdir(parents=True, exist_ok=True)
    (toolbox_dir / "tool.md").write_text("# Toolbox", encoding="utf-8")

    return persona_dir


def test_persona_bundle_roundtrip_restores_directory(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    _copy_persona_schema(root)
    _setup_metadata(root)
    persona_dir = _create_persona(root, "Explorer")

    bundle_bytes, exported_persona = export_persona_bundle_bytes(
        "Explorer",
        signing_key="secret",
        config_manager=config,
    )

    payload = json.loads(bundle_bytes.decode("utf-8"))
    archive_block = payload.get("archive")
    assert isinstance(archive_block, dict)
    assert archive_block.get("format") == "tar.gz"
    assert archive_block.get("encoding") == "base64"
    assert archive_block.get("data")

    shutil.rmtree(persona_dir)

    result = import_persona_bundle_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test import",
    )

    restored_dir = root / "modules" / "Personas" / "Explorer"
    memory_file = restored_dir / "Memory" / "notes.txt"
    config_file = restored_dir / "Config" / "settings.yaml"
    toolbox_file = restored_dir / "Toolbox" / "tool.md"

    assert memory_file.read_text(encoding="utf-8") == "Remember to review context"
    assert config_file.read_text(encoding="utf-8") == "mode: exploratory"
    assert toolbox_file.read_text(encoding="utf-8") == "# Toolbox"

    persisted_payload = json.loads(
        (restored_dir / "Persona" / "Explorer.json").read_text(encoding="utf-8")
    )
    persisted_persona = persisted_payload["persona"][0]

    assert persisted_persona["allowed_tools"] == ["alpha_tool"]
    assert persisted_persona["allowed_skills"] == ["memory_skill"]
    assert exported_persona["allowed_tools"] == ["alpha_tool"]
    assert result["persona"]["allowed_tools"] == ["alpha_tool"]
    assert result["success"] is True
