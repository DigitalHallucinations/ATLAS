import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


@pytest.fixture
def persona_cli_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    # Schema is at project_root/modules/Personas/schema.json
    project_root = Path(__file__).resolve().parents[2]
    schema_src = project_root / "modules" / "Personas" / "schema.json"
    schema_dst = root / "modules" / "Personas" / "schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    tool_manifest = [
        {
            "name": "alpha_tool",
            "description": "Alpha",
        }
    ]
    _write_json(root / "modules" / "Tools" / "tool_maps" / "functions.json", tool_manifest)

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
                "allowed_tools": ["alpha_tool"],
            }
        ]
    }
    _write_json(root / "modules" / "Personas" / "Atlas" / "Persona" / "Atlas.json", persona_payload)

    return root


def test_cli_export_import_round_trip(persona_cli_fixture: Path, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]  # Go up to project root
    bundle_path = tmp_path / "bundle.json"
    key_path = tmp_path / "signing.key"
    key_path.write_text("super-secret", encoding="utf-8")

    export_cmd = [
        sys.executable,
        "scripts/persona_tools.py",
        "--app-root",
        str(persona_cli_fixture),
        "export",
        "Atlas",
        str(bundle_path),
        "--signing-key-file",
        str(key_path),
    ]
    # Ensure PYTHONPATH includes repo root for module imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    export_proc = subprocess.run(
        export_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert export_proc.returncode == 0, export_proc.stderr
    assert bundle_path.exists()

    persona_file = persona_cli_fixture / "modules" / "Personas" / "Atlas" / "Persona" / "Atlas.json"
    persona_file.unlink()

    # Remove the original tool to trigger a missing tool warning on import
    replacement_manifest = [
        {
            "name": "beta_tool",
            "description": "Beta",
        }
    ]
    _write_json(persona_cli_fixture / "modules" / "Tools" / "tool_maps" / "functions.json", replacement_manifest)

    import_cmd = [
        sys.executable,
        "scripts/persona_tools.py",
        "--app-root",
        str(persona_cli_fixture),
        "import",
        str(bundle_path),
        "--signing-key-file",
        str(key_path),
        "--rationale",
        "CLI test",
    ]
    import_proc = subprocess.run(
        import_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert import_proc.returncode == 0, import_proc.stderr
    assert "WARNING: Missing tools pruned" in import_proc.stdout

    imported_payload = json.loads(persona_file.read_text(encoding="utf-8"))
    persona_entry = imported_payload["persona"][0]
    assert persona_entry["allowed_tools"] == []
