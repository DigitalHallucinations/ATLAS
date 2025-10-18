import concurrent.futures
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

import ATLAS.ToolManager as tool_manager


def _manifest_entry(name, description='d', **overrides):
    entry = {
        'name': name,
        'description': description,
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': [],
        },
        'version': '1.0.0',
        'side_effects': 'none',
        'default_timeout': 30,
        'auth': {'required': False},
        'allow_parallel': True,
    }
    entry.update(overrides)
    return entry


class _StubCapabilityRegistry:
    def __init__(self, payload):
        self.payload = payload
        self.revision = 1
        self.manifest_calls = []
        self.refresh_calls = 0

    def refresh_if_stale(self):
        self.refresh_calls += 1

    def refresh(self, *, force: bool = False) -> None:
        self.refresh_calls += 1

    def get_tool_manifest_payload(self, *, persona, allowed_names=None):
        signature = None if allowed_names is None else tuple(allowed_names)
        self.manifest_calls.append((persona, signature))
        if allowed_names is None:
            return [json.loads(json.dumps(entry)) for entry in self.payload]
        if not allowed_names:
            return []
        selected = []
        for name in allowed_names:
            for entry in self.payload:
                if entry["name"] == name:
                    selected.append(json.loads(json.dumps(entry)))
                    break
        return selected

    def persona_has_tool_manifest(self, persona):
        return persona is not None

    def get_tool_metadata_lookup(self, *, persona, names=None):
        return {name: {} for name in (names or [])}


def _install_registry_stub(monkeypatch, payload):
    stub = _StubCapabilityRegistry(payload)

    def _resolver(*, config_manager=None):
        return stub

    monkeypatch.setattr(tool_manager, "get_capability_registry", _resolver)
    return stub


@pytest.fixture
def _persona_workspace(monkeypatch):
    persona_name = f"CachePersona_{uuid.uuid4().hex}"
    base_dir = Path("modules") / "Personas" / persona_name / "Toolbox"
    base_dir.mkdir(parents=True, exist_ok=True)

    functions_path = base_dir / "functions.json"
    maps_path = base_dir / "maps.py"

    functions_path.write_text(json.dumps([_manifest_entry('initial')]))
    maps_path.write_text(
        "def _sample_tool():\n"
        "    return 'ok'\n\n"
        "function_map = {'sample_tool': _sample_tool}\n"
    )

    module_name = f"persona_{persona_name}_maps"

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    try:
        yield {
            "persona": {"name": persona_name},
            "functions_path": functions_path,
            "module_name": module_name,
        }
    finally:
        tool_manager._function_payload_cache.pop(persona_name, None)
        tool_manager._function_map_cache.pop(persona_name, None)
        sys.modules.pop(module_name, None)
        shutil.rmtree(base_dir.parent, ignore_errors=True)


def test_load_functions_from_json_uses_cached_payload(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    functions_path = _persona_workspace["functions_path"].resolve()
    payload = json.loads(functions_path.read_text())
    registry = _install_registry_stub(monkeypatch, payload)

    persona = _persona_workspace["persona"]

    first = tool_manager.load_functions_from_json(persona)
    second = tool_manager.load_functions_from_json(persona)

    assert first == second
    # Two manifest lookups occur during the first call (validation + selection).
    assert len(registry.manifest_calls) == 2


def test_concurrent_loads_share_cached_payload(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    functions_path = _persona_workspace["functions_path"].resolve()
    payload = json.loads(functions_path.read_text())
    registry = _install_registry_stub(monkeypatch, payload)

    persona = _persona_workspace["persona"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(tool_manager.load_functions_from_json, persona)
            for _ in range(5)
        ]
        results = [future.result() for future in futures]

    assert len(registry.manifest_calls) == 2
    first_result = results[0]
    for result in results:
        assert result == first_result


def test_load_functions_reloads_when_timestamp_changes(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    persona = _persona_workspace["persona"]
    functions_path = _persona_workspace["functions_path"]
    payload = json.loads(functions_path.read_text())
    registry = _install_registry_stub(monkeypatch, payload)

    first = tool_manager.load_functions_from_json(persona)

    updated_payload = [_manifest_entry("updated", description="changed")]
    functions_path.write_text(json.dumps(updated_payload))
    registry.payload = updated_payload
    registry.revision += 1

    second = tool_manager.load_functions_from_json(persona)

    assert second == updated_payload
    assert first != second


def test_refresh_clears_function_payload_cache(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    persona = _persona_workspace["persona"]

    payload = json.loads(_persona_workspace["functions_path"].read_text())
    registry = _install_registry_stub(monkeypatch, payload)

    tool_manager.load_functions_from_json(persona)
    cached_before = tool_manager._function_payload_cache.get(persona["name"])
    assert cached_before is not None

    function_map = tool_manager.load_function_map_from_current_persona(persona, refresh=True)

    assert isinstance(function_map, dict)
    assert "sample_tool" in function_map

    cached_after = tool_manager._function_payload_cache.get(persona["name"])
    assert cached_after is not None
    assert cached_after is not cached_before
    assert cached_after[1] == cached_before[1]


def test_loader_reuses_cached_config_manager(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    persona = _persona_workspace["persona"]

    monkeypatch.setattr(tool_manager, "_default_config_manager", None, raising=False)

    instantiations = []

    class _TrackingConfig:
        def __init__(self):
            instantiations.append(1)

        def get_app_root(self):
            return os.fspath(Path.cwd())

    monkeypatch.setattr(tool_manager, "ConfigManager", _TrackingConfig)

    payload = json.loads(_persona_workspace["functions_path"].read_text())
    _install_registry_stub(monkeypatch, payload)

    tool_manager.load_functions_from_json(persona)
    tool_manager.load_functions_from_json(persona)
    tool_manager.load_function_map_from_current_persona(persona)
    tool_manager.load_function_map_from_current_persona(persona)

    assert instantiations == [1]
