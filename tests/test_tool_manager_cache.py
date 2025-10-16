import concurrent.futures
import json
import os
import shutil
import sys
import uuid
import time
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


@pytest.fixture
def _persona_workspace(monkeypatch):
    persona_name = f"CachePersona_{uuid.uuid4().hex}"
    base_dir = Path("modules") / "Personas" / persona_name / "Toolbox"
    base_dir.mkdir(parents=True, exist_ok=True)

    functions_path = base_dir / "functions.json"
    maps_path = base_dir / "maps.py"

    functions_path.write_text(json.dumps([{"name": "initial", "description": "d"}]))
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

    load_calls = []
    original_json_load = tool_manager.json.load

    def _counting_json_load(file_obj, *args, **kwargs):
        load_calls.append(1)
        return original_json_load(file_obj, *args, **kwargs)

    monkeypatch.setattr(tool_manager.json, "load", _counting_json_load)

    persona = _persona_workspace["persona"]

    first = tool_manager.load_functions_from_json(persona)
    second = tool_manager.load_functions_from_json(persona)

    assert first == second
    assert len(load_calls) == 1


def test_concurrent_loads_share_cached_payload(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    load_calls = []
    original_json_load = tool_manager.json.load

    def _counting_json_load(file_obj, *args, **kwargs):
        load_calls.append(1)
        time.sleep(0.05)
        return original_json_load(file_obj, *args, **kwargs)

    monkeypatch.setattr(tool_manager.json, "load", _counting_json_load)

    persona = _persona_workspace["persona"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(tool_manager.load_functions_from_json, persona)
            for _ in range(5)
        ]
        results = [future.result() for future in futures]

    assert len(load_calls) == 1
    first_result = results[0]
    for result in results:
        assert result == first_result


def test_load_functions_reloads_when_timestamp_changes(monkeypatch, _persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    load_calls = []
    original_json_load = tool_manager.json.load

    def _counting_json_load(file_obj, *args, **kwargs):
        load_calls.append(1)
        return original_json_load(file_obj, *args, **kwargs)

    monkeypatch.setattr(tool_manager.json, "load", _counting_json_load)

    persona = _persona_workspace["persona"]
    functions_path = _persona_workspace["functions_path"]

    first = tool_manager.load_functions_from_json(persona)

    updated_payload = [{"name": "updated", "description": "changed"}]
    functions_path.write_text(json.dumps(updated_payload))
    new_time = os.path.getmtime(functions_path) + 1
    os.utime(functions_path, (new_time, new_time))

    second = tool_manager.load_functions_from_json(persona)

    assert len(load_calls) == 2
    assert second == updated_payload
    assert first != second


def test_refresh_clears_function_payload_cache(_persona_workspace):
    tool_manager._function_payload_cache.clear()
    tool_manager._function_map_cache.clear()

    persona = _persona_workspace["persona"]

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

    tool_manager.load_functions_from_json(persona)
    tool_manager.load_functions_from_json(persona)
    tool_manager.load_function_map_from_current_persona(persona)
    tool_manager.load_function_map_from_current_persona(persona)

    assert instantiations == [1]
