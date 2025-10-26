import sys
import types
from typing import Any, Dict

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")

    def _noop_safe_load(*_args, **_kwargs):
        return {}

    def _simple_dump(data, stream=None, **_kwargs):
        text = str(data)
        if stream is not None:
            stream.write(text)
        return text

    yaml_module.safe_load = _noop_safe_load
    yaml_module.safe_dump = _simple_dump
    yaml_module.dump = _simple_dump
    sys.modules["yaml"] = yaml_module

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_module.set_key = lambda *_args, **_kwargs: None
    dotenv_module.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_module

import pytest

sqlalchemy_mod = pytest.importorskip(
    "sqlalchemy",
    reason="SQLAlchemy is required for ATLAS tool configuration tests",
)
if getattr(getattr(sqlalchemy_mod, "create_engine", None), "__module__", "").startswith(
    "tests.conftest"
):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for ATLAS tool configuration tests",
        allow_module_level=True,
    )

pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for ATLAS tool configuration tests",
)

from ATLAS.ATLAS import ATLAS
from ATLAS.config import ConfigManager
from modules.orchestration.capability_registry import reset_capability_registry
from ATLAS import ToolManager as ToolManagerModule


@pytest.fixture(autouse=True)
def configure_conversation_store(monkeypatch, postgresql):
    monkeypatch.setenv("CONVERSATION_DATABASE_URL", postgresql.dsn())


@pytest.fixture
def config_manager_with_temp_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("{}", encoding="utf-8")

    def fake_compute(self):
        return str(yaml_path)

    monkeypatch.setattr(ConfigManager, "_compute_yaml_path", fake_compute, raising=False)
    manager = ConfigManager()
    manager._yaml_path = str(yaml_path)
    return manager, yaml_path


@pytest.fixture
def atlas_with_temp_config(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("{}", encoding="utf-8")

    def fake_compute(self):
        return str(yaml_path)

    monkeypatch.setattr(ConfigManager, "_compute_yaml_path", fake_compute, raising=False)
    reset_capability_registry()
    atlas = ATLAS()
    atlas.config_manager._yaml_path = str(yaml_path)
    try:
        yield atlas, yaml_path
    finally:
        reset_capability_registry()


def _patch_persist_env(monkeypatch, calls):
    def fake_persist(self, env_key, value):
        calls.append((env_key, value))
        if value is None:
            self.env_config.pop(env_key, None)
            self.config.pop(env_key, None)
        else:
            self.env_config[env_key] = value
            self.config[env_key] = value
        self._sync_provider_warning(env_key, value)

    monkeypatch.setattr(ConfigManager, "_persist_env_value", fake_persist, raising=False)


def test_set_tool_settings_persists_yaml(config_manager_with_temp_yaml):
    manager, yaml_path = config_manager_with_temp_yaml
    settings = {"timeout_seconds": 45, "enabled": True, "providers": ["primary"]}

    result = manager.set_tool_settings("example_tool", settings)

    assert result == {
        "timeout_seconds": 45,
        "enabled": True,
        "providers": ["primary"],
    }

    content = yaml_path.read_text(encoding="utf-8")
    assert "example_tool" in content
    assert "timeout_seconds" in content
    assert "45" in content
    assert manager.config["tools"]["example_tool"]["enabled"] is True


def test_set_tool_credentials_uses_persist_and_masks(monkeypatch, config_manager_with_temp_yaml):
    manager, _ = config_manager_with_temp_yaml
    calls = []
    _patch_persist_env(monkeypatch, calls)

    manifest_auth = {"env": "EXAMPLE_API_KEY"}
    status = manager.set_tool_credentials(
        "example_tool",
        {"EXAMPLE_API_KEY": "  supersecret-token  "},
        manifest_auth=manifest_auth,
    )

    assert calls == [("EXAMPLE_API_KEY", "supersecret-token")]
    assert status["EXAMPLE_API_KEY"]["configured"] is True
    assert status["EXAMPLE_API_KEY"]["hint"] == "••••••••"


def test_tool_snapshot_masks_credentials(config_manager_with_temp_yaml):
    manager, _ = config_manager_with_temp_yaml
    manager.config["EXAMPLE_API_KEY"] = "supersecret-token"

    snapshot = manager.get_tool_config_snapshot(
        manifest_lookup={"example_tool": {"auth": {"env": "EXAMPLE_API_KEY"}}}
    )

    metadata = snapshot["example_tool"]["credentials"]["EXAMPLE_API_KEY"]
    assert metadata["configured"] is True
    assert metadata["hint"] == "••••••••"


def test_atlas_list_tools_includes_manifest_and_credentials(atlas_with_temp_config):
    atlas, _ = atlas_with_temp_config
    tools = atlas.list_tools()

    google_entry = next(tool for tool in tools if tool["name"] == "google_search" and tool["persona"] is None)
    assert google_entry["auth"]["required"] is True
    assert google_entry["credentials"]["GOOGLE_API_KEY"]["configured"] is False


def test_update_tool_settings_refreshes_cache(monkeypatch, atlas_with_temp_config):
    atlas, yaml_path = atlas_with_temp_config
    calls: Dict[str, Dict[str, Any]] = {}

    def fake_load_default_function_map(*, refresh=False, config_manager=None):
        calls["refresh"] = {"refresh": refresh, "manager": config_manager}
        return {}

    monkeypatch.setattr(
        ToolManagerModule,
        "load_default_function_map",
        fake_load_default_function_map,
        raising=False,
    )

    updated = atlas.update_tool_settings("google_search", {"default_timeout": 15})
    assert updated == {"default_timeout": 15}
    assert calls["refresh"]["refresh"] is True

    content = yaml_path.read_text(encoding="utf-8")
    assert "google_search" in content
    assert "default_timeout" in content
    assert "15" in content


def test_update_tool_credentials_refreshes_cache(monkeypatch, atlas_with_temp_config):
    atlas, _ = atlas_with_temp_config
    calls = []
    _patch_persist_env(monkeypatch, calls)

    load_calls = []

    def fake_load_default_function_map(*, refresh=False, config_manager=None):
        load_calls.append({"refresh": refresh, "manager": config_manager})
        return {}

    monkeypatch.setattr(
        ToolManagerModule,
        "load_default_function_map",
        fake_load_default_function_map,
        raising=False,
    )

    status = atlas.update_tool_credentials("google_search", {"GOOGLE_API_KEY": " new-value "})

    assert ("GOOGLE_API_KEY", "new-value") in calls
    assert status["GOOGLE_API_KEY"]["configured"] is True
    assert load_calls and load_calls[-1]["refresh"] is True
