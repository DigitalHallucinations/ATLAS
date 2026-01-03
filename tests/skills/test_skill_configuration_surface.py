import sys
import types
from types import SimpleNamespace


if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")

    def _noop_safe_load(_data):
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
    reason="SQLAlchemy is required for ATLAS skill configuration tests",
)
if getattr(getattr(sqlalchemy_mod, "create_engine", None), "__module__", "").startswith(
    "tests.conftest"
):
    pytest.skip(
        "SQLAlchemy runtime is unavailable for ATLAS skill configuration tests",
        allow_module_level=True,
    )

pytest.importorskip(
    "pytest_postgresql",
    reason="PostgreSQL fixture is required for ATLAS skill configuration tests",
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.conversation_store import ConversationStoreRepository

from ATLAS.ATLAS import ATLAS
from ATLAS.config import ConfigManager
from ATLAS.persona_manager import PersonaManager
from modules.orchestration.capability_registry import reset_capability_registry


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


@pytest.fixture(autouse=True)
def configure_conversation_store(monkeypatch, postgresql):
    dsn = postgresql.dsn()
    monkeypatch.setenv("CONVERSATION_DATABASE_URL", dsn)

    engine = create_engine(dsn, future=True)
    try:
        factory = sessionmaker(bind=engine, future=True)
        ConversationStoreRepository(factory).create_schema()
    finally:
        engine.dispose()


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


def _dummy_skill(name="ExampleSkill", *, persona=None, env_key="EXAMPLE_SKILL_KEY"):
    return SimpleNamespace(
        name=name,
        persona=persona,
        version="1.0.0",
        instruction_prompt="Do the thing",
        required_tools=["some_tool"],
        required_capabilities=["analysis"],
        safety_notes="Be careful",
        summary="Example skill",
        category="Demo",
        capability_tags=["demo"],
        source="skills.json",
        collaboration={"enabled": True},
        auth={"env": env_key},
    )


def test_set_skill_settings_persists_yaml(config_manager_with_temp_yaml):
    manager, yaml_path = config_manager_with_temp_yaml
    settings = {"enabled": True, "default_timeout": 30}

    result = manager.set_skill_settings("ExampleSkill", settings)

    assert result == {"enabled": True, "default_timeout": 30}

    content = yaml_path.read_text(encoding="utf-8")
    assert "ExampleSkill" in content
    assert "enabled" in content
    assert "True" in content
    assert manager.config["skills"]["ExampleSkill"]["enabled"] is True


def test_set_skill_credentials_uses_persist_and_masks(monkeypatch, config_manager_with_temp_yaml):
    manager, _ = config_manager_with_temp_yaml
    calls = []
    _patch_persist_env(monkeypatch, calls)

    manifest_auth = {"env": "EXAMPLE_SKILL_KEY"}
    status = manager.set_skill_credentials(
        "ExampleSkill",
        {"EXAMPLE_SKILL_KEY": "  secret-value  "},
        manifest_auth=manifest_auth,
    )

    assert calls == [("EXAMPLE_SKILL_KEY", "secret-value")]
    assert status["EXAMPLE_SKILL_KEY"]["configured"] is True
    assert status["EXAMPLE_SKILL_KEY"]["hint"] == "••••••••"


def test_skill_snapshot_masks_credentials(config_manager_with_temp_yaml):
    manager, _ = config_manager_with_temp_yaml
    manager.config["EXAMPLE_SKILL_KEY"] = "secret-value"

    snapshot = manager.get_skill_config_snapshot(
        manifest_lookup={"ExampleSkill": {"auth": {"env": "EXAMPLE_SKILL_KEY"}}},
    )

    metadata = snapshot["ExampleSkill"]["credentials"]["EXAMPLE_SKILL_KEY"]
    assert metadata["configured"] is True
    assert metadata["hint"] == "••••••••"


def test_atlas_list_skills_includes_manifest_and_credentials(monkeypatch, atlas_with_temp_config):
    atlas, _ = atlas_with_temp_config
    skill = _dummy_skill()
    atlas.config_manager.config.setdefault("skills", {})["ExampleSkill"] = {"enabled": True}
    atlas.config_manager.config["EXAMPLE_SKILL_KEY"] = "secret-value"

    monkeypatch.setattr(
        "ATLAS.services.tooling.load_skill_metadata",
        lambda **_kwargs: [skill],
        raising=False,
    )

    skills = atlas.list_skills()

    assert skills
    entry = next(item for item in skills if item["name"] == "ExampleSkill")
    assert entry["auth"] == {"env": "EXAMPLE_SKILL_KEY"}
    assert entry["settings"] == {"enabled": True}
    assert entry["credentials"]["EXAMPLE_SKILL_KEY"]["configured"] is True


def test_update_skill_settings_refreshes_cache(monkeypatch, atlas_with_temp_config):
    atlas, yaml_path = atlas_with_temp_config
    skill = _dummy_skill()

    monkeypatch.setattr(
        "ATLAS.services.tooling.load_skill_metadata",
        lambda **_kwargs: [skill],
        raising=False,
    )

    class DummyRegistry:
        def __init__(self):
            self.calls = []

        def refresh(self, *, force=False):
            self.calls.append(force)
            return True

    registry = DummyRegistry()
    monkeypatch.setattr(
        "ATLAS.services.tooling.get_capability_registry",
        lambda **_kwargs: registry,
        raising=False,
    )

    if atlas.persona_manager is None:
        user_identifier, _ = atlas._ensure_user_identity()
        atlas.persona_manager = PersonaManager(
            master=atlas,
            user=user_identifier,
            config_manager=atlas.config_manager,
        )
    manager = atlas.persona_manager
    manager._skill_metadata_cache = (["ExampleSkill"], {"ExampleSkill": {"shared": {}}})

    updated = atlas.update_skill_settings("ExampleSkill", {"enabled": False})

    assert updated == {"enabled": False}
    assert registry.calls and registry.calls[-1] is True
    assert manager._skill_metadata_cache is None

    content = yaml_path.read_text(encoding="utf-8")
    assert "ExampleSkill" in content
    assert "enabled" in content
    assert "False" in content


def test_update_skill_credentials_refreshes_cache(monkeypatch, atlas_with_temp_config):
    atlas, _ = atlas_with_temp_config
    skill = _dummy_skill()

    monkeypatch.setattr(
        "ATLAS.services.tooling.load_skill_metadata",
        lambda **_kwargs: [skill],
        raising=False,
    )

    class DummyRegistry:
        def __init__(self):
            self.calls = []

        def refresh(self, *, force=False):
            self.calls.append(force)
            return True

    registry = DummyRegistry()
    monkeypatch.setattr(
        "ATLAS.services.tooling.get_capability_registry",
        lambda **_kwargs: registry,
        raising=False,
    )

    if atlas.persona_manager is None:
        user_identifier, _ = atlas._ensure_user_identity()
        atlas.persona_manager = PersonaManager(
            master=atlas,
            user=user_identifier,
            config_manager=atlas.config_manager,
        )
    manager = atlas.persona_manager
    manager._skill_metadata_cache = (["ExampleSkill"], {"ExampleSkill": {"shared": {}}})

    calls = []
    _patch_persist_env(monkeypatch, calls)

    status = atlas.update_skill_credentials("ExampleSkill", {"EXAMPLE_SKILL_KEY": " new "})

    assert ("EXAMPLE_SKILL_KEY", "new") in calls
    assert status["EXAMPLE_SKILL_KEY"]["configured"] is True
    assert registry.calls and registry.calls[-1] is True
    assert manager._skill_metadata_cache is None
