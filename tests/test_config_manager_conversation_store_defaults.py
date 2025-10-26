from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

DEFAULT_CONVERSATION_DSN = "postgresql+psycopg://atlas:atlas@localhost:5432/atlas"


def _install_dummy_conversation_bootstrap(monkeypatch, bootstrap_func):
    dummy_module = types.ModuleType("modules.conversation_store.bootstrap")

    class DummyBootstrapError(RuntimeError):
        pass

    dummy_module.BootstrapError = DummyBootstrapError
    dummy_module.bootstrap_conversation_store = bootstrap_func

    monkeypatch.setitem(
        sys.modules,
        "modules.conversation_store.bootstrap",
        dummy_module,
    )

    parent_module = sys.modules.get("modules.conversation_store")
    if parent_module is None:
        parent_module = types.ModuleType("modules.conversation_store")
        monkeypatch.setitem(sys.modules, "modules.conversation_store", parent_module)

    setattr(parent_module, "bootstrap", dummy_module)

    return DummyBootstrapError


@pytest.fixture
def config_manager(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    monkeypatch.setenv("OPENAI_API_KEY", "initial-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")

    def stub_module(name: str, **attrs):
        module = types.ModuleType(name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub_module(
        "modules.orchestration.message_bus",
        InMemoryQueueBackend=type("InMemoryQueueBackend", (), {}),
        MessageBus=type("MessageBus", (), {}),
        RedisStreamBackend=type("RedisStreamBackend", (), {}),
        configure_message_bus=lambda *args, **kwargs: None,
    )
    stub_module(
        "modules.job_store",
        JobService=type("JobService", (), {}),
    )
    stub_module(
        "modules.job_store.repository",
        JobStoreRepository=type("JobStoreRepository", (), {}),
    )
    stub_module(
        "modules.task_store",
        TaskService=type("TaskService", (), {}),
        TaskStoreRepository=type("TaskStoreRepository", (), {}),
    )

    base_tools_pkg = types.ModuleType("modules.Tools.Base_Tools")
    base_tools_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modules.Tools.Base_Tools", base_tools_pkg)

    stub_module(
        "modules.Tools.Base_Tools.task_queue",
        TaskQueueService=type("TaskQueueService", (), {}),
        get_default_task_queue_service=lambda *args, **kwargs: None,
    )
    stub_module(
        "modules.Tools.Base_Tools.vector_store",
        QueryMatch=type("QueryMatch", (), {}),
        VectorRecord=type("VectorRecord", (), {}),
        VectorStoreService=type("VectorStoreService", (), {}),
    )
    stub_module(
        "modules.Tools.Base_Tools.browser_lite",
        BrowserLite=type("BrowserLite", (), {}),
        BrowserLiteError=type("BrowserLiteError", (Exception,), {}),
        DomainNotAllowlistedError=type("DomainNotAllowlistedError", (Exception,), {}),
        FormSubmissionNotAllowedError=type("FormSubmissionNotAllowedError", (Exception,), {}),
        NavigationFailedError=type("NavigationFailedError", (Exception,), {}),
        NavigationLimitError=type("NavigationLimitError", (Exception,), {}),
        PersonaPolicyViolationError=type("PersonaPolicyViolationError", (Exception,), {}),
        RobotsBlockedError=type("RobotsBlockedError", (Exception,), {}),
    )
    stub_module(
        "modules.Tools.Base_Tools.Google_search",
        GoogleSearch=type("GoogleSearch", (), {}),
    )

    monkeypatch.delitem(sys.modules, "ATLAS.config", raising=False)

    config_module = importlib.import_module("ATLAS.config")
    ConfigManager = config_module.ConfigManager

    recorded = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_module, "set_key", fake_set_key)
    monkeypatch.setattr(config_module, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ConfigManager,
        "_load_env_config",
        lambda self: {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "DEFAULT_PROVIDER": os.getenv("DEFAULT_PROVIDER", "OpenAI"),
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4o"),
            "MISTRAL_API_KEY": None,
            "HUGGINGFACE_API_KEY": None,
            "GOOGLE_API_KEY": None,
            "ANTHROPIC_API_KEY": None,
            "GROK_API_KEY": None,
            "APP_ROOT": tmp_path.as_posix(),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_ORGANIZATION": os.getenv("OPENAI_ORGANIZATION"),
        },
    )
    monkeypatch.setattr(ConfigManager, "_load_yaml_config", lambda self: {})

    manager = ConfigManager()
    manager._recorded_set_key = recorded
    manager._env_path = str(env_file)

    yield manager

    sys.modules.pop("ATLAS.config", None)


def test_ensure_postgres_conversation_store_defaults_and_persists(config_manager, monkeypatch):
    writes = []

    monkeypatch.setattr(
        config_manager,
        "_write_yaml_config",
        lambda: writes.append(
            dict(config_manager.yaml_config.get("conversation_database", {}))
        ),
    )

    def fake_bootstrap(url, **kwargs):
        return url

    _install_dummy_conversation_bootstrap(monkeypatch, fake_bootstrap)

    result = config_manager.ensure_postgres_conversation_store()

    assert result == DEFAULT_CONVERSATION_DSN
    assert config_manager.config["conversation_database"]["url"] == DEFAULT_CONVERSATION_DSN
    assert config_manager.yaml_config["conversation_database"]["url"] == DEFAULT_CONVERSATION_DSN
    assert writes


def test_ensure_postgres_conversation_store_updates_persisted_url(config_manager, monkeypatch):
    initial_block = {
        "url": "postgresql+psycopg://existing:pass@db.example.com:5432/atlas",
        "pool": {"size": 5},
    }
    config_manager.config["conversation_database"] = dict(initial_block)
    config_manager.yaml_config["conversation_database"] = dict(initial_block)

    writes = []

    monkeypatch.setattr(
        config_manager,
        "_write_yaml_config",
        lambda: writes.append(
            dict(config_manager.yaml_config.get("conversation_database", {}))
        ),
    )

    updated_url = (
        "postgresql+psycopg://existing:newpass@db.example.com:5432/atlas"
    )

    def fake_bootstrap(url, **kwargs):
        assert url == initial_block["url"]
        return updated_url

    _install_dummy_conversation_bootstrap(monkeypatch, fake_bootstrap)

    result = config_manager.ensure_postgres_conversation_store()

    assert result == updated_url
    assert config_manager.config["conversation_database"]["url"] == updated_url
    assert config_manager.yaml_config["conversation_database"]["url"] == updated_url
    assert config_manager.yaml_config["conversation_database"]["pool"] == {"size": 5}
    assert writes
