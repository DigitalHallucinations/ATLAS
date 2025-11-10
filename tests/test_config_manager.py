import json
import math
import os
import sys
import types
import logging
from typing import Any, Callable, Dict, Optional

import pytest


class _StubLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args, **kwargs):
        if args:
            try:
                message = message % args
            except Exception:
                message = str(message)
        self.warnings.append(str(message))

    def info(self, *args, **kwargs):  # pragma: no cover - no-op logging helpers
        return None

    def error(self, *args, **kwargs):  # pragma: no cover - no-op logging helpers
        return None

    def debug(self, *args, **kwargs):  # pragma: no cover - no-op logging helpers
        return None




try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback only used in limited environments
    import json as _json

    yaml = types.ModuleType("yaml")

    def _safe_load_fallback(stream):
        if hasattr(stream, "read"):
            content = stream.read()
        else:
            content = stream or ""

        content = content or ""
        if not content.strip():
            return {}

        try:
            return _json.loads(content)
        except _json.JSONDecodeError:
            return {}

    def _dump_fallback(data, stream=None, **_kwargs):
        text = _json.dumps(data or {})
        if stream is None:
            return text
        stream.write(text)
        return text

    yaml.safe_load = _safe_load_fallback
    yaml.dump = _dump_fallback
    sys.modules["yaml"] = yaml

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *args, **kwargs: None
    dotenv_module.set_key = lambda *args, **kwargs: None
    dotenv_module.find_dotenv = lambda *args, **kwargs: ""
    dotenv_module.dotenv_values = lambda *args, **kwargs: {}
    sys.modules["dotenv"] = dotenv_module

import ATLAS.config as config_module
import ATLAS.config.config_manager as config_impl
from ATLAS.config import ConfigManager
from ATLAS.config.messaging import MessagingConfigSection
from ATLAS.config.persistence import PersistenceConfigSection
from ATLAS.config.tooling import ToolingConfigSection


class _DummyURL:
    def __init__(self, value: str, drivername: str = "postgresql") -> None:
        self._value = value
        self.drivername = drivername

    def set(self, **kwargs):
        return _DummyURL(self._value, kwargs.get("drivername", self.drivername))

    def render_as_string(self, hide_password: bool = False) -> str:
        return self._value
def _make_persistence_section(
    *,
    config: Optional[dict] = None,
    yaml_config: Optional[dict] = None,
    env: Optional[dict] = None,
    create_engine: Optional[Callable[..., Any]] = None,
    inspect_engine: Optional[Callable[..., Any]] = None,
    make_url: Optional[Callable[[str], Any]] = None,
    sessionmaker_factory: Optional[Callable[..., Any]] = None,
    conversation_required_tables: Optional[Callable[[], set[str]]] = None,
) -> PersistenceConfigSection:
    writes: list[bool] = []

    def _write_yaml_callback() -> None:
        writes.append(True)

    section = PersistenceConfigSection(
        config=config or {},
        yaml_config=yaml_config or {},
        env_config=env or {},
        logger=_StubLogger(),
        normalize_job_store_url=lambda value, source: str(value),
        write_yaml_callback=_write_yaml_callback,
        create_engine=create_engine
        or (lambda *args, **kwargs: types.SimpleNamespace(dispose=lambda: None)),
        inspect_engine=inspect_engine
        or (lambda *_args, **_kwargs: types.SimpleNamespace(get_table_names=lambda: [])),
        make_url=make_url or (lambda value: _DummyURL(str(value))),
        sessionmaker_factory=sessionmaker_factory
        or (lambda **kwargs: ("factory", kwargs)),
        conversation_required_tables=conversation_required_tables or (lambda: set()),
        default_conversation_dsn="postgresql://default",
    )
    section._write_calls = writes  # type: ignore[attr-defined]
    return section


@pytest.fixture
def config_manager(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("")

    monkeypatch.setenv("OPENAI_API_KEY", "initial-key")
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")

    recorded = {}

    def fake_set_key(path, key, value):
        recorded[(path, key)] = value

    monkeypatch.setattr(config_impl, "set_key", fake_set_key)
    monkeypatch.setattr(config_impl, "find_dotenv", lambda: str(env_file))
    monkeypatch.setattr(config_impl, "load_dotenv", lambda *args, **kwargs: None)
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
    return manager


def test_ui_config_wrap_roundtrip(config_manager):
    ui_config = config_manager.ui_config
    writes: list[bool] = []
    ui_config._write_callback = lambda: writes.append(True)  # type: ignore[attr-defined]

    assert ui_config.get_terminal_wrap_enabled() is True

    result = ui_config.set_terminal_wrap_enabled(False)

    assert result is False
    assert config_manager.config["UI_TERMINAL_WRAP_ENABLED"] is False
    assert config_manager.yaml_config["UI_TERMINAL_WRAP_ENABLED"] is False
    assert ui_config.get_terminal_wrap_enabled() is False
    assert writes


def test_ui_config_debug_preferences_roundtrip(config_manager):
    ui_config = config_manager.ui_config
    writes: list[bool] = []
    ui_config._write_callback = lambda: writes.append(True)  # type: ignore[attr-defined]

    persisted_level = ui_config.set_debug_log_level(logging.DEBUG)
    assert persisted_level in {"DEBUG", logging.DEBUG}
    assert config_manager.config["UI_DEBUG_LOG_LEVEL"] == persisted_level
    assert config_manager.yaml_config["UI_DEBUG_LOG_LEVEL"] == persisted_level

    max_lines = ui_config.set_debug_log_max_lines("500")
    assert max_lines == 500
    assert config_manager.config["UI_DEBUG_LOG_MAX_LINES"] == 500
    assert config_manager.yaml_config["UI_DEBUG_LOG_MAX_LINES"] == 500
    assert ui_config.get_debug_log_max_lines() == 500

    logger_names = ui_config.set_debug_logger_names([" atlas ", "core", ""])
    assert logger_names == ["atlas", "core"]
    assert ui_config.get_debug_logger_names() == ["atlas", "core"]
    assert config_manager.yaml_config["UI_DEBUG_LOGGERS"] == ["atlas", "core"]

    assert ui_config.get_debug_log_initial_lines(123) == 123
    assert ui_config.get_debug_log_format() is None
    assert ui_config.get_debug_log_file_name() is None
    assert ui_config.get_app_root() == config_manager.env_config.get("APP_ROOT")

    ui_config.set_debug_log_level(None)
    assert "UI_DEBUG_LOG_LEVEL" not in config_manager.config
    assert writes


def test_tooling_section_apply_sets_defaults():
    config: dict[str, Any] = {
        "tool_safety": {"network_allowlist": ["  example.com  ", ""]},
        "tool_logging": {},
    }
    section = ToolingConfigSection(
        config=config,
        yaml_config={},
        env_config={
            "JAVASCRIPT_EXECUTOR_BIN": "/usr/bin/node",
            "JAVASCRIPT_EXECUTOR_ARGS": "--inspect",
        },
        logger=_StubLogger(),
    )

    section.apply()

    assert config["tool_defaults"]["timeout_seconds"] == 30
    assert config["tool_safety"]["network_allowlist"] == ["example.com"]
    assert config["tools"]["javascript_executor"]["args"] == ["--inspect"]


def test_persistence_kv_section_set_settings_updates_blocks():
    section = _make_persistence_section(env={"ATLAS_KV_STORE_URL": "postgresql://env"})

    section.apply()
    updated = section.kv_store.set_settings(
        url="postgresql://override",
        namespace_quota_bytes=2048,
        pool={"size": "5"},
    )

    postgres_settings = section.config["tools"]["kv_store"]["adapters"]["postgres"]
    assert postgres_settings["namespace_quota_bytes"] == 2048
    assert section.yaml_config["tools"]["kv_store"]["adapters"]["postgres"]["url"] == "postgresql://override"
    assert updated["adapters"]["postgres"]["pool"]["size"] == 5
    assert section._write_calls  # type: ignore[attr-defined]


def test_persistence_conversation_retention_updates_yaml():
    section = _make_persistence_section()
    section.apply()

    retention = section.conversation.set_retention(days=7, history_limit=50)

    assert retention == {"days": 7, "history_message_limit": 50}
    assert (
        section.yaml_config["conversation_database"]["retention"]["history_message_limit"]
        == 50
    )
    assert section._write_calls  # type: ignore[attr-defined]


def test_persistence_conversation_ensure_postgres_persists_default_url():
    class _Engine:
        def __init__(self):
            self.dispose_called = False

        def dispose(self):
            self.dispose_called = True

    engine = _Engine()
    section = _make_persistence_section(
        env={},
        create_engine=lambda *args, **kwargs: engine,
        inspect_engine=lambda *_args, **_kwargs: types.SimpleNamespace(
            get_table_names=lambda: ["atlas_conversations"]
        ),
        conversation_required_tables=lambda: {"atlas_conversations"},
    )
    section.apply()

    url = section.conversation.ensure_postgres_store()

    assert url == "postgresql://default"
    assert section.conversation.is_verified() is True
    assert section.config["conversation_database"]["url"] == "postgresql://default"
    assert section.yaml_config["conversation_database"]["url"] == "postgresql://default"
    assert engine.dispose_called is True
    assert section._write_calls  # type: ignore[attr-defined]


def test_persistence_conversation_ensure_postgres_missing_tables_raises():
    engine = types.SimpleNamespace(dispose=lambda: None)

    section = _make_persistence_section(
        env={},
        create_engine=lambda *args, **kwargs: engine,
        inspect_engine=lambda *_args, **_kwargs: types.SimpleNamespace(
            get_table_names=lambda: ["other_table"]
        ),
        conversation_required_tables=lambda: {"atlas_conversations"},
    )
    section.apply()

    with pytest.raises(RuntimeError) as excinfo:
        section.conversation.ensure_postgres_store()

    assert "missing required tables" in str(excinfo.value)


def test_messaging_section_apply_and_set():
    config = {"messaging": {"backend": "redis"}}
    yaml_config: dict[str, Any] = {}
    writes: list[bool] = []
    section = MessagingConfigSection(
        config=config,
        yaml_config=yaml_config,
        env_config={"REDIS_URL": "redis://cache:6379/1"},
        logger=_StubLogger(),
        write_yaml_callback=lambda: writes.append(True),
    )

    section.apply()
    assert config["messaging"]["redis_url"] == "redis://cache:6379/1"

    result = section.set_settings(
        backend="redis",
        redis_url="redis://app-cache:6379/2",
        stream_prefix="atlas",
    )

    assert result["stream_prefix"] == "atlas"
    assert yaml_config["messaging"]["redis_url"] == "redis://app-cache:6379/2"
    assert writes
