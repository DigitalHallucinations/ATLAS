import asyncio
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda *_args, **_kwargs: {}
    yaml_module.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_module

if "dotenv" not in sys.modules:
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *_args, **_kwargs: None
    dotenv_module.set_key = lambda *_args, **_kwargs: None
    dotenv_module.find_dotenv = lambda *_args, **_kwargs: ""
    sys.modules["dotenv"] = dotenv_module

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "Tools" / "Base_Tools"

if "modules" not in sys.modules:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    sys.modules["modules"] = modules_pkg
else:
    modules_pkg = sys.modules["modules"]

if "modules.Tools" not in sys.modules:
    tools_pkg = types.ModuleType("modules.Tools")
    tools_pkg.__path__ = []
    sys.modules["modules.Tools"] = tools_pkg
else:
    tools_pkg = sys.modules["modules.Tools"]

setattr(modules_pkg, "Tools", tools_pkg)

if "modules.Tools.Base_Tools" not in sys.modules:
    base_tools_pkg = types.ModuleType("modules.Tools.Base_Tools")
    base_tools_pkg.__path__ = [str(MODULE_PATH)]
    sys.modules["modules.Tools.Base_Tools"] = base_tools_pkg
else:
    base_tools_pkg = sys.modules["modules.Tools.Base_Tools"]

setattr(tools_pkg, "Base_Tools", base_tools_pkg)

if "modules.logging" not in sys.modules:
    logging_pkg = types.ModuleType("modules.logging")
    logging_pkg.__path__ = []
    sys.modules["modules.logging"] = logging_pkg

if "modules.logging.logger" not in sys.modules:
    logger_module = types.ModuleType("modules.logging.logger")

    def _setup_logger(name):
        return logging.getLogger(name)

    logger_module.setup_logger = _setup_logger
    sys.modules["modules.logging.logger"] = logger_module

spec = importlib.util.spec_from_file_location(
    "modules.Tools.Base_Tools.Google_search",
    MODULE_PATH / "Google_search.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
setattr(base_tools_pkg, "Google_search", module)

GoogleSearch = module.GoogleSearch
config_manager = module.config_manager


SEARCH_KEYS = ("GOOGLE_API_KEY", "SERPAPI_KEY")


class _StubConfigManager:
    def __init__(self):
        self.config = {}

    def get_config(self, key):
        return self.config.get(key)


@pytest.fixture
def reset_search_config():
    original_manager = getattr(config_manager, "_manager", None)
    original_config = None
    if original_manager is not None:
        original_config = dict(getattr(original_manager, "config", {}))

    stub_manager = _StubConfigManager()
    config_manager.set_manager(stub_manager)

    original_env = {key: os.getenv(key) for key in SEARCH_KEYS}
    try:
        yield
    finally:
        if original_manager is None:
            config_manager.set_manager(None)
        else:
            config_manager.set_manager(original_manager)
            if original_config is not None:
                current = getattr(original_manager, "config", None)
                if isinstance(current, dict):
                    current.clear()
                    current.update(original_config)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_google_search_prefers_configured_google_key_over_other_sources(monkeypatch, reset_search_config):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    monkeypatch.setenv("SERPAPI_KEY", "legacy-env")
    config_manager.config["GOOGLE_API_KEY"] = "from-config"
    config_manager.config["SERPAPI_KEY"] = "legacy-config"

    search = GoogleSearch()

    assert search.api_key == "from-config"


def test_google_search_uses_google_env_before_legacy_sources(monkeypatch, reset_search_config):
    config_manager.config.pop("GOOGLE_API_KEY", None)
    config_manager.config.pop("SERPAPI_KEY", None)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    monkeypatch.setenv("SERPAPI_KEY", "legacy-env")

    search = GoogleSearch()

    assert search.api_key == "from-env"


def test_google_search_falls_back_to_legacy_config(monkeypatch, reset_search_config):
    for key in SEARCH_KEYS:
        config_manager.config.pop(key, None)
        monkeypatch.delenv(key, raising=False)

    config_manager.config["SERPAPI_KEY"] = "legacy-config"

    search = GoogleSearch()

    assert search.api_key == "legacy-config"


def test_google_search_falls_back_to_legacy_environment(monkeypatch, reset_search_config):
    for key in SEARCH_KEYS:
        config_manager.config.pop(key, None)
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("SERPAPI_KEY", "legacy-env")

    search = GoogleSearch()

    assert search.api_key == "legacy-env"


def test_google_search_returns_error_when_key_missing(monkeypatch, reset_search_config):
    for key in SEARCH_KEYS:
        monkeypatch.delenv(key, raising=False)
        config_manager.config.pop(key, None)

    called = False

    def fake_get(*_args, **_kwargs):  # pragma: no cover - should not execute
        nonlocal called
        called = True
        raise AssertionError("requests.get should not be called when API key is missing")

    monkeypatch.setattr(
        "modules.Tools.Base_Tools.Google_search.requests.get",
        fake_get,
    )

    search = GoogleSearch()
    status, message = asyncio.run(search._search("test query"))

    assert status == -1
    assert "GOOGLE_API_KEY" in message
    assert "SERPAPI_KEY" in message
    assert called is False
