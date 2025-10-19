import asyncio
import os
import sys
import types

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

from modules.Tools.Base_Tools.Google_search import GoogleSearch, config_manager


SEARCH_KEYS = ("GOOGLE_API_KEY", "SERPAPI_KEY")


@pytest.fixture
def reset_search_config():
    original_config = dict(config_manager.config)
    original_env = {key: os.getenv(key) for key in SEARCH_KEYS}
    try:
        yield
    finally:
        config_manager.config.clear()
        config_manager.config.update(original_config)
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
