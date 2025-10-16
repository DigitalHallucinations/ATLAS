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


@pytest.fixture
def reset_serpapi_config():
    original_config = dict(config_manager.config)
    original_env = os.getenv("SERPAPI_KEY")
    try:
        yield
    finally:
        config_manager.config.clear()
        config_manager.config.update(original_config)
        if original_env is None:
            os.environ.pop("SERPAPI_KEY", None)
        else:
            os.environ["SERPAPI_KEY"] = original_env


def test_google_search_prefers_configured_key_over_environment(monkeypatch, reset_serpapi_config):
    monkeypatch.setenv("SERPAPI_KEY", "from-env")
    config_manager.config["SERPAPI_KEY"] = "from-config"

    search = GoogleSearch()

    assert search.api_key == "from-config"


def test_google_search_returns_error_when_key_missing(monkeypatch, reset_serpapi_config):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    config_manager.config.pop("SERPAPI_KEY", None)

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
    assert "SerpAPI key is not configured" in message
    assert called is False
