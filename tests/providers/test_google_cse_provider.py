import asyncio
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1] / "modules"
BASE_TOOLS_PATH = MODULE_ROOT / "Tools" / "Base_Tools"
PROVIDERS_PATH = MODULE_ROOT / "Tools" / "providers"

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

if "modules" not in sys.modules:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = [str(MODULE_ROOT)]
    sys.modules["modules"] = modules_pkg
else:
    modules_pkg = sys.modules["modules"]

if "modules.Tools" not in sys.modules:
    tools_pkg = types.ModuleType("modules.Tools")
    tools_pkg.__path__ = [str(MODULE_ROOT / "Tools")]
    sys.modules["modules.Tools"] = tools_pkg
else:
    tools_pkg = sys.modules["modules.Tools"]

setattr(modules_pkg, "Tools", tools_pkg)

if "modules.Tools.Base_Tools" not in sys.modules:
    base_tools_pkg = types.ModuleType("modules.Tools.Base_Tools")
    base_tools_pkg.__path__ = [str(BASE_TOOLS_PATH)]
    sys.modules["modules.Tools.Base_Tools"] = base_tools_pkg
else:
    base_tools_pkg = sys.modules["modules.Tools.Base_Tools"]

setattr(tools_pkg, "Base_Tools", base_tools_pkg)

if "modules.Tools.providers" not in sys.modules:
    providers_pkg = types.ModuleType("modules.Tools.providers")
    providers_pkg.__path__ = [str(PROVIDERS_PATH)]
    sys.modules["modules.Tools.providers"] = providers_pkg
else:
    providers_pkg = sys.modules["modules.Tools.providers"]

setattr(tools_pkg, "providers", providers_pkg)

if "modules.logging" not in sys.modules:
    logging_pkg = types.ModuleType("modules.logging")
    logging_pkg.__path__ = []
    sys.modules["modules.logging"] = logging_pkg
else:
    logging_pkg = sys.modules["modules.logging"]

if "modules.logging.logger" not in sys.modules:
    logger_module = types.ModuleType("modules.logging.logger")

    def _setup_logger(name):
        return logging.getLogger(name)

    logger_module.setup_logger = _setup_logger
    sys.modules["modules.logging.logger"] = logger_module
else:
    logger_module = sys.modules["modules.logging.logger"]

setattr(logging_pkg, "logger", logger_module)

if "ATLAS" not in sys.modules:
    atlas_pkg = types.ModuleType("ATLAS")
    atlas_pkg.__path__ = []
    sys.modules["ATLAS"] = atlas_pkg
else:
    atlas_pkg = sys.modules["ATLAS"]

if "ATLAS.config" not in sys.modules:
    atlas_config_module = types.ModuleType("ATLAS.config")

    class _StubConfigManager:
        def __init__(self):
            self.config = {}

        def get_config(self, key):
            return self.config.get(key)

    atlas_config_module.ConfigManager = _StubConfigManager
    sys.modules["ATLAS.config"] = atlas_config_module
else:
    atlas_config_module = sys.modules["ATLAS.config"]

setattr(atlas_pkg, "config", atlas_config_module)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


google_search_module = _load_module(
    "modules.Tools.Base_Tools.Google_search", BASE_TOOLS_PATH / "Google_search.py"
)
setattr(base_tools_pkg, "Google_search", google_search_module)

providers_base_module = _load_module(
    "modules.Tools.providers.base", PROVIDERS_PATH / "base.py"
)
setattr(providers_pkg, "base", providers_base_module)

providers_registry_module = _load_module(
    "modules.Tools.providers.registry", PROVIDERS_PATH / "registry.py"
)
setattr(providers_pkg, "registry", providers_registry_module)

providers_router_module = _load_module(
    "modules.Tools.providers.router", PROVIDERS_PATH / "router.py"
)
setattr(providers_pkg, "router", providers_router_module)

providers_serpapi_module = _load_module(
    "modules.Tools.providers.serpapi", PROVIDERS_PATH / "serpapi.py"
)
setattr(providers_pkg, "serpapi", providers_serpapi_module)

providers_google_cse_module = _load_module(
    "modules.Tools.providers.google_cse", PROVIDERS_PATH / "google_cse.py"
)
setattr(providers_pkg, "google_cse", providers_google_cse_module)


GoogleSearch = google_search_module.GoogleSearch
ToolProviderRouter = providers_router_module.ToolProviderRouter


GOOGLE_PROVIDER_SPEC = {
    "name": "google_cse",
    "priority": 0,
    "health_check_interval": 300,
    "config": {
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_config": "GOOGLE_API_KEY",
        "cse_id_env": "GOOGLE_CSE_ID",
        "cse_id_config": "GOOGLE_CSE_ID",
    },
}

SERP_PROVIDER_SPEC = {
    "name": "serpapi",
    "priority": 10,
    "health_check_interval": 300,
}


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GOOGLE_CSE_ID", "SERPAPI_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield


def _build_router(*, fallback=None):
    if fallback is None:
        fallback = GoogleSearch()._search
    return ToolProviderRouter(
        tool_name="google_search",
        provider_specs=[GOOGLE_PROVIDER_SPEC, SERP_PROVIDER_SPEC],
        fallback_callable=fallback,
    )


def test_google_cse_selected_when_credentials_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "search-engine")

    serp_called = False

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"items": [{"title": "result"}]}

        @staticmethod
        def raise_for_status():
            return None

    def fake_cse_get(url, params=None, timeout=None):
        assert url == "https://customsearch.googleapis.com/customsearch/v1"
        assert params["key"] == "test-google-key"
        assert params["cx"] == "search-engine"
        assert timeout == 10
        return DummyResponse()

    monkeypatch.setattr("modules.Tools.providers.google_cse.requests.get", fake_cse_get)
    async def unexpected_fallback(**_kwargs):  # pragma: no cover - should not execute
        nonlocal serp_called
        serp_called = True
        raise AssertionError("Fallback should not run when CSE credentials are configured")

    router = _build_router(fallback=unexpected_fallback)
    status, payload = asyncio.run(router.call(query="atlas", k=3))

    assert status == 200
    assert isinstance(payload, dict)
    assert serp_called is False
    assert router._states[0].name == "google_cse"
    assert router._states[0].health.successes >= 1


def test_google_search_falls_back_to_serpapi_when_cse_credentials_missing(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "serp-key")

    cse_called = False

    def fail_cse_get(*_args, **_kwargs):  # pragma: no cover - health check prevents call
        nonlocal cse_called
        cse_called = True
        raise AssertionError("CSE client should not be invoked when credentials are missing")

    monkeypatch.setattr("modules.Tools.providers.google_cse.requests.get", fail_cse_get)
    serp_called = False

    async def fake_serp_search(self, **kwargs):
        nonlocal serp_called
        serp_called = True
        return 200, {"source": "serpapi", "kwargs": kwargs}

    async def unexpected_fallback(**_kwargs):  # pragma: no cover - should not execute
        raise AssertionError("Fallback should not run when SerpAPI is healthy")

    router = _build_router(fallback=unexpected_fallback)
    monkeypatch.setattr("modules.Tools.Base_Tools.Google_search.GoogleSearch._search", fake_serp_search, raising=False)

    status, payload = asyncio.run(router.call(query="atlas", k=2))

    assert status == 200
    assert isinstance(payload, dict)
    assert payload.get("source") == "serpapi"
    assert serp_called is True
    assert cse_called is False
    assert router._states[0].health.failures >= 1
    assert router._states[1].health.successes >= 1
