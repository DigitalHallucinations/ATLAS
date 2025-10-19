import asyncio
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

from modules.Tools.Base_Tools.Google_search import GoogleSearch
from modules.Tools.providers.router import ToolProviderRouter


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
