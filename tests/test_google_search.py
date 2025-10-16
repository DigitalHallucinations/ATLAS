import asyncio
import logging
import sys
import types

if "yaml" not in sys.modules:
    yaml_stub = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.SimpleNamespace(
        load_dotenv=lambda *args, **kwargs: None,
        set_key=lambda *args, **kwargs: None,
        find_dotenv=lambda *args, **kwargs: None,
    )
    sys.modules["dotenv"] = dotenv_stub

from modules.Tools.Base_Tools import Google_search


class DummyResponse:
    status_code = 200

    @staticmethod
    def raise_for_status():
        return None

    @staticmethod
    def json():
        return {"organic_results": [{"title": "example"}]}


def test_google_search_logs_redact_api_key(monkeypatch):
    api_key = "super-secret-key"
    google_search = Google_search.GoogleSearch(api_key=api_key)

    def fake_get(url, params=None, timeout=None):
        assert params == {"q": "python", "api_key": api_key}
        return DummyResponse()

    monkeypatch.setattr(Google_search.requests, "get", fake_get)

    handler = _ListHandler()
    Google_search.logger.addHandler(handler)

    try:
        status_code, data = asyncio.run(google_search._search("python"))
    finally:
        Google_search.logger.removeHandler(handler)

    assert status_code == 200
    assert data == {"organic_results": [{"title": "example"}]}

    messages = [record.getMessage() for record in handler.records]
    assert any("***REDACTED***" in message for message in messages)
    assert all(api_key not in message for message in messages)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)
