import asyncio
import sys

import pytest

if "yaml" not in sys.modules:
    sys.modules["yaml"] = type(
        "_StubYaml",
        (),
        {
            "safe_load": staticmethod(lambda *_args, **_kwargs: {}),
            "dump": staticmethod(lambda *_args, **_kwargs: None),
        },
    )()

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = type(
        "_StubDotenv",
        (),
        {
            "load_dotenv": staticmethod(lambda *_args, **_kwargs: None),
            "set_key": staticmethod(lambda *_args, **_kwargs: None),
            "find_dotenv": staticmethod(lambda *_args, **_kwargs: None),
        },
    )()

if "pytz" not in sys.modules:
    sys.modules["pytz"] = type(
        "_StubPytz",
        (),
        {"timezone": staticmethod(lambda name: type("TZ", (), {"localize": lambda self, dt: dt})())},
    )()

if "aiohttp" not in sys.modules:
    class _StubClientError(Exception):
        pass

    class _StubClientResponseError(_StubClientError):
        def __init__(self, *_, status=0, message="", headers=None, request_info=None, history=()):
            super().__init__(message)
            self.status = status
            self.headers = headers
            self.request_info = request_info
            self.history = history

    class _StubClientTimeout:
        def __init__(self, total=None):
            self.total = total

    sys.modules["aiohttp"] = type(
        "_StubAiohttp",
        (),
        {
            "ClientSession": object,
            "ClientTimeout": _StubClientTimeout,
            "ClientError": _StubClientError,
            "ClientResponseError": _StubClientResponseError,
        },
    )()

from modules.Tools.Base_Tools import webpage_fetch
from modules.Tools.Base_Tools.webpage_fetch import (
    ContentTooLargeError,
    DomainNotAllowedError,
    FetchTimeoutError,
    WebpageFetcher,
)


class _MockStream:
    def __init__(self, chunks):
        self._chunks = [chunk if isinstance(chunk, bytes) else chunk.encode("utf-8") for chunk in chunks]

    async def iter_chunked(self, size):  # pragma: no cover - size unused in deterministic mock
        for chunk in self._chunks:
            yield chunk


class _MockResponse:
    def __init__(self, *, text: str, status: int = 200, headers=None, url: str = "https://example.com"):
        self._text = text
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.charset = "utf-8"
        self.content = _MockStream([text])

    def raise_for_status(self):
        if self.status >= 400:
            raise webpage_fetch.aiohttp.ClientResponseError(  # pragma: no cover - not triggered
                request_info=None,
                history=(),
                status=self.status,
                message="error",
                headers=None,
            )


class _MockRequestContext:
    def __init__(self, response=None, *, exception=None):
        self._response = response
        self._exception = exception

    async def __aenter__(self):
        if self._exception:
            raise self._exception
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _MockSession:
    def __init__(self, response=None, *, exception=None):
        self._response = response
        self._exception = exception

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *_args, **_kwargs):
        return _MockRequestContext(self._response, exception=self._exception)


def _patch_session(monkeypatch, response=None, *, exception=None):
    def _factory(*_args, **_kwargs):
        return _MockSession(response, exception=exception)

    monkeypatch.setattr(webpage_fetch.aiohttp, "ClientSession", _factory)


def test_webpage_fetch_success(monkeypatch):
    markup = (
        "<html><head><title>Example Domain</title><script>window.ads=1;</script></head>"
        "<body><div class='ad-banner'>ignore me</div><p>Hello <b>World</b></p></body></html>"
    )
    response = _MockResponse(text=markup, headers={"Content-Length": str(len(markup))})
    _patch_session(monkeypatch, response=response)

    fetcher = WebpageFetcher(allowed_domains=("example.com",))
    result = asyncio.run(fetcher.fetch("https://example.com/article"))

    assert result.url == "https://example.com"
    assert result.title == "Example Domain"
    assert "Hello World" in result.text
    assert "ignore me" not in result.text


def test_webpage_fetch_blocks_non_allowlisted_domain():
    fetcher = WebpageFetcher(allowed_domains=("example.com",))

    with pytest.raises(DomainNotAllowedError):
        asyncio.run(fetcher.fetch("https://blocked.invalid/path"))


def test_webpage_fetch_raises_on_timeout(monkeypatch):
    _patch_session(monkeypatch, exception=asyncio.TimeoutError())
    fetcher = WebpageFetcher(allowed_domains=("example.com",))

    with pytest.raises(FetchTimeoutError):
        asyncio.run(fetcher.fetch("https://example.com/slow"))


def test_webpage_fetch_enforces_content_length(monkeypatch):
    payload = "<html><body>" + "x" * 50 + "</body></html>"
    response = _MockResponse(text=payload, headers={"Content-Length": "999999"})
    _patch_session(monkeypatch, response=response)

    fetcher = WebpageFetcher(allowed_domains=("example.com",), max_content_length=32)

    with pytest.raises(ContentTooLargeError):
        asyncio.run(fetcher.fetch("https://example.com/large"))


def test_webpage_fetch_merges_allowlists_without_duplicates(monkeypatch):
    class _StubConfigManager:
        def get_config(self, key, default):
            assert key == "tool_safety"
            return {"network_allowlist": ["example.com", "Example.org", ""]}

    _patch_session(monkeypatch, response=_MockResponse(text="<html></html>"))

    fetcher = WebpageFetcher(
        allowed_domains=["Example.com", "Sub.Example.org", "example.com"],
        config_manager=_StubConfigManager(),
    )

    assert fetcher._allowed_domains == ("example.com", "sub.example.org", "example.org")
