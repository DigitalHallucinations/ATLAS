import asyncio
import datetime
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

yaml_stub = ModuleType("yaml")
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
yaml_stub.dump = lambda *_args, **_kwargs: None
sys.modules.setdefault("yaml", yaml_stub)

dotenv_stub = ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
dotenv_stub.set_key = lambda *_args, **_kwargs: None
dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
sys.modules.setdefault("dotenv", dotenv_stub)


class _StubTimezone(datetime.tzinfo):
    def __init__(self, name: str) -> None:
        self._name = name

    def utcoffset(self, _dt):
        return datetime.timedelta(0)

    def dst(self, _dt):
        return datetime.timedelta(0)

    def tzname(self, _dt):
        return self._name


pytz_stub = ModuleType("pytz")
pytz_stub.timezone = lambda name: _StubTimezone(name)
sys.modules.setdefault("pytz", pytz_stub)

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "Tools" / "Base_Tools" / "browser_lite.py"
spec = importlib.util.spec_from_file_location("browser_lite_test_module", MODULE_PATH)
browser_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = browser_module
spec.loader.exec_module(browser_module)

BrowserLite = browser_module.BrowserLite
PersonaPolicyViolationError = browser_module.PersonaPolicyViolationError
RobotsBlockedError = browser_module.RobotsBlockedError
_HTTPResult = browser_module._HTTPResult


class StubFetcher:
    def __init__(self, pages):
        self._pages = pages
        self.calls = []

    async def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        page = self._pages[url]
        return _HTTPResult(
            url=url,
            status_code=page.get("status", 200),
            headers=page.get("headers", {"Content-Type": "text/html"}),
            content=page.get("body", "").encode("utf-8"),
        )


class StubRobots:
    def __init__(self, allow_map):
        self.allow_map = allow_map

    async def allows(self, _user_agent, url):
        return self.allow_map.get(url, True)


async def _screenshotter(url, _content):
    return f"shot:{url}".encode("utf-8")


def test_navigation_and_text_extraction():
    pages = {
        "https://example.test/": {"body": "<html><body><p>Hello world</p><script>ignore</script></body></html>"},
        "https://example.test/next": {"body": "<html><body><p>Next page</p></body></html>"},
    }
    fetcher = StubFetcher(pages)
    robots = StubRobots({"https://example.test/": True, "https://example.test/next": True})
    tool = BrowserLite(
        allowed_domains=["example.test"],
        throttle_seconds=0,
        fetcher=fetcher,
        robots_cache=robots,
        screenshotter=_screenshotter,
    )

    result = asyncio.run(
        tool.run(
            "https://example.test/",
            actions=[{"type": "navigate", "url": "next"}],
            take_screenshot=True,
            extract_text=True,
        )
    )

    assert result["final_url"] == "https://example.test/next"
    assert len(result["pages"]) == 2
    assert result["pages"][0]["text"] == "Hello world"
    assert result["pages"][1]["screenshot"].startswith("c2hvdDpodHRwczovL2V4YW1wbGUudGVzdC9uZXh0"[:16])

def test_robots_enforced():
    pages = {"https://blocked.test/": {"body": "<p>ok</p>"}}
    fetcher = StubFetcher(pages)
    robots = StubRobots({"https://blocked.test/": False})
    tool = BrowserLite(
        allowed_domains=["blocked.test"],
        throttle_seconds=0,
        fetcher=fetcher,
        robots_cache=robots,
    )

    with pytest.raises(RobotsBlockedError):
        asyncio.run(tool.run("https://blocked.test/"))


def test_form_submission_requires_policy():
    pages = {
        "https://example.test/": {"body": "<p>start</p>"},
        "https://example.test/form": {"body": "<p>submitted</p>"},
    }
    fetcher = StubFetcher(pages)
    robots = StubRobots({"https://example.test/": True, "https://example.test/form": True})
    tool = BrowserLite(
        allowed_domains=["example.test"],
        throttle_seconds=0,
        fetcher=fetcher,
        robots_cache=robots,
    )

    with pytest.raises(PersonaPolicyViolationError):
        asyncio.run(
            tool.run(
                "https://example.test/",
                actions=[
                    {
                        "type": "submit_form",
                        "url": "https://example.test/form",
                        "data": {"q": "value"},
                    }
                ],
                allow_forms=True,
            )
        )


def test_form_submission_allowed_with_persona():
    pages = {
        "https://example.test/": {"body": "<p>start</p>"},
        "https://example.test/form": {"body": "<p>submitted</p>"},
    }
    fetcher = StubFetcher(pages)
    robots = StubRobots({"https://example.test/": True, "https://example.test/form": True})
    tool = BrowserLite(
        allowed_domains=["example.test"],
        throttle_seconds=0,
        fetcher=fetcher,
        robots_cache=robots,
    )

    persona = {"browser_high_risk_enabled": True}

    result = asyncio.run(
        tool.run(
            "https://example.test/",
            actions=[
                {
                    "type": "submit_form",
                    "url": "https://example.test/form",
                    "method": "POST",
                    "data": {"field": "value"},
                }
            ],
            allow_forms=True,
            persona_context=persona,
            extract_text=True,
        )
    )

    assert result["pages"][-1]["text"] == "submitted"
    assert fetcher.calls[-1][0] == "https://example.test/form"
