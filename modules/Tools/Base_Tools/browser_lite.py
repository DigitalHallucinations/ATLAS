"""Lightweight browser automation helper with strong safety guarantees."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import urljoin, urlparse, urlencode

import requests
from requests import Response

try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - playwright is optional during testing
    async_playwright = None  # type: ignore

from ATLAS.config import ConfigManager


logger = logging.getLogger(__name__)


class BrowserLiteError(RuntimeError):
    """Base class for browser-lite failures."""


class DomainNotAllowlistedError(BrowserLiteError):
    """Raised when attempting to navigate to a blocked host."""


class RobotsBlockedError(BrowserLiteError):
    """Raised when robots.txt disallows a navigation target."""


class NavigationLimitError(BrowserLiteError):
    """Raised when the session exceeds the configured page budget."""


class FormSubmissionNotAllowedError(BrowserLiteError):
    """Raised when form submission is attempted without authorization."""


class PersonaPolicyViolationError(BrowserLiteError):
    """Raised when persona policy flags do not permit a high risk action."""


class NavigationFailedError(BrowserLiteError):
    """Raised when the upstream request fails."""


@dataclass(frozen=True)
class BrowserPage:
    """Structured payload describing a visited page."""

    url: str
    status: int
    content_type: str
    content_length: int
    text: Optional[str]
    screenshot_base64: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "text": self.text,
            "screenshot": self.screenshot_base64,
        }


@dataclass(frozen=True)
class _HTTPResult:
    url: str
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    @property
    def content_type(self) -> str:
        header = self.headers.get("Content-Type", "text/html")
        if isinstance(header, str):
            return header
        return str(header)


class _TextExtractor(HTMLParser):
    """Convert HTML documents into normalized whitespace text."""

    _SKIP_TAGS = {"script", "style", "noscript", "iframe", "object", "embed"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: Iterable[tuple[str, Optional[str]]]) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if self._skip_depth:
            self._skip_depth += 1
            return

        if lowered == "title":
            self._capture_title = True

        if lowered in self._SKIP_TAGS:
            self._skip_depth = 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if lowered == "title":
            self._capture_title = False

        if lowered in {"p", "br", "div", "li", "section"}:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    def get_text(self) -> str:
        joined = " ".join(self._chunks)
        return " ".join(joined.split())


class RobotsCache:
    """Cache and evaluate robots.txt directives using the configured fetcher."""

    def __init__(self, fetcher: Callable[[str], Awaitable[_HTTPResult]]) -> None:
        self._fetcher = fetcher
        self._parsers: MutableMapping[str, "RobotFileParser"] = {}
        self._locks: MutableMapping[str, asyncio.Lock] = {}

    async def allows(self, user_agent: str, url: str) -> bool:
        from urllib.robotparser import RobotFileParser

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = self._parsers.get(robots_url)
        if parser is None:
            lock = self._locks.setdefault(robots_url, asyncio.Lock())
            async with lock:
                parser = self._parsers.get(robots_url)
                if parser is None:
                    try:
                        result = await self._fetcher(robots_url)
                        text = result.content.decode("utf-8", errors="ignore")
                    except Exception:
                        parser = RobotFileParser()
                        parser.parse(["User-agent: *", "Allow: /"])
                    else:
                        parser = RobotFileParser()
                        parser.set_url(robots_url)
                        parser.parse(text.splitlines())
                    self._parsers[robots_url] = parser
        return parser.can_fetch(user_agent, url)


def _default_fetch(url: str, *, method: str = "GET", data: Optional[Mapping[str, str]] = None, headers: Optional[Mapping[str, str]] = None, timeout: float = 20.0) -> _HTTPResult:
    try:
        response: Response = requests.request(
            method,
            url,
            data=data,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise NavigationFailedError(str(exc)) from exc

    return _HTTPResult(
        url=response.url,
        status_code=response.status_code,
        headers=dict(response.headers),
        content=response.content,
    )


class BrowserLite:
    """Perform constrained browsing tasks with robots and persona awareness."""

    DEFAULT_USER_AGENT = "ATLASBrowserLite/1.0"

    def __init__(
        self,
        *,
        allowed_domains: Optional[Sequence[str]] = None,
        max_pages_per_session: int = 5,
        throttle_seconds: float = 0.5,
        request_timeout: float = 20.0,
        user_agent: Optional[str] = None,
        fetcher: Optional[Callable[..., Awaitable[_HTTPResult]]] = None,
        screenshotter: Optional[Callable[[str, bytes], Awaitable[Optional[bytes]]]] = None,
        robots_cache: Optional[RobotsCache] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        self._config_manager = config_manager or ConfigManager()
        self._allowed_domains = self._build_allowlist(allowed_domains)
        self._max_pages = max(1, int(max_pages_per_session))
        self._throttle_seconds = max(0.0, float(throttle_seconds))
        self._timeout = float(request_timeout)
        self._user_agent = (user_agent or self.DEFAULT_USER_AGENT).strip()

        self._base_fetcher = fetcher
        if self._base_fetcher is None:
            self._base_fetcher = lambda url, **kwargs: asyncio.to_thread(  # type: ignore[assignment]
                _default_fetch,
                url,
                **kwargs,
            )

        self._robots_cache = robots_cache or RobotsCache(
            lambda url: self._base_fetcher(url, method="GET", headers={"User-Agent": self._user_agent})
        )

        self._screenshotter = screenshotter or _build_default_screenshotter()

        if not self._allowed_domains:
            logger.warning(
                "BrowserLite initialized without any network allowlist. All navigation attempts will be rejected."
            )

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]],
        *,
        config_manager: Optional[ConfigManager] = None,
    ) -> "BrowserLite":
        config = dict(config or {})
        allowed = config.get("allowed_domains")
        throttle = config.get("throttle_seconds", 0.5)
        max_pages = config.get("max_pages_per_session", 5)
        timeout = config.get("request_timeout", 20.0)
        user_agent = config.get("user_agent")
        return cls(
            allowed_domains=allowed,
            throttle_seconds=float(throttle),
            max_pages_per_session=int(max_pages),
            request_timeout=float(timeout),
            user_agent=user_agent,
            config_manager=config_manager,
        )

    def _build_allowlist(self, provided: Optional[Sequence[str]]) -> tuple[str, ...]:
        configured = self._config_manager.get_config("tool_safety", {})
        configured_list: Sequence[str] = ()
        if isinstance(configured, Mapping):
            network_allowlist = configured.get("network_allowlist")
            if isinstance(network_allowlist, Sequence) and not isinstance(network_allowlist, (str, bytes)):
                configured_list = tuple(str(entry).strip().lower() for entry in network_allowlist if entry)
        entries: list[str] = []
        for source in (provided, configured_list):
            if not source:
                continue
            for domain in source:
                if not domain:
                    continue
                normalized = str(domain).strip().lower()
                if normalized and normalized not in entries:
                    entries.append(normalized)
        return tuple(entries)

    async def run(
        self,
        url: str,
        *,
        actions: Optional[Sequence[Mapping[str, Any]]] = None,
        allow_forms: bool = False,
        take_screenshot: bool = False,
        extract_text: bool = True,
        persona_context: Optional[Any] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise BrowserLiteError("A non-empty URL is required.")

        page_budget = min(self._max_pages, max_pages or self._max_pages)
        tasks = [
            {"type": "navigate", "url": url},
            *list(actions or []),
        ]

        pages: list[BrowserPage] = []
        current_url = url
        last_fetch_time = 0.0

        for index, task in enumerate(tasks):
            if index >= page_budget:
                raise NavigationLimitError(
                    f"Navigation aborted: exceeded limit of {page_budget} pages for this session."
                )

            task_type = str(task.get("type", "navigate")).lower()
            if task_type == "navigate":
                target_url = self._resolve_url(current_url, task.get("url"))
                result, last_fetch_time = await self._navigate(
                    target_url,
                    method=str(task.get("method", "GET")),
                    last_fetch_time=last_fetch_time,
                )
            elif task_type == "submit_form":
                if not allow_forms:
                    raise FormSubmissionNotAllowedError(
                        "Form submission attempted while 'allow_forms' is disabled."
                    )
                if not self._persona_allows_high_risk(persona_context):
                    raise PersonaPolicyViolationError(
                        "Persona configuration does not allow high-risk browser actions."
                    )
                target_url = self._resolve_url(current_url, task.get("url"))
                result, last_fetch_time = await self._submit_form(
                    target_url,
                    method=str(task.get("method", "POST")),
                    data=task.get("data"),
                    last_fetch_time=last_fetch_time,
                )
            else:
                raise BrowserLiteError(f"Unsupported browser action '{task_type}'.")

            page = await self._build_page(
                result,
                take_screenshot=take_screenshot,
                extract_text=extract_text,
            )
            pages.append(page)
            current_url = result.url

        return {
            "final_url": current_url,
            "pages": [page.as_dict() for page in pages],
        }

    async def _navigate(
        self,
        url: str,
        *,
        method: str = "GET",
        last_fetch_time: float,
    ) -> tuple[_HTTPResult, float]:
        method = method.upper()
        if method != "GET":
            raise BrowserLiteError(f"Navigation method '{method}' is not supported.")
        self._enforce_allowlist(url)
        await self._respect_robots(url)
        await self._throttle(last_fetch_time)
        result = await self._fetch(url, method=method)
        return result, time.monotonic()

    async def _submit_form(
        self,
        url: str,
        *,
        method: str,
        data: Optional[Mapping[str, Any]],
        last_fetch_time: float,
    ) -> tuple[_HTTPResult, float]:
        method = method.upper()
        if method not in {"GET", "POST"}:
            raise BrowserLiteError(f"Form submission method '{method}' is not supported.")
        payload: Optional[Mapping[str, str]] = None
        if data is not None:
            payload = {str(key): str(value) for key, value in dict(data).items()}

        if method == "GET" and payload:
            query = urlencode(payload)
            separator = "&" if urlparse(url).query else "?"
            url = f"{url}{separator}{query}"
            payload = None

        self._enforce_allowlist(url)
        await self._respect_robots(url)
        await self._throttle(last_fetch_time)
        result = await self._fetch(url, method=method, data=payload)
        return result, time.monotonic()

    async def _fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        data: Optional[Mapping[str, str]] = None,
    ) -> _HTTPResult:
        headers = {"User-Agent": self._user_agent}
        result = await self._base_fetcher(
            url,
            method=method,
            data=data,
            headers=headers,
            timeout=self._timeout,
        )
        if result.status_code >= 400:
            raise NavigationFailedError(f"Upstream responded with HTTP {result.status_code} for {url}.")
        return result

    async def _build_page(
        self,
        result: _HTTPResult,
        *,
        take_screenshot: bool,
        extract_text: bool,
    ) -> BrowserPage:
        screenshot_b64: Optional[str] = None
        if take_screenshot and self._screenshotter is not None:
            try:
                capture = await self._screenshotter(result.url, result.content)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Screenshot capture failed: %s", exc)
            else:
                if capture:
                    screenshot_b64 = base64.b64encode(capture).decode("ascii")

        extracted_text: Optional[str] = None
        if extract_text and "html" in result.content_type.lower():
            parser = _TextExtractor()
            try:
                parser.feed(result.content.decode("utf-8", errors="ignore"))
            except Exception:  # pragma: no cover - html parsing errors
                extracted_text = None
            else:
                extracted_text = parser.get_text()

        return BrowserPage(
            url=result.url,
            status=result.status_code,
            content_type=result.content_type,
            content_length=len(result.content),
            text=extracted_text,
            screenshot_base64=screenshot_b64,
        )

    async def _respect_robots(self, url: str) -> None:
        allowed = await self._robots_cache.allows(self._user_agent, url)
        if not allowed:
            raise RobotsBlockedError(f"robots.txt disallows navigation to {url}.")

    async def _throttle(self, last_fetch_time: float) -> None:
        if self._throttle_seconds <= 0:
            return
        elapsed = time.monotonic() - last_fetch_time if last_fetch_time else self._throttle_seconds
        remaining = self._throttle_seconds - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

    def _enforce_allowlist(self, url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise BrowserLiteError("Unable to determine hostname for URL navigation.")
        for allowed in self._allowed_domains:
            if hostname == allowed or hostname.endswith(f".{allowed}"):
                return
        raise DomainNotAllowlistedError(f"Domain '{hostname}' is not allowlisted for browsing.")

    def _resolve_url(self, current_url: str, target: Optional[str]) -> str:
        if target is None:
            return current_url
        target = str(target).strip()
        if not target:
            return current_url
        return urljoin(current_url, target)

    def _persona_allows_high_risk(self, persona_context: Optional[Any]) -> bool:
        if persona_context is None:
            return False

        def _coerce(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "enabled"}:
                    return True
                if lowered in {"false", "0", "no", "disabled"}:
                    return False
            return None

        if isinstance(persona_context, Mapping):
            direct = persona_context.get("browser_high_risk_enabled")
            coerced = _coerce(direct)
            if coerced is not None:
                return coerced

            type_block = persona_context.get("type")
            if isinstance(type_block, Mapping):
                for key in ("personal_assistant", "research", "automation"):
                    persona_block = type_block.get(key)
                    if isinstance(persona_block, Mapping):
                        value = persona_block.get("browser_high_risk_enabled")
                        coerced = _coerce(value)
                        if coerced is not None:
                            return coerced
        return False


async def _playwright_screenshot(url: str, content: bytes) -> Optional[bytes]:  # pragma: no cover - optional runtime
    if async_playwright is None:
        return None
    html = content.decode("utf-8", errors="ignore")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.set_content(html, wait_until="load")
        screenshot = await page.screenshot(type="png")
        await browser.close()
    return screenshot


def _build_default_screenshotter() -> Optional[Callable[[str, bytes], Awaitable[Optional[bytes]]]]:
    if async_playwright is None:
        return None
    return _playwright_screenshot
