"""Utility for retrieving and sanitizing web page content."""

from __future__ import annotations

import asyncio
import html
import logging
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional, Sequence, TYPE_CHECKING
from urllib.parse import urlparse

import aiohttp

if TYPE_CHECKING:  # pragma: no cover
    from ATLAS.config import ConfigManager


logger = logging.getLogger(__name__)


class WebpageFetchError(RuntimeError):
    """Base class for webpage fetch failures."""


class DomainNotAllowedError(WebpageFetchError):
    """Raised when attempting to fetch content from a non-allowlisted host."""


class ContentTooLargeError(WebpageFetchError):
    """Raised when the retrieved payload exceeds the configured limit."""


class FetchTimeoutError(WebpageFetchError):
    """Raised when the upstream request exceeds the allotted timeout."""


_BLOCKED_TAGS = {"script", "style", "noscript", "iframe", "object", "embed"}
_AD_KEYWORDS = ("ad", "advert", "sponsored", "promo", "banner")


class _TextExtractor(HTMLParser):
    """Simple HTML to text extractor that skips scripts and ad containers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._capture_title = False
        self.title: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if self._skip_depth:
            self._skip_depth += 1
            return

        lowered = tag.lower()
        attr_map = {name.lower(): value or "" for name, value in attrs}

        if lowered == "title":
            self._capture_title = True

        if lowered in _BLOCKED_TAGS or self._should_filter(attr_map):
            self._skip_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if tag.lower() == "title":
            self._capture_title = False

        if tag.lower() in {"p", "br", "div", "li", "section"}:
            self._chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        if self._capture_title:
            if data.strip():
                if self.title:
                    self.title += " "
                self.title += data.strip()

        if data.strip():
            self._chunks.append(data.strip())

    def handle_entityref(self, name: str) -> None:  # pragma: no cover - handled via convert_charrefs
        pass

    def handle_charref(self, name: str) -> None:  # pragma: no cover - handled via convert_charrefs
        pass

    def _should_filter(self, attrs: dict[str, str]) -> bool:
        id_value = attrs.get("id", "").lower()
        class_value = attrs.get("class", "").lower()
        data_component = attrs.get("data-component", "").lower()

        joined_values = " ".join(filter(None, (id_value, class_value, data_component)))
        return any(keyword in joined_values for keyword in _AD_KEYWORDS)

    def get_text(self) -> str:
        normalized = " ".join(self._chunks)
        return " ".join(normalized.split())


@dataclass(frozen=True)
class WebpageFetchResult:
    """Normalized response returned by :class:`WebpageFetcher`."""

    url: str
    title: Optional[str]
    text: str
    content_length: int


class WebpageFetcher:
    """Fetch, sanitize, and normalize web content."""

    def __init__(
        self,
        *,
        allowed_domains: Optional[Sequence[str]] = None,
        timeout_seconds: float = 15.0,
        max_content_length: int = 2 * 1024 * 1024,
        config_manager: "ConfigManager | None" = None,
    ) -> None:
        self._config_manager = config_manager

        config_allowlist = self._resolve_config_allowlist()
        merged_allowlist = self._merge_allowlists(allowed_domains, config_allowlist)

        self._allowed_domains = merged_allowlist
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_content_length = max_content_length

        if not self._allowed_domains:
            logger.warning(
                "WebpageFetcher initialized without an allowlist; all domains will be blocked."
            )

    def _resolve_config_allowlist(self) -> tuple[str, ...]:
        manager = self._config_manager
        if manager is None:
            from ATLAS.config import ConfigManager

            manager = ConfigManager()
            self._config_manager = manager
        safety_block = manager.get_config("tool_safety", {})
        configured = safety_block.get("network_allowlist")
        if not configured:
            return ()
        normalized: list[str] = []
        for entry in configured:
            host = str(entry).strip().lower()
            if host:
                normalized.append(host)
        return tuple(dict.fromkeys(normalized))

    def _merge_allowlists(
        self,
        provided: Optional[Sequence[str]],
        configured: tuple[str, ...],
    ) -> tuple[str, ...]:
        entries: list[str] = []
        for source in (provided, configured):
            if not source:
                continue
            for domain in source:
                if not domain:
                    continue
                candidate = str(domain).strip().lower()
                if candidate and candidate not in entries:
                    entries.append(candidate)
        return tuple(entries)

    def _is_domain_allowed(self, hostname: Optional[str]) -> bool:
        if not hostname:
            return False

        hostname = hostname.lower()
        for allowed in self._allowed_domains:
            if hostname == allowed or hostname.endswith(f".{allowed}"):
                return True
        return False

    async def fetch(self, url: str) -> WebpageFetchResult:
        """Retrieve and sanitize a web page located at ``url``."""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise WebpageFetchError("Only HTTP and HTTPS URLs are supported.")

        if not self._is_domain_allowed(parsed.hostname):
            raise DomainNotAllowedError(f"Domain '{parsed.hostname or ''}' is not allowlisted.")

        headers = {
            "User-Agent": (
                "ATLASWebFetcher/1.0 (+https://github.com/Atlas-Research-Group)"
            )
        }

        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    declared_length = response.headers.get("Content-Length")
                    if declared_length and int(declared_length) > self._max_content_length:
                        raise ContentTooLargeError(
                            "Remote content-length exceeds configured maximum."
                        )

                    body = await self._read_body(response)
                    text, title = self._sanitize_html(body)

                    final_url = str(response.url) if response.url else url
                    return WebpageFetchResult(
                        url=final_url,
                        title=title or None,
                        text=text,
                        content_length=len(body.encode("utf-8")),
                    )
        except asyncio.TimeoutError as exc:
            raise FetchTimeoutError("Timed out waiting for webpage response.") from exc
        except aiohttp.ClientError as exc:
            raise WebpageFetchError(str(exc)) from exc

    async def _read_body(self, response: aiohttp.ClientResponse) -> str:
        charset = response.charset or "utf-8"
        collected: list[bytes] = []
        total = 0

        async for chunk in response.content.iter_chunked(4096):
            total += len(chunk)
            if total > self._max_content_length:
                raise ContentTooLargeError("Fetched content exceeded maximum length.")
            collected.append(chunk)

        payload = b"".join(collected)
        return payload.decode(charset, errors="replace")

    def _sanitize_html(self, markup: str) -> tuple[str, str]:
        parser = _TextExtractor()
        parser.feed(markup)
        parser.close()

        text = html.unescape(parser.get_text())
        title = html.unescape(parser.title.strip()) if parser.title else ""
        return text, title


async def fetch_webpage(url: str) -> WebpageFetchResult:
    """Convenience wrapper around :class:`WebpageFetcher` using default settings."""

    fetcher = WebpageFetcher()
    return await fetcher.fetch(url)


__all__ = [
    "ContentTooLargeError",
    "DomainNotAllowedError",
    "FetchTimeoutError",
    "WebpageFetchError",
    "WebpageFetchResult",
    "WebpageFetcher",
    "fetch_webpage",
]

