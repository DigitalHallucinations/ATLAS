"""Provider that proxies webpage fetches through :class:`WebpageFetcher`."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from modules.Tools.Base_Tools.webpage_fetch import WebpageFetcher

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


def _coerce_sequence(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else None
    if isinstance(value, Iterable):
        entries: list[str] = []
        for item in value:
            if item is None:
                continue
            candidate = str(item).strip()
            if candidate and candidate not in entries:
                entries.append(candidate)
        return entries or None
    return None


class InternalHttpClientProvider(ToolProvider):
    """Fetch and sanitize HTTP content using the shared :class:`WebpageFetcher`."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)

        allowed = _coerce_sequence(self.config.get("allowed_domains"))

        def _coerce_float(key: str, default: float) -> float:
            value = self.config.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(key: str, default: int) -> int:
            value = self.config.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        timeout = _coerce_float("timeout_seconds", _coerce_float("timeout", 15.0))
        max_length = _coerce_int("max_content_length", 2 * 1024 * 1024)

        self._fetcher = WebpageFetcher(
            allowed_domains=allowed,
            timeout_seconds=timeout,
            max_content_length=max_length,
        )

    async def call(self, **kwargs: Any) -> Any:
        url = kwargs.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("'url' must be a non-empty string")
        return await self._fetcher.fetch(url)


tool_provider_registry.register("internal_http_client", InternalHttpClientProvider)

__all__ = ["InternalHttpClientProvider"]

