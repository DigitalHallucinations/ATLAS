"""SerpAPI-backed provider implementation for Google search."""

from __future__ import annotations

from typing import Any

from modules.Tools.Base_Tools.Google_search import GoogleSearch

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


class SerpApiToolProvider(ToolProvider):
    """Provider that proxies Google search requests through SerpAPI."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        api_key = self.config.get("api_key")
        self._client = GoogleSearch(api_key=api_key)

    async def call(self, **kwargs: Any) -> Any:
        return await self._client._search(**kwargs)

    async def health_check(self) -> bool:
        # Consider the provider healthy if an API key is configured. This keeps the
        # check lightweight while still surfacing missing credentials as unhealthy.
        return bool(getattr(self._client, "api_key", "").strip())


tool_provider_registry.register("serpapi", SerpApiToolProvider)

__all__ = ["SerpApiToolProvider"]
