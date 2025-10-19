"""Provider that forwards geocode requests to OpenWeatherMap helpers."""

from __future__ import annotations

import os
from typing import Any

from modules.Tools.Base_Tools import geocode as geocode_module

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


_geocode_location = geocode_module.geocode_location
_require_api_key = getattr(geocode_module, "_require_api_key", None)


class OpenWeatherMapProvider(ToolProvider):
    """Adapter around :func:`modules.Tools.Base_Tools.geocode.geocode_location`."""

    async def call(self, **kwargs: Any) -> Any:
        location = kwargs.get("location")
        if not isinstance(location, str) or not location.strip():
            raise ValueError("'location' must be a non-empty string")
        return await _geocode_location(location)

    async def health_check(self) -> bool:
        if callable(_require_api_key):
            try:
                return bool(_require_api_key())
            except Exception:  # pragma: no cover - defensive guard
                return False
        return bool(os.getenv("OPENWEATHERMAP_API_KEY", "").strip())


tool_provider_registry.register("openweathermap", OpenWeatherMapProvider)

__all__ = ["OpenWeatherMapProvider"]

