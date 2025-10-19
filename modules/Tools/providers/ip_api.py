"""Provider that returns the caller's approximate location via IP-API."""

from __future__ import annotations

from typing import Any

from modules.Tools.Base_Tools.current_location import get_current_location

from .base import ToolProvider
from .registry import tool_provider_registry


class IpApiProvider(ToolProvider):
    """Adapter for :func:`modules.Tools.Base_Tools.current_location.get_current_location`."""

    async def call(self, **kwargs: Any) -> Any:  # noqa: ARG002 - kwargs for API parity
        return await get_current_location()


tool_provider_registry.register("ip-api", IpApiProvider)

__all__ = ["IpApiProvider"]

