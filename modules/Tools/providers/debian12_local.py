"""Provider that proxies calendar operations to the Debian 12 helper."""

from __future__ import annotations

from typing import Any

from modules.Tools.Base_Tools.debian12_calendar import debian12_calendar

from .base import ToolProvider
from .registry import tool_provider_registry


class Debian12LocalProvider(ToolProvider):
    """Adapter around :func:`modules.Tools.Base_Tools.debian12_calendar.debian12_calendar`."""

    async def call(self, **kwargs: Any) -> Any:
        return await debian12_calendar(**kwargs)


tool_provider_registry.register("debian12_local", Debian12LocalProvider)

__all__ = ["Debian12LocalProvider"]

