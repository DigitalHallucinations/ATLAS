"""Provider that delegates execution to the runtime fallback callable."""

from __future__ import annotations

import inspect
from typing import Any

from .base import ToolProvider
from .registry import tool_provider_registry


class AtlasRuntimeToolProvider(ToolProvider):
    """Execute built-in tools through the provider routing framework."""

    async def call(self, **kwargs: Any) -> Any:
        if self.fallback_callable is None:
            raise RuntimeError(
                f"Tool '{self.tool_name}' is missing its runtime fallback callable."
            )

        result = self.fallback_callable(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def health_check(self) -> bool:
        return self.fallback_callable is not None


tool_provider_registry.register("atlas_runtime", AtlasRuntimeToolProvider)

__all__ = ["AtlasRuntimeToolProvider"]

