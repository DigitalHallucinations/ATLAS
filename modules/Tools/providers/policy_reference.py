"""Built-in provider adapter for the ``policy_reference`` tool."""

from __future__ import annotations

import asyncio
from typing import Any

from modules.Tools.Base_Tools.policy_reference import policy_reference

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


class PolicyReferenceToolProvider(ToolProvider):
    """Provider that resolves policy lookups via the bundled knowledge base."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)

    async def call(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(policy_reference, **kwargs)

    async def health_check(self) -> bool:
        # The bundled policy dataset is static, so the provider is always healthy.
        return True


tool_provider_registry.register("builtin_policy", PolicyReferenceToolProvider)


__all__ = ["PolicyReferenceToolProvider"]

