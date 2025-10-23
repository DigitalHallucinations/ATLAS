"""Provider that proxies NCBI Entrez requests to the medical tools helpers."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from modules.Personas.MEDIC.Toolbox.medical_tools import search_pmc, search_pubmed

from .base import ToolProvider
from .registry import tool_provider_registry


_TOOL_DISPATCH: Dict[str, Callable[..., Awaitable[Any]]] = {
    "search_pubmed": search_pubmed,
    "search_pmc": search_pmc,
}


class NcbiEntrezProvider(ToolProvider):
    """Execute Entrez operations using the shared async helper functions."""

    async def call(self, **kwargs: Any) -> Any:
        handler = _TOOL_DISPATCH.get(self.tool_name)
        if handler is None:
            raise RuntimeError(f"Unsupported Entrez tool '{self.tool_name}'.")
        return await handler(**kwargs)


tool_provider_registry.register("ncbi_entrez", NcbiEntrezProvider)

__all__ = ["NcbiEntrezProvider"]

