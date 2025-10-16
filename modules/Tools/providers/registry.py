"""Registry utilities for tool providers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from modules.logging.logger import setup_logger

from .base import ToolProvider, ToolProviderSpec

ProviderFactory = Callable[[ToolProviderSpec, str, Optional[Any]], ToolProvider]


class ToolProviderRegistry:
    """Registry of available tool providers keyed by name."""

    def __init__(self) -> None:
        self._factories: Dict[str, ProviderFactory] = {}
        self._logger = setup_logger(__name__)

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def register(self, name: str, factory: Callable[[ToolProviderSpec, str, Optional[Any]], ToolProvider]) -> None:
        key = self._normalize(name)
        if isinstance(factory, type) and issubclass(factory, ToolProvider):
            def _constructor(spec: ToolProviderSpec, tool_name: str, fallback_callable: Optional[Any]) -> ToolProvider:
                return factory(spec, tool_name=tool_name, fallback_callable=fallback_callable)

            self._factories[key] = _constructor
        else:
            self._factories[key] = factory  # type: ignore[assignment]
        self._logger.debug("Registered tool provider '%s'", key)

    def unregister(self, name: str) -> None:
        key = self._normalize(name)
        if key in self._factories:
            self._logger.debug("Unregistered tool provider '%s'", key)
        self._factories.pop(key, None)

    def create(
        self,
        spec: ToolProviderSpec,
        *,
        tool_name: str,
        fallback_callable: Optional[Any] = None,
    ) -> ToolProvider:
        key = self._normalize(spec.name)
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"Unknown tool provider: {spec.name}")
        return factory(spec, tool_name, fallback_callable)

    @contextmanager
    def temporary_provider(
        self,
        name: str,
        factory: Callable[[ToolProviderSpec, str, Optional[Any]], ToolProvider],
    ) -> Iterator[None]:
        self.register(name, factory)
        try:
            yield
        finally:
            self.unregister(name)


tool_provider_registry = ToolProviderRegistry()


__all__ = ["ToolProviderRegistry", "tool_provider_registry", "ProviderFactory"]
