"""Provider exposing the key-value store tool operations."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional

from modules.Tools.Base_Tools.kv_store import build_kv_store_service
from modules.Tools.providers.base import ToolProvider, ToolProviderSpec
from modules.Tools.providers.registry import tool_provider_registry

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from modules.Tools.Base_Tools.kv_store import KeyValueStoreService


class KeyValueStoreProvider(ToolProvider):
    """Provider implementation routing requests to the KV store service."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        adapter_name = str(self.config.get("adapter", "postgres")).strip() or "postgres"
        adapter_config: Optional[Mapping[str, Any]]
        raw_config = self.config.get("adapter_config")
        if isinstance(raw_config, Mapping):
            adapter_config = MappingProxyType(dict(raw_config))
        else:
            adapter_config = None

        self._adapter_name = adapter_name
        self._adapter_config = adapter_config
        self._config_manager: Optional["ConfigManager"] = None
        self._service: Optional["KeyValueStoreService"] = None

    @staticmethod
    def _resolve_config_manager() -> Optional["ConfigManager"]:
        try:
            from ATLAS.config import ConfigManager as _ConfigManager  # Local import to avoid circular dependency
        except Exception:  # pragma: no cover - ConfigManager unavailable in some environments
            return None
        try:
            return _ConfigManager()
        except Exception:  # pragma: no cover - defensive guard around ConfigManager instantiation
            return None

    def _get_service(self) -> "KeyValueStoreService":
        if self._service is None:
            if self._config_manager is None:
                self._config_manager = self._resolve_config_manager()
            self._service = build_kv_store_service(
                adapter_name=self._adapter_name,
                adapter_config=self._adapter_config,
                config_manager=self._config_manager,
            )
        return self._service

    async def call(self, **kwargs: Any) -> Mapping[str, Any]:
        service = self._get_service()
        namespace = str(kwargs.get("namespace", "")).strip()
        key = str(kwargs.get("key", "")).strip()
        if not namespace:
            raise ValueError("namespace must be provided")
        if not key:
            raise ValueError("key must be provided")

        if self.tool_name == "kv_get":
            return await service.get_value(namespace, key)
        if self.tool_name == "kv_set":
            if "value" not in kwargs:
                raise ValueError("value must be provided for kv_set")
            ttl = kwargs.get("ttl_seconds")
            return await service.set_value(namespace, key, kwargs["value"], ttl_seconds=ttl)
        if self.tool_name == "kv_delete":
            return await service.delete_value(namespace, key)
        if self.tool_name == "kv_increment":
            delta = kwargs.get("delta", 1)
            ttl = kwargs.get("ttl_seconds")
            initial = kwargs.get("initial_value", 0)
            return await service.increment_value(
                namespace,
                key,
                delta=int(delta),
                ttl_seconds=ttl,
                initial_value=int(initial),
            )
        raise RuntimeError(f"Unsupported KV store operation '{self.tool_name}'")


tool_provider_registry.register("kv_store_postgres", KeyValueStoreProvider)


__all__ = ["KeyValueStoreProvider"]
