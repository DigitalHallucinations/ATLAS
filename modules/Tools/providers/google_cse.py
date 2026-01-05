"""Google Programmable Search (Custom Search JSON API) provider."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, TYPE_CHECKING, Union

import requests

from modules.Tools.Base_Tools.Google_search import DEFAULT_RESULTS_COUNT

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


if TYPE_CHECKING:  # pragma: no cover - import used only for typing
    from core.config import ConfigManager


class _LazyConfigManagerAccessor:
    """Lazily instantiate and proxy access to :class:`ConfigManager`."""

    __slots__ = ("_manager",)

    def __init__(self) -> None:
        self._manager: Optional["ConfigManager"] = None

    def get_manager(self) -> "ConfigManager":
        if self._manager is None:
            from core.config import ConfigManager  # Local import avoids circular deps

            self._manager = ConfigManager()
        return self._manager

    def set_manager(self, manager: "ConfigManager") -> None:
        self._manager = manager


_CONFIG_MANAGER_ACCESSOR = _LazyConfigManagerAccessor()

ConfigManagerSource = Union["ConfigManager", _LazyConfigManagerAccessor]


class GoogleCseToolProvider(ToolProvider):
    """Provider that queries the Google Programmable Search JSON API."""

    _API_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"

    def __init__(
        self,
        spec: ToolProviderSpec,
        *,
        tool_name: str,
        fallback_callable=None,
        config_manager: Optional["ConfigManager"] = None,
    ) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)
        if config_manager is None:
            source: ConfigManagerSource = _CONFIG_MANAGER_ACCESSOR
        else:
            source = config_manager
        self._config_manager_source: ConfigManagerSource = source
        self._api_key_env = str(self.config.get("api_key_env") or "GOOGLE_API_KEY")
        self._api_key_config = str(self.config.get("api_key_config") or self._api_key_env)
        self._cse_id_env = str(self.config.get("cse_id_env") or "GOOGLE_CSE_ID")
        self._cse_id_config = str(self.config.get("cse_id_config") or self._cse_id_env)
        self._safe_search = self.config.get("safe")
        self._default_num = int(self.config.get("default_results", DEFAULT_RESULTS_COUNT))
        self._timeout = float(self.config.get("timeout", 10))

        self._api_key: str = self._resolve_secret(self._api_key_env, self._api_key_config)
        self._cse_id: str = self._resolve_secret(self._cse_id_env, self._cse_id_config)

    def _resolve_config_manager(self) -> Optional["ConfigManager"]:
        source = self._config_manager_source
        if isinstance(source, _LazyConfigManagerAccessor):
            manager = source.get_manager()
        else:
            manager = source
        self._config_manager_source = manager
        return manager

    def _resolve_secret(self, env_name: Optional[str], config_key: Optional[str]) -> str:
        candidate: Optional[str] = None
        manager = self._resolve_config_manager() if config_key else None
        if manager and config_key:
            getter = getattr(manager, "get_config", None)
            if getter:
                try:
                    candidate = getter(config_key)
                except Exception:  # pragma: no cover - defensive guard
                    candidate = None
        if not candidate and env_name:
            candidate = os.getenv(env_name)
        return (candidate or "").strip()

    async def call(self, **kwargs: Any) -> Any:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")

        k_value = kwargs.get("k")
        if k_value is None:
            num_results = self._default_num
        else:
            try:
                num_results = int(k_value)
            except (TypeError, ValueError):
                num_results = self._default_num
            else:
                if num_results <= 0:
                    num_results = self._default_num
        num_results = max(1, min(num_results, 10))

        if not self._api_key or not self._cse_id:
            raise RuntimeError("Google CSE credentials are not configured")

        params = {
            "key": self._api_key,
            "cx": self._cse_id,
            "q": query,
            "num": num_results,
        }
        if self._safe_search:
            params["safe"] = self._safe_search

        redacted_params = dict(params)
        if redacted_params.get("key"):
            redacted_params["key"] = "***REDACTED***"

        self.logger.info("Requesting %s with params: %s", self._API_ENDPOINT, redacted_params)

        def _perform_request() -> requests.Response:
            response = requests.get(self._API_ENDPOINT, params=params, timeout=self._timeout)
            response.raise_for_status()
            return response

        response = await asyncio.to_thread(_perform_request)
        data = response.json()
        return response.status_code, data

    async def health_check(self) -> bool:
        return bool(self._api_key and self._cse_id)


tool_provider_registry.register("google_cse", GoogleCseToolProvider)

__all__ = ["GoogleCseToolProvider"]
