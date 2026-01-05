# modules/Tools/Base_Tools/Universal_Tools/Google_search.py

import asyncio
import os
from typing import Tuple, Union, TYPE_CHECKING

import requests
from modules.logging.logger import setup_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from core.config import ConfigManager  # pylint: disable=ungrouped-imports

logger = setup_logger(__name__)


class _LazyConfigManagerAccessor:
    """Lazily instantiate and proxy access to :class:`ConfigManager`."""

    __slots__ = ("_manager",)

    def __init__(self) -> None:
        self._manager = None

    def _ensure_manager(self):
        if self._manager is None:
            from core.config import ConfigManager  # Local import avoids circulars

            self._manager = ConfigManager()
        return self._manager

    def set_manager(self, manager) -> None:
        self._manager = manager

    def get_manager(self):
        return self._ensure_manager()

    def __getattr__(self, item):  # pragma: no cover - exercised indirectly
        manager = self._ensure_manager()
        return getattr(manager, item)


config_manager = _LazyConfigManagerAccessor()

DEFAULT_RESULTS_COUNT = 10


class GoogleSearch:
    _KEY_PREFERENCE = ("GOOGLE_API_KEY", "SERPAPI_KEY")

    def __init__(self, api_key: Union[str, None] = None, config_manager=None):
        self.timeout = 10
        self._config_source = config_manager
        if api_key is not None:
            resolved_key = api_key
        else:
            resolved_key = self._resolve_configured_key()

        self.api_key = (resolved_key or "").strip()

    def _resolve_configured_key(self) -> Union[str, None]:
        manager = self._resolve_config_manager()
        getter = getattr(manager, "get_config", None)
        for key_name in self._KEY_PREFERENCE:
            candidate = getter(key_name) if getter else None
            if not candidate:
                candidate = os.getenv(key_name)
            if candidate:
                logger.debug("Resolved Google Search key from %s", key_name)
                return candidate
        return None

    def _resolve_config_manager(self):
        source = self._config_source
        if source is None:
            source = config_manager

        if isinstance(source, _LazyConfigManagerAccessor):
            manager = source.get_manager()
        else:
            manager = source

        self._config_source = manager
        return manager


    async def _search(
        self, query: str, k: Union[int, None] = None
    ) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to SerpAPI."""

        logger.info("Starting search with term: %s", query)

        if not self.api_key:
            message = (
                "Google Search API key is not configured. Set GOOGLE_API_KEY (preferred) "
                "or SERPAPI_KEY in your environment or configuration to enable Google search."
            )
            logger.error(message)
            return -1, message

        if k is None:
            k = DEFAULT_RESULTS_COUNT
        else:
            try:
                k = int(k)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid 'k' value provided (%s); falling back to default %d",
                    k,
                    DEFAULT_RESULTS_COUNT,
                )
                k = DEFAULT_RESULTS_COUNT
            else:
                if k <= 0:
                    logger.warning(
                        "Non-positive 'k' value provided (%d); falling back to default %d",
                        k,
                        DEFAULT_RESULTS_COUNT,
                    )
                    k = DEFAULT_RESULTS_COUNT

        url = "https://serpapi.com/search"
        params = {"q": query, "api_key": self.api_key}
        sanitized_params = params.copy()
        if sanitized_params.get("api_key"):
            sanitized_params["api_key"] = "***REDACTED***"
        logger.info("Requesting %s with params: %s", url, sanitized_params)

        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
                params=params,
                timeout=self.timeout,
            )
            logger.info("Response status code: %d", response.status_code)
            response.raise_for_status()

            data = response.json()

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        data[key] = value[:k]

            return response.status_code, data
        except requests.HTTPError as http_err:
            logger.error("Error during the request: %s", http_err)
            status_code = (
                http_err.response.status_code if http_err.response else -1
            )
            return status_code, str(http_err)
        except Exception as e:
            logger.error("An error occurred: %s", e)
            return -1, str(e)
