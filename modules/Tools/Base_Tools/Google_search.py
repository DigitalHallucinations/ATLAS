# modules/Tools/Base_Tools/Universal_Tools/Google_search.py

import asyncio
import os
from typing import Tuple, Union

import requests
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

config_manager = ConfigManager()
logger = setup_logger(__name__)

DEFAULT_RESULTS_COUNT = 10


class GoogleSearch:
    def __init__(self, api_key: Union[str, None] = None):
        self.timeout = 10
        if api_key is not None:
            resolved_key = api_key
        else:
            resolved_key = config_manager.get_config("SERPAPI_KEY")
            if not resolved_key:
                resolved_key = os.getenv("SERPAPI_KEY")

        self.api_key = (resolved_key or "").strip()


    async def _search(
        self, query: str, k: Union[int, None] = None
    ) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to SerpAPI."""

        logger.info("Starting search with term: %s", query)

        if not self.api_key:
            message = (
                "SerpAPI key is not configured. Set SERPAPI_KEY in your environment "
                "or configuration to enable Google search."
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

        url = f"https://serpapi.com/search?q={query}&api_key={self.api_key}"
        logger.info("URL: %s", url)

        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
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
