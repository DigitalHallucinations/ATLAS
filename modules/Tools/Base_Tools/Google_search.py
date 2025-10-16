# modules/Tools/Base_Tools/Universal_Tools/Google_search.py

import asyncio
import os
from typing import Tuple, Union

import requests
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

config_manager = ConfigManager()
logger = setup_logger(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

DEFAULT_RESULTS_COUNT = 10


class GoogleSearch:
    def __init__(self, api_key=SERPAPI_KEY):
        self.timeout = 10
        self.api_key = api_key


    async def _search(
        self, query: str, k: Union[int, None] = None
    ) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to SerpAPI."""

        logger.info("Starting search with term: %s", query)

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
