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

class GoogleSearch:
    def __init__(self, api_key=SERPAPI_KEY):
        self.timeout = 10
        self.api_key = api_key
        

    async def _search(self, query: str) -> Tuple[int, Union[dict, str]]:
        """HTTP requests to SerpAPI."""
        
        logger.info("Starting search with term: %s", query)

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
            return response.status_code, response.json()
        except requests.HTTPError as http_err:
            logger.error("Error during the request: %s", http_err)
            status_code = (
                http_err.response.status_code if http_err.response else -1
            )
            return status_code, str(http_err)
        except Exception as e:
            logger.error("An error occurred: %s", e)
            return -1, str(e)
