"""Shared helpers for NCBI Entrez tooling."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Tuple

import requests

from core.config import ConfigManager
from modules.logging.logger import setup_logger

config_manager = ConfigManager()
logger = setup_logger(__name__)

_API_KEY = (config_manager.get_config("NCBI_API_KEY") or os.getenv("NCBI_API_KEY") or "").strip()
_CONTACT_EMAIL = (
    config_manager.get_config("NCBI_API_EMAIL")
    or os.getenv("NCBI_API_EMAIL")
    or ""
).strip()

# NCBI allows up to 10 requests/second with an API key and ~3 without.
_RATE_LIMIT_WITH_KEY = 0.11
_RATE_LIMIT_WITHOUT_KEY = 0.34

_rate_lock = asyncio.Lock()
_last_request_at = 0.0

_DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        f"ATLAS-medical-tools/1.0 (+mailto:{_CONTACT_EMAIL})"
        if _CONTACT_EMAIL
        else "ATLAS-medical-tools/1.0 (contact: set NCBI_API_EMAIL)"
    ),
}


def get_auth_params() -> Dict[str, str]:
    """Return API parameters required for authenticated Entrez calls."""

    if _API_KEY:
        return {"api_key": _API_KEY}
    return {}


async def enforce_rate_limit() -> None:
    """Sleep just long enough to satisfy NCBI's rate limits."""

    global _last_request_at

    min_interval = _RATE_LIMIT_WITH_KEY if _API_KEY else _RATE_LIMIT_WITHOUT_KEY
    async with _rate_lock:
        elapsed = time.monotonic() - _last_request_at
        delay = max(0.0, min_interval - elapsed)
        if delay > 0:
            await asyncio.sleep(delay)
        _last_request_at = time.monotonic()


async def perform_entrez_request(
    url: str,
    params: Dict[str, Any],
    *,
    timeout: float = 15.0,
) -> Tuple[int, Dict[str, Any]]:
    """Execute an HTTP GET against the Entrez API with standard handling.

    Returns a tuple of ``(status_code, payload)`` where payload is the parsed
    JSON document on success or an ``{"error": message}`` structure on
    failure.
    """

    await enforce_rate_limit()

    merged_params = {**params, **get_auth_params()}
    sanitized_params = merged_params.copy()
    if "api_key" in sanitized_params:
        sanitized_params["api_key"] = "***REDACTED***"

    logger.info("Requesting %s with params: %s", url, sanitized_params)

    try:
        response = await asyncio.to_thread(
            requests.get,
            url,
            params=merged_params,
            headers=_DEFAULT_HEADERS,
            timeout=timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - exercised in tests
        logger.error("Entrez request failed: %s", exc)
        return -1, {"error": str(exc)}

    status_code = response.status_code

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Entrez returned an HTTP error: %s", exc)
        message = str(exc)
        if exc.response is not None:
            try:
                payload = exc.response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                message = payload.get("error") or message
        return status_code, {"error": message}

    try:
        payload = response.json()
    except ValueError:
        logger.error("Entrez response was not valid JSON")
        return status_code, {"error": "Entrez returned an invalid JSON payload."}

    if not isinstance(payload, dict):
        logger.error("Unexpected Entrez payload type: %s", type(payload))
        return status_code, {"error": "Entrez returned an unexpected response structure."}

    return status_code, payload
