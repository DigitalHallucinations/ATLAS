"""High-level geocoding tool wrapper."""

from __future__ import annotations

import os
from typing import Any, Dict

from modules.logging.logger import setup_logger
from modules.Tools.location_services.geocode import (
    geocode_location as _service_geocode_location,
)


logger = setup_logger(__name__)


def _error(message: str) -> Dict[str, str]:
    """Return a normalized error payload while logging the failure."""
    logger.error(message)
    return {"error": message}


def _require_api_key() -> str:
    """Return the configured OpenWeatherMap API key or an empty string."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    return api_key


async def geocode_location(location: str) -> Dict[str, Any]:
    """Geocode the supplied location name using the configured provider."""

    sanitized = (location or "").strip()
    if not sanitized:
        return _error("A location value is required to perform geocoding.")

    api_key = _require_api_key()
    if not api_key:
        return _error(
            "OpenWeatherMap API key is required to geocode locations. "
            "Set the OPENWEATHERMAP_API_KEY environment variable."
        )

    try:
        result = await _service_geocode_location(sanitized)
    except Exception as exc:  # pragma: no cover - unexpected provider failure
        logger.exception("Unexpected error during geocoding request: %s", exc)
        return _error("Failed to retrieve geocoding information from the provider.")

    if not isinstance(result, dict):
        return _error("Unexpected response from geocoding provider.")

    if "error" in result:
        error_message = str(result.get("error", "Unknown error"))
        logger.error("Geocoding provider returned an error: %s", error_message)
        return {"error": error_message}

    return result


__all__ = ["geocode_location"]
