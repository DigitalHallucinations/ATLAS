"""Current location tool wrapper."""

from __future__ import annotations

from typing import Any, Dict

from modules.logging.logger import setup_logger
from modules.Tools.location_services.ip_api import get_current_location as _service_get_current_location


logger = setup_logger(__name__)


def _error(message: str) -> Dict[str, str]:
    """Return a normalized error payload while logging the failure."""
    logger.error(message)
    return {"error": message}


async def get_current_location() -> Dict[str, Any]:
    """Return the caller's approximate current location using IP geolocation."""

    try:
        result = await _service_get_current_location()
    except Exception as exc:  # pragma: no cover - unexpected provider failure
        logger.exception("Unexpected error while retrieving current location: %s", exc)
        return _error("Failed to retrieve current location from the provider.")

    if not isinstance(result, dict):
        return _error("Unexpected response from current location provider.")

    if "error" in result:
        error_message = str(result.get("error", "Unknown error"))
        logger.error("Current location provider returned an error: %s", error_message)
        return {"error": error_message}

    return result


__all__ = ["get_current_location"]
