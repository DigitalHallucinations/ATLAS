"""Helpers for polling NOAA/NWS weather alerts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import aiohttp

_API_ROOT = "https://api.weather.gov/alerts/active"
_DEFAULT_HEADERS = {
    "User-Agent": "ATLAS WeatherGenius (support@atlas)",
    "Accept": "application/geo+json",
}


class WeatherAlertPollingError(RuntimeError):
    """Raised when the NWS alert feed cannot be retrieved."""


async def _request_alert_feed(params: Dict[str, str]) -> Dict[str, Any]:
    """Execute the HTTP request against the NWS alert endpoint."""

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout, headers=_DEFAULT_HEADERS) as session:
        async with session.get(_API_ROOT, params=params) as response:
            if response.status != 200:
                detail = await _safe_error_message(response)
                raise WeatherAlertPollingError(
                    f"NWS alert feed request failed with status {response.status}: {detail}"
                )
            return await response.json()


async def _safe_error_message(response: aiohttp.ClientResponse) -> str:
    try:
        payload = await response.json()
    except Exception:  # pragma: no cover - defensive guard
        return response.reason or "unknown error"

    message = payload.get("title") or payload.get("detail") or "unknown error"
    return str(message)


def _normalize_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    properties = feature.get("properties", {})
    return {
        "id": properties.get("id") or feature.get("id"),
        "event": properties.get("event"),
        "headline": properties.get("headline"),
        "severity": properties.get("severity"),
        "urgency": properties.get("urgency"),
        "certainty": properties.get("certainty"),
        "area_description": properties.get("areaDesc"),
        "sent": properties.get("sent"),
        "effective": properties.get("effective"),
        "onset": properties.get("onset"),
        "expires": properties.get("expires"),
        "ends": properties.get("ends"),
        "instruction": properties.get("instruction"),
        "description": properties.get("description"),
        "response": properties.get("response"),
        "severity_level": properties.get("severity"),
        "source": properties.get("senderName"),
        "parameters": properties.get("parameters", {}),
        "raw": properties,
    }


async def weather_alert_feed(
    lat: float,
    lon: float,
    *,
    status: str = "actual",
    severity: Optional[str] = None,
    urgency: Optional[str] = None,
    certainty: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Fetch active NWS weather alerts near the provided coordinates.

    Args:
        lat: Latitude for the point query.
        lon: Longitude for the point query.
        status: Optional NWS status filter (defaults to ``"actual"``).
        severity: Optional severity filter (e.g., ``"moderate"``, ``"severe"``, ``"extreme"``).
        urgency: Optional urgency filter (e.g., ``"immediate"``, ``"expected"``).
        certainty: Optional certainty filter (e.g., ``"likely"``, ``"observed"``).
        limit: Maximum number of alerts to return (NWS default is 20).

    Returns:
        A dictionary containing the timestamp of the feed and a list of normalized alerts.
    """

    if limit <= 0:
        raise ValueError("limit must be a positive integer")

    params: Dict[str, str] = {
        "point": f"{lat},{lon}",
        "status": status,
        "limit": str(limit),
    }

    if severity:
        params["severity"] = severity
    if urgency:
        params["urgency"] = urgency
    if certainty:
        params["certainty"] = certainty

    payload = await _request_alert_feed(params)
    features: List[Dict[str, Any]] = payload.get("features", [])
    alerts = [_normalize_feature(feature) for feature in features]

    return {
        "updated": payload.get("updated") or payload.get("timestamp"),
        "title": payload.get("title"),
        "alerts": alerts,
        "query": params,
    }


__all__ = ["weather_alert_feed", "WeatherAlertPollingError"]
