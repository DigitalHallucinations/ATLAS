"""Anthropic adapter wrapping HTTP discovery and caching."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


async def list_models(
    config_manager,
    model_manager,
    logger,
    *,
    base_url: Optional[str] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Discover Anthropic models using stored credentials."""

    getter = getattr(config_manager, "get_anthropic_api_key", None)
    if not callable(getter):
        logger.error("Configuration backend does not expose an Anthropic API key accessor.")
        return {
            "models": [],
            "error": "Anthropic credentials are unavailable.",
            "base_url": base_url,
        }

    api_key = getter() or ""
    if not api_key:
        return {
            "models": [],
            "error": "Anthropic API key is not configured.",
            "base_url": base_url,
        }

    effective_base_url = (base_url or "https://api.anthropic.com").rstrip("/")
    endpoint = f"{effective_base_url}/v1/models"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "accept": "application/json",
    }

    def _fetch_models() -> Any:
        request = Request(endpoint, headers=headers, method="GET")
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - trusted URL built from config
            encoding = response.headers.get_content_charset("utf-8")
            payload = response.read().decode(encoding)
        return json.loads(payload)

    try:
        raw_response = await asyncio.to_thread(_fetch_models)
    except HTTPError as exc:
        detail = f"HTTP {exc.code}: {exc.reason}"
        try:
            body = exc.read()
            if body:
                detail = f"{detail} - {body.decode('utf-8', 'ignore')}"
        except Exception:  # pragma: no cover - best effort logging
            pass
        logger.error("Anthropic model listing failed with HTTP error: %s", detail, exc_info=True)
        return {
            "models": [],
            "error": detail,
            "base_url": effective_base_url,
        }
    except URLError as exc:
        detail = getattr(exc, "reason", None) or str(exc)
        logger.error("Anthropic model listing failed with network error: %s", detail, exc_info=True)
        return {
            "models": [],
            "error": str(detail),
            "base_url": effective_base_url,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unexpected error while listing Anthropic models: %s", exc, exc_info=True)
        return {
            "models": [],
            "error": str(exc),
            "base_url": effective_base_url,
        }

    entries: List[Any] = []
    if isinstance(raw_response, dict):
        entries = raw_response.get("data") or raw_response.get("models") or []
    elif isinstance(raw_response, list):
        entries = raw_response
    else:
        data = getattr(raw_response, "data", None)
        if isinstance(data, list):
            entries = data

    seen: set[str] = set()
    discovered: List[str] = []

    for entry in entries:
        model_id: Optional[str] = None
        if isinstance(entry, str):
            model_id = entry
        elif isinstance(entry, dict):
            candidate: Any = entry.get("id") or entry.get("model") or entry.get("name")
            if candidate is not None:
                model_id = str(candidate)
        else:
            for attr in ("id", "model", "name"):
                value = getattr(entry, attr, None)
                if value is not None:
                    model_id = str(value)
                    break

        if not model_id:
            continue

        normalized = model_id.strip()
        if not normalized or normalized in seen:
            continue

        discovered.append(normalized)
        seen.add(normalized)

    cached_models: List[str] = list(discovered)
    try:
        cached_models = model_manager.update_models_for_provider("Anthropic", discovered)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to update cached Anthropic models after discovery: %s",
            exc,
            exc_info=True,
        )

    if cached_models:
        logger.info("Retrieved %d Anthropic model(s) via discovery.", len(cached_models))
    else:
        logger.info("Anthropic model discovery returned no models.")

    return {
        "models": cached_models,
        "error": None,
        "base_url": effective_base_url,
        "source": "anthropic",
    }
