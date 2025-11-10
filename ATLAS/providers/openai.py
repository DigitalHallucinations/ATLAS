"""OpenAI adapter wrapping HTTP discovery and caching."""

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
    organization: Optional[str] = None,
    timeout: float = 15.0,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover OpenAI models using stored credentials."""

    resolved_settings: Dict[str, Any] = {}
    if settings is not None:
        resolved_settings = dict(settings)
    else:
        settings_getter = getattr(config_manager, "get_openai_llm_settings", None)
        if callable(settings_getter):
            try:
                raw = settings_getter()
                if isinstance(raw, dict):
                    resolved_settings = raw
            except Exception:  # pragma: no cover - best effort settings retrieval
                logger.debug(
                    "Failed to load cached OpenAI settings for discovery.",
                    exc_info=True,
                )

    configured_base_url = (
        resolved_settings.get("base_url") if isinstance(resolved_settings, dict) else None
    )
    configured_org = (
        resolved_settings.get("organization")
        if isinstance(resolved_settings, dict)
        else None
    )

    effective_base_url = (base_url if base_url is not None else configured_base_url) or "https://api.openai.com/v1"
    effective_org = organization if organization is not None else configured_org

    getter = getattr(config_manager, "get_openai_api_key", None)
    if not callable(getter):
        logger.error("Configuration backend does not expose an OpenAI API key accessor.")
        return {
            "models": [],
            "error": "OpenAI credentials are unavailable.",
            "base_url": effective_base_url,
            "organization": effective_org,
        }

    api_key = getter() or ""
    if not api_key:
        return {
            "models": [],
            "error": "OpenAI API key is not configured.",
            "base_url": effective_base_url,
            "organization": effective_org,
        }

    endpoint = f"{effective_base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if effective_org:
        headers["OpenAI-Organization"] = effective_org

    def _fetch_models() -> Dict[str, Any]:
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
        logger.error("OpenAI model listing failed with HTTP error: %s", detail, exc_info=True)
        return {
            "models": [],
            "error": detail,
            "base_url": effective_base_url,
            "organization": effective_org,
        }
    except URLError as exc:
        detail = getattr(exc, "reason", None) or str(exc)
        logger.error("OpenAI model listing failed with network error: %s", detail, exc_info=True)
        return {
            "models": [],
            "error": str(detail),
            "base_url": effective_base_url,
            "organization": effective_org,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unexpected error while listing OpenAI models: %s", exc, exc_info=True)
        return {
            "models": [],
            "error": str(exc),
            "base_url": effective_base_url,
            "organization": effective_org,
        }

    entries: List[Any] = []
    if isinstance(raw_response, dict):
        data = raw_response.get("data")
        if isinstance(data, list):
            entries = data
    elif isinstance(raw_response, list):
        entries = raw_response

    models: List[str] = []
    for entry in entries:
        model_id: Optional[str] = None
        if isinstance(entry, dict):
            model_id = entry.get("id")
        else:
            model_id = getattr(entry, "id", None)

        if isinstance(model_id, str) and model_id:
            models.append(model_id)

    unique_models = sorted(set(models))
    prioritized = [
        name
        for name in unique_models
        if any(token in name for token in ("gpt", "omni", "o1", "o3", "chat"))
    ]
    if prioritized:
        prioritized_set = set(prioritized)
        trailing = [name for name in unique_models if name not in prioritized_set]
        unique_models = prioritized + trailing

    if unique_models:
        try:
            model_manager.update_models_for_provider("OpenAI", unique_models)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to update cached OpenAI models after discovery: %s",
                exc,
                exc_info=True,
            )

    return {
        "models": unique_models,
        "error": None,
        "base_url": effective_base_url,
        "organization": effective_org,
    }
