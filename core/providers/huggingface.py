"""Async helpers for interacting with Hugging Face provider utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.Providers.HuggingFace.HF_gen_response import (
    HuggingFaceGenerator,
    clear_cache as _hf_clear_cache,
    download_model as _hf_download_model,
    search_models as _hf_search_models,
    update_model_settings as _hf_update_model_settings,
)

from .base import ResultPayload, build_result

EnsureReady = Callable[[], ResultPayload]
GeneratorSupplier = Callable[[], Optional[HuggingFaceGenerator]]


def _require_generator(supplier: GeneratorSupplier) -> HuggingFaceGenerator:
    generator = supplier()
    if generator is None:
        raise RuntimeError("HuggingFace generator is not initialized.")
    return generator


async def search_models(
    ensure_ready: EnsureReady,
    generator_supplier: GeneratorSupplier,
    search_query: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    logger: Optional[logging.Logger] = None,
) -> ResultPayload:
    """Search the Hugging Face hub for models."""

    ensure_result = ensure_ready()
    if not ensure_result.get("success"):
        return ensure_result

    try:
        generator = _require_generator(generator_supplier)
        results = await _hf_search_models(generator, search_query, filters, limit)
        return build_result(True, message="Search completed.", data=results)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.error(
                "Failed to search HuggingFace models '%s': %s", search_query, exc, exc_info=True
            )
        return build_result(False, error=str(exc))


async def download_model(
    ensure_ready: EnsureReady,
    generator_supplier: GeneratorSupplier,
    model_id: str,
    *,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> ResultPayload:
    """Download a Hugging Face model to the local cache."""

    ensure_result = ensure_ready()
    if not ensure_result.get("success"):
        return ensure_result

    try:
        generator = _require_generator(generator_supplier)
        await _hf_download_model(generator, model_id, force)
        message = f"Model '{model_id}' downloaded successfully."
        return build_result(True, message=message)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.error(
                "Failed to download HuggingFace model %s: %s", model_id, exc, exc_info=True
            )
        return build_result(False, error=str(exc))


async def update_settings(
    ensure_ready: EnsureReady,
    generator_supplier: GeneratorSupplier,
    settings: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> ResultPayload:
    """Persist Hugging Face model settings."""

    ensure_result = ensure_ready()
    if not ensure_result.get("success"):
        return ensure_result

    try:
        generator = _require_generator(generator_supplier)
        updated = await asyncio.to_thread(_hf_update_model_settings, generator, settings)
        return build_result(True, data=updated, message="Settings updated successfully.")
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.error("Failed to update HuggingFace settings: %s", exc, exc_info=True)
        return build_result(False, error=str(exc))


async def clear_cache(
    ensure_ready: EnsureReady,
    generator_supplier: GeneratorSupplier,
    *,
    logger: Optional[logging.Logger] = None,
) -> ResultPayload:
    """Clear cached Hugging Face artefacts."""

    ensure_result = ensure_ready()
    if not ensure_result.get("success"):
        return ensure_result

    try:
        generator = _require_generator(generator_supplier)
        await asyncio.to_thread(_hf_clear_cache, generator)
        return build_result(True, message="HuggingFace cache cleared.")
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.error("Failed to clear HuggingFace cache: %s", exc, exc_info=True)
        return build_result(False, error=str(exc))
