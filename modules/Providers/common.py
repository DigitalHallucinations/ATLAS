"""Shared helpers for provider generator caching and cleanup."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, TypeVar
from weakref import WeakKeyDictionary

from ATLAS.config import ConfigManager
from ATLAS.model_manager import ModelManager


TGenerator = TypeVar("TGenerator")


def get_or_create_generator(
    config_manager: ConfigManager,
    cache: "WeakKeyDictionary[ConfigManager, TGenerator]",
    factory: Callable[[ConfigManager, Optional[ModelManager]], TGenerator],
    model_manager: Optional[ModelManager] = None,
) -> TGenerator:
    """Return a cached generator instance, creating it when missing.

    Args:
        config_manager: Configuration manager used as cache key.
        cache: Weak reference cache mapping configuration managers to generators.
        factory: Callable used to create a new generator instance when needed.
        model_manager: Optional model manager instance to attach to the generator.

    Returns:
        The cached or newly created generator instance.
    """

    generator = cache.get(config_manager)
    if generator is None:
        generator = factory(config_manager, model_manager)
        cache[config_manager] = generator
    elif model_manager is not None and hasattr(generator, "model_manager") and getattr(
        generator, "model_manager", None
    ) is not model_manager:
        generator.model_manager = model_manager
    return generator


async def close_client(client: Any, logger: Any, label: str) -> None:
    """Attempt to close an SDK client gracefully, logging on failure."""

    if client is None:
        return

    async_close = getattr(client, "aclose", None)
    sync_close = getattr(client, "close", None) if not callable(async_close) else None

    try:
        if callable(async_close):
            maybe_result = async_close()
            if inspect.isawaitable(maybe_result):
                await maybe_result
        elif callable(sync_close):
            maybe_result = sync_close()
            if inspect.isawaitable(maybe_result):
                await maybe_result
    except Exception as exc:  # pragma: no cover - defensive cleanup
        if logger is not None:
            logger.warning(
                "Failed to close %s client cleanly: %s", label, exc, exc_info=True
            )
