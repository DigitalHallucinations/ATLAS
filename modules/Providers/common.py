"""Shared helpers for provider generator caching."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar
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
