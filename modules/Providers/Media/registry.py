"""Media provider registry for dynamic provider registration.

This module provides a factory pattern for registering and retrieving
image generation providers. New providers register themselves on import.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from core.config import ConfigManager
    from .base import MediaProvider

logger = setup_logger(__name__)

# Type alias for provider factory functions
ProviderFactory = Callable[["ConfigManager"], Awaitable["MediaProvider"]]

# Registry of provider factories
_PROVIDER_FACTORIES: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory function.

    Args:
        name: Provider identifier (e.g., 'openai_images').
        factory: Async factory function that creates a MediaProvider instance.

    Example::

        async def create_openai_images(config_manager):
            return OpenAIImagesProvider(config_manager)

        register_provider("openai_images", create_openai_images)
    """
    if name in _PROVIDER_FACTORIES:
        logger.warning("Overwriting existing provider factory: %s", name)
    _PROVIDER_FACTORIES[name] = factory
    logger.debug("Registered media provider factory: %s", name)


def get_provider_factory(name: str) -> Optional[ProviderFactory]:
    """Retrieve a registered provider factory.

    Args:
        name: Provider identifier.

    Returns:
        Factory function if registered, None otherwise.
    """
    return _PROVIDER_FACTORIES.get(name)


def list_registered_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        List of registered provider identifiers.
    """
    return list(_PROVIDER_FACTORIES.keys())


def is_provider_registered(name: str) -> bool:
    """Check if a provider is registered.

    Args:
        name: Provider identifier.

    Returns:
        True if provider is registered.
    """
    return name in _PROVIDER_FACTORIES


def unregister_provider(name: str) -> bool:
    """Remove a provider from the registry.

    Args:
        name: Provider identifier.

    Returns:
        True if provider was removed, False if not found.
    """
    if name in _PROVIDER_FACTORIES:
        del _PROVIDER_FACTORIES[name]
        logger.debug("Unregistered media provider: %s", name)
        return True
    return False


def clear_registry() -> None:
    """Clear all registered providers (mainly for testing)."""
    _PROVIDER_FACTORIES.clear()
    logger.debug("Cleared media provider registry")
