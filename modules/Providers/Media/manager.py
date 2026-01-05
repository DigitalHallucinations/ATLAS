"""MediaProviderManager - Singleton orchestrator for image generation providers.

Follows the same patterns as ProviderManager for LLMs:
- Async singleton factory
- Provider switching and caching
- Invoker registration
- Health monitoring
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

from ATLAS.providers.base import register_invoker, get_invoker, build_result
from modules.logging.logger import setup_logger

from .base import (
    ImageGenerationRequest,
    ImageGenerationResult,
    MediaProvider,
)
from .registry import get_provider_factory, is_provider_registered

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager

logger = setup_logger(__name__)

# Type alias for media provider invokers
MediaProviderInvoker = Callable[
    ["MediaProviderManager", Callable[..., Awaitable[Any]], Dict[str, Any]],
    Awaitable[Any],
]

# Module-level singleton state
_media_provider_manager_instance: Optional["MediaProviderManager"] = None
_media_provider_manager_lock: Optional[asyncio.Lock] = None


async def get_media_provider_manager(
    config_manager: Optional["ConfigManager"] = None,
) -> "MediaProviderManager":
    """Get the global MediaProviderManager singleton.

    Creates and initializes the manager on first call.
    Subsequent calls return the same instance.

    Args:
        config_manager: Configuration manager (required on first call).

    Returns:
        Initialized MediaProviderManager instance.

    Raises:
        ValueError: If config_manager not provided on first call.
    """
    global _media_provider_manager_instance, _media_provider_manager_lock

    if _media_provider_manager_instance is not None:
        return _media_provider_manager_instance

    if _media_provider_manager_lock is None:
        _media_provider_manager_lock = asyncio.Lock()

    async with _media_provider_manager_lock:
        if _media_provider_manager_instance is None:
            if config_manager is None:
                raise ValueError(
                    "config_manager required for first MediaProviderManager initialization"
                )
            _media_provider_manager_instance = await MediaProviderManager.create(
                config_manager
            )

    return _media_provider_manager_instance


class MediaProviderManager:
    """Manages image generation providers following ProviderManager patterns.

    Provides:
    - Singleton pattern with async factory
    - Dynamic provider loading and switching
    - Health monitoring
    - Cost tracking hooks
    - Artifact storage integration

    Usage::

        manager = await MediaProviderManager.create(config_manager)
        result = await manager.generate_image(
            ImageGenerationRequest(prompt="A sunset over mountains")
        )
    """

    AVAILABLE_PROVIDERS = [
        "openai_images",
        "huggingface_images",
        "stability_ai",
        "stability",  # Alias for stability_ai
        "xai_aurora",
        "vertex_imagen",
        "gemini_images",
        "google_images",
        "imagen",
        "fal",
        "falai",
        "black_forest_labs",
        "runway",
        "ideogram",
        "replicate",
    ]

    _instance: Optional["MediaProviderManager"] = None
    _lock: Optional[asyncio.Lock] = None

    def __init__(self, config_manager: "ConfigManager"):
        """Private constructor - use create() factory method.

        Args:
            config_manager: ATLAS configuration manager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Provider state
        self.current_provider: Optional[str] = None
        self._providers: Dict[str, MediaProvider] = {}
        self._provider_health: Dict[str, bool] = {}

        # Artifact storage (lazy initialized)
        self._artifact_store: Optional[Any] = None

        # Configuration
        self._default_provider = self._get_default_provider()
        self._cost_tracking_enabled = self._get_cost_tracking_enabled()

        # Register invokers for each provider type
        self._register_invokers()

    @classmethod
    async def create(cls, config_manager: "ConfigManager") -> "MediaProviderManager":
        """Async factory method to create or retrieve singleton instance.

        Args:
            config_manager: ATLAS configuration manager.

        Returns:
            Initialized MediaProviderManager singleton.
        """
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if cls._instance is None:
                instance = cls(config_manager)
                await instance._initialize()
                cls._instance = instance
                logger.info("MediaProviderManager singleton created")
            return cls._instance

    async def _initialize(self) -> None:
        """Initialize the manager (called once during creation)."""
        # Pre-load default provider if configured
        if self._default_provider:
            try:
                await self._ensure_provider(self._default_provider)
                self.current_provider = self._default_provider
                logger.info("Initialized default media provider: %s", self._default_provider)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize default provider %s: %s",
                    self._default_provider,
                    exc,
                )

    def _get_default_provider(self) -> str:
        """Get default image provider from configuration."""
        image_settings = self._get_image_generation_settings()
        return image_settings.get("default_provider", "openai_images")

    def _get_cost_tracking_enabled(self) -> bool:
        """Check if cost tracking is enabled."""
        image_settings = self._get_image_generation_settings()
        return image_settings.get("cost_tracking_enabled", True)

    def _get_image_generation_settings(self) -> Dict[str, Any]:
        """Get image generation settings from config."""
        # Try to get from config, fall back to defaults
        try:
            settings = self.config_manager.get_config("IMAGE_GENERATION")
            if isinstance(settings, dict):
                return settings
        except Exception:
            pass

        return {
            "default_provider": "openai_images",
            "default_model": "gpt-image-1",
            "default_size": "1024x1024",
            "default_quality": "standard",
            "max_images_per_request": 4,
            "cost_tracking_enabled": True,
        }

    def _register_invokers(self) -> None:
        """Register invoker functions for media providers."""
        # OpenAI Images invoker
        register_invoker(
            "openai_images",
            lambda manager, func, kwargs: self._invoke_provider("openai_images", func, kwargs),
        )
        # HuggingFace Images invoker
        register_invoker(
            "huggingface_images",
            lambda manager, func, kwargs: self._invoke_provider("huggingface_images", func, kwargs),
        )
        # Additional providers can register their invokers here

    async def _invoke_provider(
        self,
        provider_name: str,
        func: Callable[..., Awaitable[Any]],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Generic provider invocation adapter.

        Args:
            provider_name: Provider identifier.
            func: Function to invoke on provider.
            kwargs: Arguments for the function.

        Returns:
            Result from the provider function.
        """
        provider = await self._ensure_provider(provider_name)
        return await func(provider, **kwargs)

    async def _ensure_provider(self, provider_name: str) -> MediaProvider:
        """Ensure a provider is initialized and return it.

        Args:
            provider_name: Provider identifier.

        Returns:
            Initialized MediaProvider instance.

        Raises:
            ValueError: If provider is not registered or creation fails.
        """
        if provider_name in self._providers:
            return self._providers[provider_name]

        factory = get_provider_factory(provider_name)
        if factory is None:
            # Try dynamic import based on provider name
            await self._import_provider_module(provider_name)
            factory = get_provider_factory(provider_name)

        if factory is None:
            raise ValueError(f"No factory registered for provider: {provider_name}")

        try:
            provider = await factory(self.config_manager)
            self._providers[provider_name] = provider
            self._provider_health[provider_name] = True
            logger.info("Initialized media provider: %s", provider_name)
            return provider
        except Exception as exc:
            self._provider_health[provider_name] = False
            logger.error("Failed to create provider %s: %s", provider_name, exc)
            raise ValueError(f"Failed to create provider {provider_name}: {exc}") from exc

    async def _import_provider_module(self, provider_name: str) -> None:
        """Dynamically import a provider module to trigger registration.

        Args:
            provider_name: Provider identifier.
        """
        module_map = {
            "openai_images": "modules.Providers.Media.OpenAIImages",
            "huggingface_images": "modules.Providers.Media.HuggingFace",
            # Add more as implemented
        }

        module_path = module_map.get(provider_name)
        if module_path:
            try:
                import importlib
                importlib.import_module(module_path)
                logger.debug("Imported provider module: %s", module_path)
            except ImportError as exc:
                logger.warning("Failed to import provider module %s: %s", module_path, exc)

    async def generate_image(
        self,
        request: ImageGenerationRequest,
        *,
        provider_override: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Generate image(s) using configured or specified provider.

        Args:
            request: Image generation parameters.
            provider_override: Explicitly select a provider (optional).

        Returns:
            ImageGenerationResult with generated images or error.
        """
        provider_name = provider_override or self._select_provider(request)

        try:
            provider = await self._ensure_provider(provider_name)
        except ValueError as exc:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=provider_name,
                model=request.model or "unknown",
                timing_ms=0,
                error=str(exc),
            )

        start_time = time.monotonic()
        try:
            result = await provider.generate_image(request)
        except Exception as exc:
            logger.error("Image generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=provider_name,
                model=request.model or provider.get_default_model(),
                timing_ms=int((time.monotonic() - start_time) * 1000),
                error=str(exc),
            )

        # Update timing
        result.timing_ms = int((time.monotonic() - start_time) * 1000)

        # Save artifacts if storage is configured
        if self._artifact_store and result.success:
            try:
                await self._save_artifacts(result, request)
            except Exception as exc:
                logger.warning("Failed to save image artifacts: %s", exc)

        return result

    async def edit_image(
        self,
        request: ImageGenerationRequest,
        *,
        provider_override: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Edit an existing image using configured or specified provider.

        Args:
            request: Edit request with input_images and optional mask.
            provider_override: Explicitly select a provider (optional).

        Returns:
            ImageGenerationResult with edited images or error.
        """
        provider_name = provider_override or self._select_provider(request)

        try:
            provider = await self._ensure_provider(provider_name)
        except ValueError as exc:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=provider_name,
                model=request.model or "unknown",
                timing_ms=0,
                error=str(exc),
            )

        start_time = time.monotonic()
        try:
            result = await provider.edit_image(request)
        except Exception as exc:
            logger.error("Image editing failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=provider_name,
                model=request.model or provider.get_default_model(),
                timing_ms=int((time.monotonic() - start_time) * 1000),
                error=str(exc),
            )

        result.timing_ms = int((time.monotonic() - start_time) * 1000)
        return result

    def _select_provider(self, request: ImageGenerationRequest) -> str:
        """Select provider based on request characteristics and routing rules.

        Args:
            request: Image generation request.

        Returns:
            Provider identifier.
        """
        metadata = request.metadata or {}
        prompt_lower = request.prompt.lower()

        # Intent-based routing
        if metadata.get("intent") == "typography" or any(
            kw in prompt_lower for kw in ["logo", "text", "typography", "poster", "sign"]
        ):
            if self._is_available("ideogram"):
                return "ideogram"

        if metadata.get("compliance_required") or metadata.get("enterprise"):
            if self._is_available("vertex_imagen"):
                return "vertex_imagen"

        if metadata.get("intent") == "creative_studio":
            if self._is_available("runway"):
                return "runway"

        # Quality-based routing
        if request.quality == "draft":
            if self._is_available("fal"):
                return "fal"

        # Default fallback
        return self.current_provider or self._default_provider or "openai_images"

    def _is_available(self, provider_name: str) -> bool:
        """Check if a provider is available and healthy.

        Args:
            provider_name: Provider identifier.

        Returns:
            True if provider is available.
        """
        # Check if already initialized and healthy
        if provider_name in self._providers:
            return self._provider_health.get(provider_name, False)

        # Check if registered
        return is_provider_registered(provider_name)

    async def _save_artifacts(
        self,
        result: ImageGenerationResult,
        request: ImageGenerationRequest,
    ) -> None:
        """Save generated images to artifact storage.

        Args:
            result: Generation result with images.
            request: Original request for metadata.
        """
        if self._artifact_store is None:
            return

        await self._artifact_store.save(result, request)

    async def switch_provider(self, provider_name: str) -> bool:
        """Switch the current default provider.

        Args:
            provider_name: New provider identifier.

        Returns:
            True if switch successful.
        """
        if provider_name not in self.AVAILABLE_PROVIDERS:
            logger.warning("Unknown provider: %s", provider_name)
            return False

        try:
            await self._ensure_provider(provider_name)
            self.current_provider = provider_name
            logger.info("Switched to media provider: %s", provider_name)
            return True
        except Exception as exc:
            logger.error("Failed to switch provider: %s", exc)
            return False

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all initialized providers.

        Returns:
            Dict mapping provider names to health status.
        """
        results = {}
        for name, provider in self._providers.items():
            try:
                healthy = await provider.health_check()
                self._provider_health[name] = healthy
                results[name] = healthy
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", name, exc)
                self._provider_health[name] = False
                results[name] = False
        return results

    def get_available_models(self, provider_name: Optional[str] = None) -> List[str]:
        """Get list of available models.

        Args:
            provider_name: Specific provider (optional, defaults to current).

        Returns:
            List of model identifiers.
        """
        name = provider_name or self.current_provider
        if name and name in self._providers:
            return self._providers[name].supported_models
        return []

    async def close(self) -> None:
        """Clean up all provider resources."""
        for name, provider in self._providers.items():
            try:
                await provider.close()
                logger.debug("Closed provider: %s", name)
            except Exception as exc:
                logger.warning("Error closing provider %s: %s", name, exc)

        self._providers.clear()
        self._provider_health.clear()
        logger.info("MediaProviderManager closed")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        global _media_provider_manager_instance
        cls._instance = None
        _media_provider_manager_instance = None
