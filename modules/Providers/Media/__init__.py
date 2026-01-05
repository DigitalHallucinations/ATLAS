"""Media generation providers for ATLAS.

This module provides image generation capabilities through multiple providers:
- OpenAI (DALL-E, GPT-Image)
- HuggingFace (FLUX, Stable Diffusion)
- Black Forest Labs (FLUX pro/dev/schnell direct API)
- xAI Aurora (Grok-2-Image)
- Stability AI
- Google Imagen/Gemini
- fal.ai (aggregator)

Usage::

    from modules.Providers.Media import MediaProviderManager
    
    manager = await MediaProviderManager.create(config_manager)
    result = await manager.generate_image(
        ImageGenerationRequest(prompt="A serene mountain landscape at sunset")
    )
"""

from .base import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    MediaProvider,
    OutputFormat,
)
from .registry import get_provider_factory, register_provider
from .manager import MediaProviderManager, get_media_provider_manager

__all__ = [
    # Base interfaces
    "GeneratedImage",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "MediaProvider",
    "OutputFormat",
    # Registry
    "get_provider_factory",
    "register_provider",
    # Manager
    "MediaProviderManager",
    "get_media_provider_manager",
]
