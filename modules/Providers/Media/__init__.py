"""Media generation providers for ATLAS.

This module provides image generation capabilities through multiple providers:
- OpenAI (DALL-E, GPT-Image)
- HuggingFace (FLUX, Stable Diffusion)
- Black Forest Labs (FLUX pro/dev/schnell direct API)
- xAI Aurora (Grok-2-Image)
- Stability AI (Stable Diffusion, SDXL, SD3.5)
- Google Imagen/Gemini
- fal.ai (aggregator)
- Replicate (FLUX, SDXL, and many open models)
- Ideogram (text-in-image specialist)
- Runway (Gen-3 Alpha creative tools)

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

# Import providers to trigger registration
from . import OpenAIImages  # noqa: F401
from . import HuggingFace  # noqa: F401
from . import BlackForestLabs  # noqa: F401
from . import XAI  # noqa: F401
from . import Stability  # noqa: F401
from . import Google  # noqa: F401
from . import FalAI  # noqa: F401
from . import Replicate  # noqa: F401
from . import Ideogram  # noqa: F401
from . import Runway  # noqa: F401

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
