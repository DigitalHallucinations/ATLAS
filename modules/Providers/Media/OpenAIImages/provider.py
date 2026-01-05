"""OpenAI Images provider (DALL-E, GPT-Image models).

Supports:
- DALL-E 2: Lower cost, 256x256 to 1024x1024
- DALL-E 3: Higher quality, 1024x1024 to 1792x1024, style options
- GPT-Image-1: Latest model with generation + editing
- GPT-Image-1-Mini: Faster, lower cost variant
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger
from modules.Providers.Media.base import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    MediaProvider,
    OutputFormat,
)
from modules.Providers.Media.registry import register_provider

if TYPE_CHECKING:
    from core.config import ConfigManager

logger = setup_logger(__name__)


class OpenAIImagesProvider(MediaProvider):
    """OpenAI image generation provider (DALL-E 2/3, GPT-Image-1).

    Uses the OpenAI Python SDK for image generation.
    Follows patterns from modules/Providers/OpenAI/OA_gen_response.py.
    """

    SUPPORTED_MODELS = [
        "dall-e-2",
        "dall-e-3",
        "gpt-image-1",
        "gpt-image-1-mini",
    ]

    # Size constraints per model
    MODEL_SIZES = {
        "dall-e-2": ["256x256", "512x512", "1024x1024"],
        "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"],
        "gpt-image-1": ["1024x1024", "1024x1792", "1792x1024", "auto"],
        "gpt-image-1-mini": ["1024x1024", "1024x1792", "1792x1024", "auto"],
    }

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize OpenAI Images provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API key not configured.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        api_key = config_manager.get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not configured")

        # Get OpenAI settings for base_url, etc.
        settings = config_manager.get_openai_llm_settings()
        client_kwargs: Dict[str, Any] = {"api_key": api_key}

        base_url = settings.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url

        organization = settings.get("organization")
        if organization:
            client_kwargs["organization"] = organization

        # Import AsyncOpenAI
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for OpenAI Images provider. "
                "Install with: pip install openai"
            ) from exc

        self.client = AsyncOpenAI(**client_kwargs)
        self.logger.debug("OpenAI Images provider initialized")

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "openai_images"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using OpenAI's images API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        model = request.model or "gpt-image-1"
        if model not in self.SUPPORTED_MODELS:
            self.logger.warning("Unknown model %s, defaulting to gpt-image-1", model)
            model = "gpt-image-1"

        # Build API request
        api_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": request.n,
        }

        # Size handling
        size = self._resolve_size(request, model)
        api_kwargs["size"] = size

        # Quality (DALL-E 3 / GPT-Image)
        if request.quality and model in ("dall-e-3", "gpt-image-1", "gpt-image-1-mini"):
            quality_map = {"draft": "standard", "standard": "standard", "hd": "hd"}
            api_kwargs["quality"] = quality_map.get(request.quality, "standard")

        # Style (DALL-E 3 only)
        if request.style_preset and model == "dall-e-3":
            if request.style_preset in ("vivid", "natural"):
                api_kwargs["style"] = request.style_preset

        # Response format
        if request.output_format == OutputFormat.URL:
            api_kwargs["response_format"] = "url"
        else:
            api_kwargs["response_format"] = "b64_json"

        self.logger.debug(
            "Generating image with model=%s, size=%s, n=%d",
            model,
            size,
            request.n,
        )

        try:
            response = await self.client.images.generate(**api_kwargs)
        except Exception as exc:
            self.logger.error("OpenAI image generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error=str(exc),
            )

        # Parse response
        images: List[GeneratedImage] = []
        for item in response.data:
            img = GeneratedImage(
                id=str(uuid.uuid4()),
                mime="image/png",
                url=getattr(item, "url", None),
                b64=getattr(item, "b64_json", None),
                seed_used=None,  # OpenAI doesn't expose seed
                revised_prompt=getattr(item, "revised_prompt", None),
            )
            images.append(img)

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,  # Filled by manager
            cost_estimate=self._estimate_cost(model, request.n, size),
        )

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using OpenAI's images/edits API.

        Args:
            request: Edit request with input_images and optional mask.

        Returns:
            ImageGenerationResult with edited images or error.
        """
        if not request.input_images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="dall-e-2",
                timing_ms=0,
                error="Input image required for editing.",
            )

        # Prepare image file
        image_path = request.input_images[0]

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as exc:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="dall-e-2",
                timing_ms=0,
                error=f"Failed to read input image: {exc}",
            )

        api_kwargs: Dict[str, Any] = {
            "image": image_data,
            "prompt": request.prompt,
            "n": request.n,
            "size": self._resolve_size(request, "dall-e-2"),
        }

        # Add mask if provided
        if request.mask_image:
            try:
                with open(request.mask_image, "rb") as f:
                    api_kwargs["mask"] = f.read()
            except Exception as exc:
                self.logger.warning("Failed to read mask image: %s", exc)

        # Response format
        if request.output_format == OutputFormat.URL:
            api_kwargs["response_format"] = "url"
        else:
            api_kwargs["response_format"] = "b64_json"

        try:
            response = await self.client.images.edit(**api_kwargs)
        except Exception as exc:
            self.logger.error("OpenAI image edit failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="dall-e-2",
                timing_ms=0,
                error=str(exc),
            )

        images: List[GeneratedImage] = []
        for item in response.data:
            img = GeneratedImage(
                id=str(uuid.uuid4()),
                mime="image/png",
                url=getattr(item, "url", None),
                b64=getattr(item, "b64_json", None),
            )
            images.append(img)

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model="dall-e-2",
            timing_ms=0,
            cost_estimate=self._estimate_cost("dall-e-2", request.n, api_kwargs["size"]),
        )

    def _resolve_size(self, request: ImageGenerationRequest, model: str) -> str:
        """Resolve image size from request parameters.

        Args:
            request: Image generation request.
            model: Model identifier.

        Returns:
            Size string (e.g., "1024x1024").
        """
        if request.size:
            allowed = self.MODEL_SIZES.get(model, ["1024x1024"])
            if request.size in allowed:
                return request.size
            self.logger.warning(
                "Size %s not allowed for %s, using default", request.size, model
            )

        if request.aspect_ratio:
            return self._aspect_ratio_to_size(request.aspect_ratio, model)

        # Default sizes per model
        if model == "dall-e-2":
            return "1024x1024"
        return "1024x1024"

    def _aspect_ratio_to_size(self, aspect_ratio: str, model: str) -> str:
        """Convert aspect ratio to supported size.

        Args:
            aspect_ratio: Ratio string (e.g., "16:9").
            model: Model identifier.

        Returns:
            Closest matching size.
        """
        ratio_map = {
            "1:1": "1024x1024",
            "16:9": "1792x1024",
            "9:16": "1024x1792",
            "4:3": "1024x1024",
            "3:4": "1024x1024",
            "3:2": "1792x1024",
            "2:3": "1024x1792",
        }

        size = ratio_map.get(aspect_ratio, "1024x1024")

        # Validate for model
        allowed = self.MODEL_SIZES.get(model, ["1024x1024"])
        if size not in allowed:
            return allowed[0]

        return size

    def _estimate_cost(
        self, model: str, n: int, size: Optional[str]
    ) -> Dict[str, Any]:
        """Estimate cost based on model and parameters.

        Args:
            model: Model identifier.
            n: Number of images.
            size: Image size.

        Returns:
            Cost estimate dictionary.
        """
        # Approximate pricing (as of 2025-2026, update as needed)
        base_costs = {
            "dall-e-2": {
                "256x256": 0.016,
                "512x512": 0.018,
                "1024x1024": 0.020,
            },
            "dall-e-3": {
                "1024x1024": 0.040,
                "1024x1792": 0.080,
                "1792x1024": 0.080,
            },
            "gpt-image-1": {
                "1024x1024": 0.040,
                "1024x1792": 0.080,
                "1792x1024": 0.080,
                "auto": 0.060,
            },
            "gpt-image-1-mini": {
                "1024x1024": 0.020,
                "1024x1792": 0.040,
                "1792x1024": 0.040,
                "auto": 0.030,
            },
        }

        model_costs = base_costs.get(model, {})
        per_image = model_costs.get(size or "1024x1024", 0.040)

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "size": size,
            "count": n,
        }

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible.

        Returns:
            True if API responds successfully.
        """
        try:
            # List models as a lightweight health check
            await asyncio.wait_for(
                self.client.models.list(),
                timeout=10.0,
            )
            return True
        except Exception as exc:
            self.logger.warning("OpenAI health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Clean up client resources."""
        if hasattr(self.client, "close"):
            await self.client.close()


# Factory function for registry
async def _create_openai_images(config_manager: "ConfigManager") -> OpenAIImagesProvider:
    """Factory function for creating OpenAI Images provider."""
    return OpenAIImagesProvider(config_manager)


# Register on module import
register_provider("openai_images", _create_openai_images)
