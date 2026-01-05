"""HuggingFace Images provider (FLUX, Stable Diffusion models).

Uses the HuggingFace InferenceClient for text-to-image generation.
Leverages existing HuggingFace infrastructure from modules/Providers/HuggingFace/.

Supports:
- FLUX.1-dev, FLUX.1-schnell (Black Forest Labs)
- Stable Diffusion XL, SD3, SD3.5 (Stability AI)
- Other HuggingFace-hosted image models
"""

from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path
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
    from ATLAS.config import ConfigManager
    from PIL import Image as PILImage

logger = setup_logger(__name__)


class HuggingFaceImagesProvider(MediaProvider):
    """HuggingFace image generation provider using InferenceClient.

    Uses the text_to_image() method from huggingface_hub.InferenceClient.
    Follows patterns from modules/Providers/HuggingFace/components/huggingface_model_manager.py.
    """

    SUPPORTED_MODELS = [
        # FLUX models (Black Forest Labs)
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
        # Stable Diffusion models
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        # Other popular models
        "prompthero/openjourney-v4",
        "runwayml/stable-diffusion-v1-5",
    ]

    # Default model for fast generation
    DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize HuggingFace Images provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API key not configured.
            ImportError: If huggingface_hub not installed.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        api_key = config_manager.get_huggingface_api_key()
        if not api_key:
            raise ValueError("HuggingFace API key not configured")

        # Import InferenceClient
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub package required for HuggingFace Images provider. "
                "Install with: pip install huggingface_hub"
            ) from exc

        self.client = InferenceClient(token=api_key)
        self.logger.debug("HuggingFace Images provider initialized")

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "huggingface_images"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using HuggingFace Inference API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        model = request.model or self.DEFAULT_MODEL

        # Validate model (warn but proceed for custom models)
        if model not in self.SUPPORTED_MODELS:
            self.logger.warning(
                "Model %s not in known list, attempting anyway", model
            )

        # Resolve dimensions
        width, height = self._resolve_dimensions(request)

        # Build generation parameters
        gen_kwargs: Dict[str, Any] = {}

        if request.seed is not None:
            gen_kwargs["seed"] = request.seed

        # Negative prompt if style implies it
        negative_prompt = self._get_negative_prompt(request.style_preset)
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        self.logger.debug(
            "Generating image with model=%s, size=%dx%d, n=%d",
            model,
            width,
            height,
            request.n,
        )

        images: List[GeneratedImage] = []
        errors: List[str] = []

        # Generate n images (InferenceClient generates one at a time)
        for i in range(request.n):
            seed = request.seed + i if request.seed else None
            if seed is not None:
                gen_kwargs["seed"] = seed

            try:
                # Run synchronous text_to_image in thread pool
                result = await asyncio.to_thread(
                    self.client.text_to_image,
                    request.prompt,
                    model=model,
                    width=width,
                    height=height,
                    **gen_kwargs,
                )

                # Result is a PIL Image
                img = self._process_pil_image(result, request.output_format, seed)
                images.append(img)

            except Exception as exc:
                self.logger.error(
                    "HuggingFace image generation failed for image %d: %s",
                    i + 1,
                    exc,
                    exc_info=True,
                )
                errors.append(str(exc))

        if not images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error="; ".join(errors) if errors else "No images generated",
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,  # Filled by manager
            cost_estimate=self._estimate_cost(model, len(images)),
        )

    def _process_pil_image(
        self,
        pil_image: "PILImage.Image",
        output_format: OutputFormat,
        seed: Optional[int] = None,
    ) -> GeneratedImage:
        """Convert PIL Image to GeneratedImage.

        Args:
            pil_image: PIL Image from InferenceClient.
            output_format: Desired output format.
            seed: Random seed used (if known).

        Returns:
            GeneratedImage with appropriate data.
        """
        import base64

        image_id = str(uuid.uuid4())

        # Determine MIME type
        img_format = pil_image.format or "PNG"
        mime_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
        }
        mime = mime_map.get(img_format.upper(), "image/png")

        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=img_format.upper() if img_format else "PNG")
        image_bytes = buffer.getvalue()

        if output_format == OutputFormat.BASE64:
            return GeneratedImage(
                id=image_id,
                mime=mime,
                b64=base64.b64encode(image_bytes).decode("utf-8"),
                seed_used=seed,
            )
        elif output_format == OutputFormat.FILEPATH:
            # For now, just include base64; artifact store will handle saving
            return GeneratedImage(
                id=image_id,
                mime=mime,
                b64=base64.b64encode(image_bytes).decode("utf-8"),
                seed_used=seed,
            )
        else:  # URL - not available for local generation
            return GeneratedImage(
                id=image_id,
                mime=mime,
                b64=base64.b64encode(image_bytes).decode("utf-8"),
                seed_used=seed,
            )

    def _resolve_dimensions(self, request: ImageGenerationRequest) -> tuple[int, int]:
        """Resolve image dimensions from request.

        Args:
            request: Image generation request.

        Returns:
            Tuple of (width, height).
        """
        if request.size:
            try:
                w, h = request.size.split("x")
                return int(w), int(h)
            except ValueError:
                self.logger.warning("Invalid size format: %s", request.size)

        if request.aspect_ratio:
            return self._aspect_ratio_to_dimensions(request.aspect_ratio)

        # Default to 1024x1024
        return 1024, 1024

    def _aspect_ratio_to_dimensions(self, aspect_ratio: str) -> tuple[int, int]:
        """Convert aspect ratio to dimensions.

        Args:
            aspect_ratio: Ratio string (e.g., "16:9").

        Returns:
            Tuple of (width, height).
        """
        ratio_map = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "3:2": (1024, 683),
            "2:3": (683, 1024),
            "21:9": (1344, 576),
        }
        return ratio_map.get(aspect_ratio, (1024, 1024))

    def _get_negative_prompt(self, style_preset: Optional[str]) -> Optional[str]:
        """Get negative prompt for style preset.

        Args:
            style_preset: Style preset name.

        Returns:
            Negative prompt string or None.
        """
        if not style_preset:
            return None

        preset_negatives = {
            "photorealistic": "cartoon, anime, illustration, painting, drawing",
            "anime": "photorealistic, photo, realistic",
            "digital-art": "photo, photorealistic, blurry, low quality",
            "oil-painting": "photo, digital, 3d render",
            "watercolor": "photo, digital, sharp edges, photorealistic",
        }

        return preset_negatives.get(style_preset.lower())

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Estimate cost for HuggingFace inference.

        HuggingFace Inference API pricing varies by model and endpoint type.
        These are rough estimates for serverless inference.

        Args:
            model: Model identifier.
            n: Number of images.

        Returns:
            Cost estimate dictionary.
        """
        # Approximate per-image costs (serverless inference, as of 2025)
        model_costs = {
            "black-forest-labs/FLUX.1-dev": 0.03,
            "black-forest-labs/FLUX.1-schnell": 0.01,
            "stabilityai/stable-diffusion-xl-base-1.0": 0.008,
            "stabilityai/stable-diffusion-3-medium-diffusers": 0.015,
            "stabilityai/stable-diffusion-3.5-large": 0.025,
            "stabilityai/stable-diffusion-3.5-large-turbo": 0.015,
        }

        per_image = model_costs.get(model, 0.01)

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "count": n,
            "note": "Estimate based on serverless inference pricing",
        }

    async def health_check(self) -> bool:
        """Check if HuggingFace Inference API is accessible.

        Returns:
            True if API responds successfully.
        """
        try:
            # Use a lightweight model check
            await asyncio.to_thread(
                self.client.get_model_status,
                self.DEFAULT_MODEL,
            )
            return True
        except Exception as exc:
            self.logger.warning("HuggingFace health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Clean up resources."""
        # InferenceClient doesn't require explicit cleanup
        pass


# Factory function for registry
async def _create_huggingface_images(
    config_manager: "ConfigManager",
) -> HuggingFaceImagesProvider:
    """Factory function for creating HuggingFace Images provider."""
    return HuggingFaceImagesProvider(config_manager)


# Register on module import
register_provider("huggingface_images", _create_huggingface_images)
