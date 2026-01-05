"""fal.ai image generation provider aggregator.

Provides access to 600+ generative media models through fal.ai's unified API.
Supports FLUX, Stable Diffusion, Recraft, and many more models.

Authentication:
- FAL_KEY environment variable

Models supported:
- fal-ai/flux/dev - FLUX.1 Dev (high quality)
- fal-ai/flux/schnell - FLUX.1 Schnell (fast, 1-4 steps)
- fal-ai/flux-pro/v1.1-ultra - FLUX 1.1 Pro Ultra (2K resolution)
- fal-ai/recraft-v3 - Recraft V3 (SOTA quality)
- fal-ai/stable-diffusion-v35-large - SD 3.5 Large
- And many more at https://fal.ai/models?categories=text-to-image
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import base64

from modules.logging.logger import setup_logger

from ..base import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    MediaProvider,
    OutputFormat,
)

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ..manager import MediaProviderManager

logger = setup_logger(__name__)


# Model pricing (approximate USD per image)
MODEL_PRICING = {
    # FLUX models
    "fal-ai/flux/dev": 0.025,
    "fal-ai/flux/schnell": 0.003,
    "fal-ai/flux-pro": 0.05,
    "fal-ai/flux-pro/v1.1": 0.055,
    "fal-ai/flux-pro/v1.1-ultra": 0.06,
    "fal-ai/flux-realism": 0.025,
    "fal-ai/flux-lora": 0.025,
    # Recraft
    "fal-ai/recraft-v3": 0.04,
    # Stable Diffusion
    "fal-ai/stable-diffusion-v35-large": 0.03,
    "fal-ai/fast-sdxl": 0.01,
    "fal-ai/sdxl-lightning": 0.005,
    # Other popular models
    "fal-ai/ideogram/v2": 0.08,
    "fal-ai/aura-flow": 0.02,
}


class FalAIProvider(MediaProvider):
    """Image generation provider using fal.ai's unified API.

    Provides access to 600+ generative media models through a single API.
    """

    POPULAR_MODELS = [
        "fal-ai/flux/dev",
        "fal-ai/flux/schnell",
        "fal-ai/flux-pro/v1.1-ultra",
        "fal-ai/flux-realism",
        "fal-ai/recraft-v3",
        "fal-ai/stable-diffusion-v35-large",
        "fal-ai/fast-sdxl",
        "fal-ai/ideogram/v2",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize fal.ai provider.

        Args:
            api_key: fal.ai API key (FAL_KEY).
            output_dir: Directory to save generated images.
        """
        self._api_key = api_key or os.environ.get("FAL_KEY", "")
        self._output_dir = output_dir or os.path.join(
            os.path.expanduser("~"), ".atlas", "media", "generated", "falai"
        )
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "falai"

    @property
    def supported_models(self) -> List[str]:
        return self.POPULAR_MODELS

    async def _ensure_client(self) -> Any:
        """Lazy initialize fal client."""
        if self._client is not None:
            return self._client

        try:
            import fal_client

            # Configure with API key
            if self._api_key:
                os.environ["FAL_KEY"] = self._api_key

            self._client = fal_client
            return self._client
        except ImportError:
            raise ImportError(
                "fal-client package required. Install with: "
                "pip install fal-client"
            )

    def _normalize_model_id(self, model: str) -> str:
        """Normalize model identifier to fal.ai format."""
        # If already in fal.ai format, return as-is
        if model.startswith("fal-ai/"):
            return model

        # Common aliases
        aliases = {
            "flux-dev": "fal-ai/flux/dev",
            "flux-schnell": "fal-ai/flux/schnell",
            "flux-pro": "fal-ai/flux-pro/v1.1-ultra",
            "flux-realism": "fal-ai/flux-realism",
            "recraft-v3": "fal-ai/recraft-v3",
            "sdxl": "fal-ai/fast-sdxl",
            "sd3.5-large": "fal-ai/stable-diffusion-v35-large",
            "ideogram": "fal-ai/ideogram/v2",
        }

        return aliases.get(model.lower(), f"fal-ai/{model}")

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using fal.ai API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images.
        """
        model = self._normalize_model_id(request.model or "flux/dev")
        start_time = time.monotonic()

        try:
            client = await self._ensure_client()

            # Build input parameters
            input_params: Dict[str, Any] = {
                "prompt": request.prompt,
            }

            # Add optional parameters based on model capabilities
            if request.n and request.n > 1:
                input_params["num_images"] = min(request.n, 4)

            if request.seed is not None:
                input_params["seed"] = request.seed

            if request.aspect_ratio:
                input_params["aspect_ratio"] = request.aspect_ratio
            elif request.size:
                # Parse size into dimensions
                try:
                    w, h = map(int, request.size.lower().split("x"))
                    input_params["image_size"] = {"width": w, "height": h}
                except ValueError:
                    pass

            if request.style_preset:
                input_params["style"] = request.style_preset

            # For img2img operations
            if request.input_images:
                input_params["image_url"] = request.input_images[0]

            # Run generation (subscribe waits for result)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.subscribe(model, arguments=input_params),
            )

            # Process result
            images = self._process_result(result, request)
            timing_ms = int((time.monotonic() - start_time) * 1000)

            return ImageGenerationResult(
                success=True,
                images=images,
                provider=self.name,
                model=model,
                timing_ms=timing_ms,
                cost_estimate=self._estimate_cost(model, len(images)),
            )

        except Exception as exc:
            logger.error("fal.ai generation failed: %s", exc, exc_info=True)
            timing_ms = int((time.monotonic() - start_time) * 1000)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=timing_ms,
                error=str(exc),
            )

    def _process_result(
        self, result: Dict[str, Any], request: ImageGenerationRequest
    ) -> List[GeneratedImage]:
        """Process fal.ai result into GeneratedImage objects."""
        images: List[GeneratedImage] = []

        # fal.ai returns images in different formats depending on model
        image_data_list = result.get("images", []) or result.get("image", [])

        # Handle single image response
        if isinstance(image_data_list, dict):
            image_data_list = [image_data_list]

        for i, img_data in enumerate(image_data_list):
            image_id = str(uuid.uuid4())[:8]

            url = None
            filepath = None
            mime = "image/png"

            if isinstance(img_data, dict):
                url = img_data.get("url")
                mime = img_data.get("content_type", "image/png")

                # If we need a file, download it
                if request.output_format == OutputFormat.FILEPATH and url:
                    filepath = self._download_image(url, image_id, mime)
            elif isinstance(img_data, str):
                # Could be URL or base64
                if img_data.startswith("http"):
                    url = img_data
                    if request.output_format == OutputFormat.FILEPATH:
                        filepath = self._download_image(url, image_id, mime)
                else:
                    # Assume base64
                    filepath = self._save_base64(img_data, image_id, mime)

            # Get seed if available
            seed_used = result.get("seed")

            images.append(
                GeneratedImage(
                    id=image_id,
                    mime=mime,
                    path=filepath,
                    url=url,
                    seed_used=seed_used,
                    revised_prompt=result.get("prompt"),
                )
            )

        return images

    def _download_image(self, url: str, image_id: str, mime: str) -> str:
        """Download image from URL to local file."""
        import urllib.request

        ext = mime.split("/")[-1] if "/" in mime else "png"
        filename = f"fal_{image_id}.{ext}"
        filepath = os.path.join(self._output_dir, filename)

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as exc:
            logger.warning("Failed to download image: %s", exc)
            return ""

        return filepath

    def _save_base64(self, b64_data: str, image_id: str, mime: str) -> str:
        """Save base64 image data to file."""
        ext = mime.split("/")[-1] if "/" in mime else "png"
        filename = f"fal_{image_id}.{ext}"
        filepath = os.path.join(self._output_dir, filename)

        try:
            image_bytes = base64.b64decode(b64_data)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
        except Exception as exc:
            logger.warning("Failed to save base64 image: %s", exc)
            return ""

        return filepath

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Calculate cost estimate for generation."""
        per_image = MODEL_PRICING.get(model, 0.025)
        total = per_image * n

        return {
            "estimated_usd": total,
            "per_image_usd": per_image,
            "model": model,
            "count": n,
        }

    async def health_check(self) -> bool:
        """Check provider availability."""
        if not self._api_key:
            return False

        try:
            await self._ensure_client()
            return True
        except Exception as exc:
            logger.warning("fal.ai health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Clean up resources."""
        self._client = None


async def _create_falai_provider(config_manager: "ConfigManager") -> FalAIProvider:
    """Factory function for creating FalAI provider.
    
    Args:
        config_manager: ATLAS configuration manager.
        
    Returns:
        Initialized FalAIProvider instance.
    """
    return FalAIProvider(
        api_key=os.environ.get("FAL_KEY"),
    )


def register_with_registry() -> None:
    """Register fal.ai provider with the provider registry."""
    from modules.Providers.Media.registry import register_provider
    
    register_provider("falai", _create_falai_provider)
    register_provider("fal", _create_falai_provider)
