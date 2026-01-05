"""Black Forest Labs provider for FLUX image generation.

Black Forest Labs provides direct API access to FLUX models:
- FLUX.1-pro: Highest quality, production-grade
- FLUX.1-dev: Development/testing
- FLUX.1-schnell: Fast generation

API Documentation: https://docs.bfl.ml/
"""

from __future__ import annotations

import asyncio
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


class BlackForestLabsProvider(MediaProvider):
    """Black Forest Labs image generation provider (FLUX models).

    Uses the BFL API directly for FLUX model access.
    Requires BFL_API_KEY environment variable.
    """

    SUPPORTED_MODELS = [
        "flux-pro-1.1",
        "flux-pro",
        "flux-dev",
        "flux-schnell",
        "flux-pro-1.1-ultra",
        "flux-pro-1.1-fill",  # Inpainting
        "flux-pro-1.1-canny",  # ControlNet
        "flux-pro-1.1-depth",  # ControlNet
    ]

    # Model endpoints
    MODEL_ENDPOINTS = {
        "flux-pro-1.1": "https://api.bfl.ml/v1/flux-pro-1.1",
        "flux-pro-1.1-ultra": "https://api.bfl.ml/v1/flux-pro-1.1-ultra",
        "flux-pro": "https://api.bfl.ml/v1/flux-pro",
        "flux-dev": "https://api.bfl.ml/v1/flux-dev",
        "flux-schnell": "https://api.bfl.ml/v1/flux-schnell",
        "flux-pro-1.1-fill": "https://api.bfl.ml/v1/flux-pro-1.1-fill",
        "flux-pro-1.1-canny": "https://api.bfl.ml/v1/flux-pro-1.1-canny",
        "flux-pro-1.1-depth": "https://api.bfl.ml/v1/flux-pro-1.1-depth",
    }

    DEFAULT_MODEL = "flux-pro-1.1"

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize Black Forest Labs provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API key not configured.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Get API key from config or environment
        self.api_key = self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "Black Forest Labs API key not configured. "
                "Set BFL_API_KEY environment variable."
            )

        self._client = None
        self.logger.debug("Black Forest Labs provider initialized")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os

        # Try config first
        try:
            key = self.config_manager.get_config("BFL_API_KEY")
            if key:
                return key
        except Exception:
            pass

        # Fall back to environment
        return os.environ.get("BFL_API_KEY")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                headers={
                    "X-Key": self.api_key,
                    "Content-Type": "application/json",
                }
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "black_forest_labs"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Black Forest Labs FLUX API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        model = request.model or self.DEFAULT_MODEL
        if model not in self.SUPPORTED_MODELS:
            self.logger.warning("Unknown model %s, using %s", model, self.DEFAULT_MODEL)
            model = self.DEFAULT_MODEL

        endpoint = self.MODEL_ENDPOINTS.get(model)
        if not endpoint:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                error=f"No endpoint configured for model: {model}",
            )

        # Build request payload
        payload = self._build_payload(request, model)

        images: List[GeneratedImage] = []
        errors: List[str] = []

        # Generate n images
        for i in range(request.n):
            if request.seed is not None:
                payload["seed"] = request.seed + i

            try:
                result = await self._generate_single(endpoint, payload, model)
                if result:
                    images.append(result)
            except Exception as exc:
                self.logger.error(
                    "FLUX generation failed for image %d: %s", i + 1, exc, exc_info=True
                )
                errors.append(str(exc))

        if not images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                error="; ".join(errors) if errors else "No images generated",
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,
            cost_estimate=self._estimate_cost(model, len(images), request.size),
        )

    def _build_payload(
        self, request: ImageGenerationRequest, model: str
    ) -> Dict[str, Any]:
        """Build API request payload."""
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
        }

        # Handle dimensions
        if request.size:
            width, height = self._parse_size(request.size)
            payload["width"] = width
            payload["height"] = height
        elif request.aspect_ratio:
            width, height = self._aspect_ratio_to_dims(request.aspect_ratio)
            payload["width"] = width
            payload["height"] = height
        else:
            payload["width"] = 1024
            payload["height"] = 1024

        # Seed for reproducibility
        if request.seed is not None:
            payload["seed"] = request.seed

        # Model-specific options
        if model in ("flux-pro-1.1", "flux-pro-1.1-ultra", "flux-pro"):
            # Pro models support guidance
            if request.guidance_scale is not None:
                payload["guidance"] = request.guidance_scale

        # Steps (schnell uses fewer by default)
        if request.num_inference_steps is not None:
            payload["steps"] = request.num_inference_steps

        # Safety tolerance (1-6, higher = more permissive)
        if request.safety:
            payload["safety_tolerance"] = request.safety.get("tolerance", 2)

        # Output format
        payload["output_format"] = "png"

        return payload

    async def _generate_single(
        self, endpoint: str, payload: Dict[str, Any], model: str
    ) -> Optional[GeneratedImage]:
        """Generate a single image via the BFL API.

        BFL uses an async job pattern:
        1. POST to create job
        2. Poll for result
        """
        client = await self._get_client()

        # Submit generation job
        async with client.post(endpoint, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"BFL API error ({resp.status}): {error_text}")

            job_data = await resp.json()
            job_id = job_data.get("id")

            if not job_id:
                raise RuntimeError("No job ID returned from BFL API")

        # Poll for result
        result_url = "https://api.bfl.ml/v1/get_result"
        max_attempts = 60  # 60 seconds max
        poll_interval = 1.0

        for _ in range(max_attempts):
            await asyncio.sleep(poll_interval)

            async with client.get(result_url, params={"id": job_id}) as resp:
                if resp.status != 200:
                    continue

                result = await resp.json()
                status = result.get("status")

                if status == "Ready":
                    # Get the image URL
                    image_url = result.get("result", {}).get("sample")
                    if image_url:
                        return GeneratedImage(
                            id=str(uuid.uuid4()),
                            mime="image/png",
                            url=image_url,
                            seed_used=result.get("result", {}).get("seed"),
                        )
                    break

                elif status == "Error":
                    error_msg = result.get("result", {}).get("error", "Unknown error")
                    raise RuntimeError(f"BFL generation failed: {error_msg}")

                elif status in ("Pending", "Processing"):
                    continue

        raise RuntimeError("BFL generation timed out")

    def _parse_size(self, size: str) -> tuple[int, int]:
        """Parse size string to width, height."""
        try:
            w, h = size.lower().split("x")
            return int(w), int(h)
        except Exception:
            return 1024, 1024

    def _aspect_ratio_to_dims(self, aspect_ratio: str) -> tuple[int, int]:
        """Convert aspect ratio to dimensions."""
        ratio_map = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "21:9": (1536, 640),
        }
        return ratio_map.get(aspect_ratio, (1024, 1024))

    def _estimate_cost(
        self, model: str, n: int, size: Optional[str]
    ) -> Dict[str, Any]:
        """Estimate generation cost."""
        # BFL pricing (approximate, per image)
        base_costs = {
            "flux-pro-1.1": 0.04,
            "flux-pro-1.1-ultra": 0.06,
            "flux-pro": 0.05,
            "flux-dev": 0.025,
            "flux-schnell": 0.003,
            "flux-pro-1.1-fill": 0.05,
            "flux-pro-1.1-canny": 0.05,
            "flux-pro-1.1-depth": 0.05,
        }

        per_image = base_costs.get(model, 0.04)

        # Megapixel scaling for some models
        if size:
            width, height = self._parse_size(size)
            megapixels = (width * height) / 1_000_000
            if megapixels > 1:
                per_image *= megapixels

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "size": size,
        }

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using FLUX fill model (inpainting)."""
        if not request.input_images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="flux-pro-1.1-fill",
                error="Input image required for editing.",
            )

        # Use the fill model for inpainting
        request.model = "flux-pro-1.1-fill"

        # For inpainting, we need to load and encode the image
        # This is a simplified implementation
        return await super().edit_image(request)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Register provider factory
async def _create_provider(config_manager: "ConfigManager") -> BlackForestLabsProvider:
    return BlackForestLabsProvider(config_manager)


register_provider("black_forest_labs", _create_provider)
