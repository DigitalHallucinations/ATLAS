"""Stability AI provider for image generation.

Stability AI provides multiple image generation services:
- Stable Image Ultra: Highest quality, photorealistic (8 credits)
- Stable Image Core: Fast and affordable (3 credits)
- SD 3.5 Large/Medium/Turbo/Flash: Direct model access

API Documentation: https://platform.stability.ai/docs/api-reference
"""

from __future__ import annotations

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
    from ATLAS.config import ConfigManager

logger = setup_logger(__name__)


class StabilityAIProvider(MediaProvider):
    """Stability AI image generation provider.

    Supports Stable Image Ultra, Core, and SD3.5 models.
    Requires STABILITY_API_KEY environment variable.
    """

    SUPPORTED_MODELS = [
        # Stable Image services
        "stable-image-ultra",
        "stable-image-core",
        # SD 3.5 variants
        "sd3.5-large",
        "sd3.5-large-turbo",
        "sd3.5-medium",
        "sd3.5-flash",
        # Legacy
        "stable-diffusion-xl-1024-v1-0",
    ]

    # Model to endpoint mapping
    MODEL_ENDPOINTS = {
        "stable-image-ultra": "/v2beta/stable-image/generate/ultra",
        "stable-image-core": "/v2beta/stable-image/generate/core",
        "sd3.5-large": "/v2beta/stable-image/generate/sd3",
        "sd3.5-large-turbo": "/v2beta/stable-image/generate/sd3",
        "sd3.5-medium": "/v2beta/stable-image/generate/sd3",
        "sd3.5-flash": "/v2beta/stable-image/generate/sd3",
        "stable-diffusion-xl-1024-v1-0": "/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
    }

    DEFAULT_MODEL = "stable-image-core"

    BASE_URL = "https://api.stability.ai"

    # Style presets
    STYLE_PRESETS = [
        "3d-model", "analog-film", "anime", "cinematic", "comic-book",
        "digital-art", "enhance", "fantasy-art", "isometric", "line-art",
        "low-poly", "modeling-compound", "neon-punk", "origami",
        "photographic", "pixel-art", "tile-texture",
    ]

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize Stability AI provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API key not configured.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        self.api_key = self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "Stability AI API key not configured. "
                "Set STABILITY_API_KEY environment variable."
            )

        self._client = None
        self.logger.debug("Stability AI provider initialized")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os

        # Try config first
        try:
            key = self.config_manager.get_config("STABILITY_API_KEY")
            if key:
                return key
        except Exception:
            pass

        return os.environ.get("STABILITY_API_KEY")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                }
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "stability_ai"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Stability AI API.

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
                timing_ms=0,
                error=f"No endpoint configured for model: {model}",
            )

        # Route to appropriate handler
        if model.startswith("stable-image"):
            return await self._generate_stable_image(request, model, endpoint)
        elif model.startswith("sd3"):
            return await self._generate_sd3(request, model, endpoint)
        else:
            return await self._generate_sdxl(request, model, endpoint)

    async def _generate_stable_image(
        self, request: ImageGenerationRequest, model: str, endpoint: str
    ) -> ImageGenerationResult:
        """Generate with Stable Image Ultra/Core."""
        import aiohttp

        client = await self._get_client()
        url = f"{self.BASE_URL}{endpoint}"

        # Build multipart form data
        form = aiohttp.FormData()
        form.add_field("prompt", request.prompt)

        # Optional parameters from metadata
        negative_prompt = request.metadata.get("negative_prompt") if request.metadata else None
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)

        if request.aspect_ratio:
            form.add_field("aspect_ratio", request.aspect_ratio)
        elif request.size:
            # Convert size to aspect ratio
            ar = self._size_to_aspect_ratio(request.size)
            if ar:
                form.add_field("aspect_ratio", ar)

        if request.seed is not None and request.seed > 0:
            form.add_field("seed", str(request.seed))

        if request.style_preset and request.style_preset in self.STYLE_PRESETS:
            form.add_field("style_preset", request.style_preset)

        # Output format
        output_format = "png"
        if request.output_format:
            output_format = request.output_format.value
        form.add_field("output_format", output_format)

        images: List[GeneratedImage] = []

        # Generate n images (Stability doesn't batch, so we loop)
        for i in range(request.n):
            if request.seed is not None and i > 0:
                # Vary seed for subsequent images
                form._fields = [
                    f for f in form._fields if f[0].get("name") != "seed"
                ]
                form.add_field("seed", str(request.seed + i))

            try:
                async with client.post(url, data=form) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(
                            "Stability API error (%d): %s", resp.status, error_text
                        )
                        continue

                    result = await resp.json()

                    # Response contains base64 image
                    if "image" in result:
                        images.append(
                            GeneratedImage(
                                id=str(uuid.uuid4()),
                                mime=f"image/{output_format}",
                                b64=result["image"],
                                seed_used=result.get("seed"),
                            )
                        )
                    elif "artifacts" in result:
                        for artifact in result["artifacts"]:
                            if artifact.get("finishReason") == "SUCCESS":
                                images.append(
                                    GeneratedImage(
                                        id=str(uuid.uuid4()),
                                        mime=f"image/{output_format}",
                                        b64=artifact.get("base64"),
                                        seed_used=artifact.get("seed"),
                                    )
                                )

            except Exception as exc:
                self.logger.error(
                    "Stability generation failed: %s", exc, exc_info=True
                )

        if not images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error="No images generated",
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,
            cost_estimate=self._estimate_cost(model, len(images)),
        )

    async def _generate_sd3(
        self, request: ImageGenerationRequest, model: str, endpoint: str
    ) -> ImageGenerationResult:
        """Generate with SD 3.5 models."""
        import aiohttp

        client = await self._get_client()
        url = f"{self.BASE_URL}{endpoint}"

        form = aiohttp.FormData()
        form.add_field("prompt", request.prompt)
        form.add_field("model", model)

        # Optional parameters from metadata
        negative_prompt = request.metadata.get("negative_prompt") if request.metadata else None
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)

        if request.aspect_ratio:
            form.add_field("aspect_ratio", request.aspect_ratio)

        if request.seed is not None and request.seed > 0:
            form.add_field("seed", str(request.seed))

        if request.style_preset and request.style_preset in self.STYLE_PRESETS:
            form.add_field("style_preset", request.style_preset)

        guidance_scale = request.metadata.get("guidance_scale") if request.metadata else None
        if guidance_scale is not None:
            form.add_field("cfg_scale", str(guidance_scale))

        output_format = "png"
        if request.output_format:
            output_format = request.output_format.value
        form.add_field("output_format", output_format)

        images: List[GeneratedImage] = []

        for i in range(request.n):
            try:
                async with client.post(url, data=form) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error("SD3 API error (%d): %s", resp.status, error_text)
                        continue

                    result = await resp.json()

                    if "image" in result:
                        images.append(
                            GeneratedImage(
                                id=str(uuid.uuid4()),
                                mime=f"image/{output_format}",
                                b64=result["image"],
                                seed_used=result.get("seed"),
                            )
                        )

            except Exception as exc:
                self.logger.error("SD3 generation failed: %s", exc, exc_info=True)

        if not images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error="No images generated",
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,
            cost_estimate=self._estimate_cost(model, len(images)),
        )

    async def _generate_sdxl(
        self, request: ImageGenerationRequest, model: str, endpoint: str
    ) -> ImageGenerationResult:
        """Generate with SDXL 1.0 (legacy v1 API)."""
        client = await self._get_client()
        url = f"{self.BASE_URL}{endpoint}"

        # Build JSON payload for v1 API
        num_steps = request.metadata.get("num_inference_steps", 30) if request.metadata else 30
        payload: Dict[str, Any] = {
            "text_prompts": [{"text": request.prompt, "weight": 1.0}],
            "samples": min(request.n, 10),
            "steps": num_steps,
        }

        # Handle dimensions
        width, height = 1024, 1024
        if request.size:
            width, height = self._parse_size(request.size)
        payload["width"] = width
        payload["height"] = height

        guidance_scale = request.metadata.get("guidance_scale") if request.metadata else None
        if guidance_scale is not None:
            payload["cfg_scale"] = guidance_scale

        if request.seed is not None and request.seed > 0:
            payload["seed"] = request.seed

        if request.style_preset and request.style_preset in self.STYLE_PRESETS:
            payload["style_preset"] = request.style_preset

        images: List[GeneratedImage] = []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            async with client.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return ImageGenerationResult(
                        success=False,
                        images=[],
                        provider=self.name,
                        model=model,
                        timing_ms=0,
                        error=f"SDXL API error ({resp.status}): {error_text}",
                    )

                result = await resp.json()

                for artifact in result.get("artifacts", []):
                    if artifact.get("finishReason") == "SUCCESS":
                        images.append(
                            GeneratedImage(
                                id=str(uuid.uuid4()),
                                mime="image/png",
                                b64=artifact.get("base64"),
                                seed_used=artifact.get("seed"),
                            )
                        )

        except Exception as exc:
            self.logger.error("SDXL generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error=str(exc),
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=0,
            cost_estimate=self._estimate_cost(model, len(images)),
        )

    def _size_to_aspect_ratio(self, size: str) -> Optional[str]:
        """Convert size string to aspect ratio."""
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)

            # Find closest standard ratio
            ratios = {
                (16, 9): "16:9",
                (9, 16): "9:16",
                (1, 1): "1:1",
                (21, 9): "21:9",
                (9, 21): "9:21",
                (2, 3): "2:3",
                (3, 2): "3:2",
                (4, 5): "4:5",
                (5, 4): "5:4",
            }

            from math import gcd
            g = gcd(width, height)
            simplified = (width // g, height // g)

            # Check for exact match first
            if simplified in ratios:
                return ratios[simplified]

            # Find closest
            aspect = width / height
            closest = min(
                ratios.keys(),
                key=lambda r: abs((r[0] / r[1]) - aspect)
            )
            return ratios[closest]

        except Exception:
            return None

    def _parse_size(self, size: str) -> tuple[int, int]:
        """Parse size string to width, height."""
        try:
            w, h = size.lower().split("x")
            return int(w), int(h)
        except Exception:
            return 1024, 1024

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Estimate generation cost in credits."""
        # Credit costs per generation
        credit_costs = {
            "stable-image-ultra": 8,
            "stable-image-core": 3,
            "sd3.5-large": 6.5,
            "sd3.5-large-turbo": 4,
            "sd3.5-medium": 3.5,
            "sd3.5-flash": 2.5,
            "stable-diffusion-xl-1024-v1-0": 0.9,
        }

        credits_per_image = credit_costs.get(model, 3)

        # Approximate USD (credits are roughly $0.01 each)
        usd_per_credit = 0.01

        return {
            "credits": credits_per_image * n,
            "estimated_usd": credits_per_image * n * usd_per_credit,
            "per_image_credits": credits_per_image,
            "model": model,
        }

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using Stability inpainting."""
        if not request.input_images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="inpaint",
                timing_ms=0,
                error="Input image required for editing.",
            )

        # Use inpaint endpoint
        return await self._inpaint(request)

    async def _inpaint(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Inpaint using Stability edit/inpaint endpoint."""
        import aiohttp

        client = await self._get_client()
        url = f"{self.BASE_URL}/v2beta/stable-image/edit/inpaint"

        form = aiohttp.FormData()
        form.add_field("prompt", request.prompt)

        # Add image
        if request.input_images:
            img = request.input_images[0]
            if isinstance(img, bytes):
                form.add_field(
                    "image",
                    img,
                    filename="image.png",
                    content_type="image/png",
                )
            elif isinstance(img, str):
                # Assume path
                with open(img, "rb") as f:
                    form.add_field(
                        "image",
                        f.read(),
                        filename="image.png",
                        content_type="image/png",
                    )

        # Add mask if provided
        if request.mask_image:
            if isinstance(request.mask_image, bytes):
                form.add_field(
                    "mask",
                    request.mask_image,
                    filename="mask.png",
                    content_type="image/png",
                )
            elif isinstance(request.mask_image, str):
                with open(request.mask_image, "rb") as f:
                    form.add_field(
                        "mask",
                        f.read(),
                        filename="mask.png",
                        content_type="image/png",
                    )

        form.add_field("output_format", "png")

        try:
            async with client.post(url, data=form) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return ImageGenerationResult(
                        success=False,
                        images=[],
                        provider=self.name,
                        model="inpaint",
                        timing_ms=0,
                        error=f"Inpaint API error ({resp.status}): {error_text}",
                    )

                result = await resp.json()

                images = []
                if "image" in result:
                    images.append(
                        GeneratedImage(
                            id=str(uuid.uuid4()),
                            mime="image/png",
                            b64=result["image"],
                        )
                    )

                return ImageGenerationResult(
                    success=True,
                    images=images,
                    provider=self.name,
                    model="inpaint",
                    timing_ms=0,
                    cost_estimate={"credits": 5, "estimated_usd": 0.05},
                )

        except Exception as exc:
            self.logger.error("Inpaint failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="inpaint",
                timing_ms=0,
                error=str(exc),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Register provider factory
async def _create_provider(config_manager: "ConfigManager") -> StabilityAIProvider:
    return StabilityAIProvider(config_manager)


register_provider("stability_ai", _create_provider)
register_provider("stability", _create_provider)  # Alias
