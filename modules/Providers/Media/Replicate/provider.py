"""
Replicate.com Image Generation Provider.

Replicate provides API access to thousands of open-source models including:
- FLUX by Black Forest Labs
- Stable Diffusion variants
- SDXL and fine-tunes
- Specialized image models

This provider supports running any image generation model on Replicate.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

from ..base import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
    MediaProvider,
)
from ..registry import register_provider

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager


class ReplicateProvider(MediaProvider):
    """Replicate.com image generation provider.

    Provides access to open-source image models via Replicate's API,
    including FLUX, Stable Diffusion, SDXL, and many specialized models.

    Attributes:
        POPULAR_MODELS: Common model identifiers with their Replicate versions.
        DEFAULT_MODEL: Default model for generation requests.
    """

    # Popular models with their Replicate model identifiers
    # Format: "owner/model-name:version" or just "owner/model-name" for latest
    POPULAR_MODELS = {
        # FLUX models (Black Forest Labs)
        "flux-schnell": "black-forest-labs/flux-schnell",
        "flux-dev": "black-forest-labs/flux-dev",
        "flux-pro": "black-forest-labs/flux-pro",
        "flux-1.1-pro": "black-forest-labs/flux-1.1-pro",
        "flux-1.1-pro-ultra": "black-forest-labs/flux-1.1-pro-ultra",
        # Stable Diffusion
        "sdxl": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "stable-diffusion": "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        # Specialized
        "realvisxl": "lucataco/realvisxl-v4.0:e2c3469f74c44c9d2acfa06e83b7c4a2c9cc58f89bb4bc89a6dc94c7de8bff92",
        "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
        "kandinsky-2.2": "ai-forever/kandinsky-2.2:ea1addaab376f4dc227f5368bbd8ac01fcb023d6b1cc6b2ef6ba2d25e2e28a4a",
        "openjourney": "prompthero/openjourney:ad59ca21177f9e217b9075e7300cf6e14f7e5b4505b87b9689dbd866e9768969",
        "anything-v5": "cjwbw/anything-v5:208a264c1a1e65ee4ebd5e3b5c8ba5b0b5c77e26d3e8f6c32d5d3d2a0eb5fbab4",
    }

    # Model aliases for convenience
    MODEL_ALIASES = {
        "flux": "flux-schnell",
        "sdxl-turbo": "sdxl",
        "sd": "stable-diffusion",
        "sd15": "stable-diffusion",
        "realistic": "realvisxl",
        "anime": "anything-v5",
    }

    DEFAULT_MODEL = "flux-schnell"

    # Replicate API base URL
    BASE_URL = "https://api.replicate.com/v1"

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize Replicate provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API token not configured.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        self.api_token = self._get_api_token()
        if not self.api_token:
            raise ValueError(
                "Replicate API token not configured. "
                "Set REPLICATE_API_TOKEN environment variable."
            )

        self._client = None
        self.logger.debug("Replicate provider initialized")

    def _get_api_token(self) -> Optional[str]:
        """Get API token from config or environment."""
        import os

        # Try config first
        try:
            token = self.config_manager.get_config("REPLICATE_API_TOKEN")
            if token:
                return token
        except Exception:
            pass

        return os.environ.get("REPLICATE_API_TOKEN")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp

            self._client = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json",
                }
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "replicate"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return list(self.POPULAR_MODELS.keys()) + list(self.MODEL_ALIASES.keys())

    def _resolve_model(self, model: str) -> str:
        """Resolve model name to Replicate model identifier.

        Args:
            model: Short model name or full Replicate identifier.

        Returns:
            Full Replicate model identifier (owner/model:version).
        """
        # Check aliases first
        if model in self.MODEL_ALIASES:
            model = self.MODEL_ALIASES[model]

        # Check popular models
        if model in self.POPULAR_MODELS:
            return self.POPULAR_MODELS[model]

        # If it looks like a full Replicate identifier, use as-is
        if "/" in model:
            return model

        # Default to flux-schnell
        self.logger.warning("Unknown model '%s', using %s", model, self.DEFAULT_MODEL)
        return self.POPULAR_MODELS[self.DEFAULT_MODEL]

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Replicate API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        start_time = time.time()

        model = request.model or self.DEFAULT_MODEL
        replicate_model = self._resolve_model(model)

        # Build input parameters
        inputs = self._build_inputs(request, replicate_model)

        images: List[GeneratedImage] = []
        error_msg: Optional[str] = None

        try:
            # Generate requested number of images
            for i in range(request.n):
                # Vary seed for subsequent images
                if request.seed is not None and i > 0:
                    inputs["seed"] = request.seed + i

                result = await self._run_prediction(replicate_model, inputs)

                if result.get("error"):
                    error_msg = result["error"]
                    continue

                # Process outputs
                output = result.get("output")
                if output:
                    # Output can be a URL string, list of URLs, or dict
                    urls = self._extract_output_urls(output)
                    for url in urls:
                        image_data = await self._download_image(url)
                        if image_data:
                            # Determine MIME type
                            mime = self._get_mime_from_url(url)
                            images.append(
                                GeneratedImage(
                                    id=str(uuid.uuid4()),
                                    mime=mime,
                                    b64=base64.b64encode(image_data).decode(),
                                    url=url,
                                )
                            )

        except Exception as exc:
            self.logger.error(
                "Replicate generation failed: %s", exc, exc_info=True
            )
            error_msg = str(exc)

        elapsed_ms = int((time.time() - start_time) * 1000)

        if not images and error_msg:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=elapsed_ms,
                error=error_msg,
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=elapsed_ms,
            cost_estimate=self._estimate_cost(replicate_model, len(images)),
        )

    def _build_inputs(
        self, request: ImageGenerationRequest, replicate_model: str
    ) -> Dict[str, Any]:
        """Build Replicate input parameters from request.

        Args:
            request: Image generation request.
            replicate_model: Full Replicate model identifier.

        Returns:
            Dict of input parameters for the model.
        """
        inputs: Dict[str, Any] = {
            "prompt": request.prompt,
        }

        # Negative prompt
        negative = request.metadata.get("negative_prompt") if request.metadata else None
        if negative:
            inputs["negative_prompt"] = negative

        # Dimensions
        if request.size:
            try:
                w, h = request.size.lower().split("x")
                inputs["width"] = int(w)
                inputs["height"] = int(h)
            except ValueError:
                pass
        elif request.aspect_ratio:
            # Map aspect ratio to dimensions
            dims = self._aspect_ratio_to_dims(request.aspect_ratio)
            if dims:
                inputs["width"], inputs["height"] = dims

        # Seed
        if request.seed is not None and request.seed > 0:
            inputs["seed"] = request.seed

        # Guidance scale
        guidance = request.metadata.get("guidance_scale") if request.metadata else None
        if guidance is not None:
            # Different models use different parameter names
            inputs["guidance_scale"] = guidance
            inputs["guidance"] = guidance  # FLUX uses this

        # Steps
        steps = request.metadata.get("num_inference_steps") if request.metadata else None
        if steps is not None:
            inputs["num_inference_steps"] = steps
            inputs["steps"] = steps  # Alternative name

        # Output format
        if request.output_format:
            inputs["output_format"] = request.output_format.value

        # Model-specific parameters from metadata
        if request.metadata:
            # Pass through any extra parameters
            for key in ["scheduler", "lora_scale", "refiner"]:
                if key in request.metadata:
                    inputs[key] = request.metadata[key]

        return inputs

    def _aspect_ratio_to_dims(self, aspect_ratio: str) -> Optional[tuple[int, int]]:
        """Convert aspect ratio to dimensions."""
        ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "21:9": (1536, 640),
            "9:21": (640, 1536),
        }
        return ratios.get(aspect_ratio)

    async def _run_prediction(
        self, model: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a Replicate prediction and wait for completion.

        Args:
            model: Replicate model identifier.
            inputs: Model input parameters.

        Returns:
            Prediction result dict with 'output' or 'error'.
        """
        client = await self._get_client()

        # Create prediction
        create_url = f"{self.BASE_URL}/predictions"
        payload = {
            "version": model.split(":")[-1] if ":" in model else None,
            "input": inputs,
        }

        # If version not specified, use model identifier
        if not payload["version"]:
            payload = {
                "model": model,
                "input": inputs,
            }

        try:
            async with client.post(create_url, json=payload) as resp:
                if resp.status != 201:
                    error_text = await resp.text()
                    self.logger.error(
                        "Replicate create error (%d): %s", resp.status, error_text
                    )
                    return {"error": f"Failed to create prediction: {error_text}"}

                prediction = await resp.json()

        except Exception as exc:
            return {"error": f"Request failed: {exc}"}

        # Poll for completion
        prediction_id = prediction.get("id")
        get_url = f"{self.BASE_URL}/predictions/{prediction_id}"

        max_wait = 300  # 5 minutes
        poll_interval = 1.0
        waited = 0

        while waited < max_wait:
            try:
                async with client.get(get_url) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(poll_interval)
                        waited += poll_interval
                        continue

                    prediction = await resp.json()
                    status = prediction.get("status")

                    if status == "succeeded":
                        return {"output": prediction.get("output")}
                    elif status in ("failed", "canceled"):
                        return {"error": prediction.get("error", "Prediction failed")}

                    # Still processing
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval

                    # Back off polling interval
                    if waited > 10:
                        poll_interval = min(poll_interval * 1.2, 5.0)

            except Exception as exc:
                self.logger.warning("Poll error: %s", exc)
                await asyncio.sleep(poll_interval)
                waited += poll_interval

        return {"error": "Prediction timed out"}

    def _extract_output_urls(self, output: Any) -> List[str]:
        """Extract image URLs from prediction output.

        Args:
            output: Prediction output (string, list, or dict).

        Returns:
            List of image URLs.
        """
        urls = []

        if isinstance(output, str):
            # Single URL
            if output.startswith("http"):
                urls.append(output)
        elif isinstance(output, list):
            # List of URLs or dicts
            for item in output:
                if isinstance(item, str) and item.startswith("http"):
                    urls.append(item)
                elif isinstance(item, dict) and "url" in item:
                    urls.append(item["url"])
        elif isinstance(output, dict):
            # Dict with URL field
            if "url" in output:
                urls.append(output["url"])
            elif "image" in output:
                urls.append(output["image"])

        return urls

    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL.

        Args:
            url: Image URL.

        Returns:
            Image bytes or None on failure.
        """
        client = await self._get_client()

        try:
            # Use separate headers for download (no auth needed)
            async with client.get(url, headers={}) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    self.logger.warning(
                        "Failed to download image (%d): %s", resp.status, url
                    )
        except Exception as exc:
            self.logger.warning("Image download error: %s", exc)

        return None

    def _get_mime_from_url(self, url: str) -> str:
        """Infer MIME type from URL."""
        url_lower = url.lower()
        if ".png" in url_lower:
            return "image/png"
        elif ".webp" in url_lower:
            return "image/webp"
        elif ".jpg" in url_lower or ".jpeg" in url_lower:
            return "image/jpeg"
        return "image/png"  # Default

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Estimate generation cost.

        Replicate charges per second of GPU time. Costs vary by model.
        """
        # Approximate costs per image (varies by model and hardware)
        model_costs = {
            "flux-schnell": 0.003,  # Very fast, cheap
            "flux-dev": 0.025,
            "flux-pro": 0.055,
            "flux-1.1-pro": 0.04,
            "flux-1.1-pro-ultra": 0.06,
            "sdxl": 0.01,
            "stable-diffusion": 0.005,
        }

        # Find cost for this model
        per_image = 0.02  # Default
        for key, cost in model_costs.items():
            if key in model.lower():
                per_image = cost
                break

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "note": "Actual cost depends on GPU time",
        }

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using img2img models.

        Args:
            request: Request with input image and instructions.

        Returns:
            ImageGenerationResult with edited image.
        """
        if not request.input_images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="img2img",
                timing_ms=0,
                error="Input image required for editing.",
            )

        # Use SDXL img2img or similar model
        model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

        inputs: Dict[str, Any] = {
            "prompt": request.prompt,
            "image": self._encode_input_image(request.input_images[0]),
        }

        # Strength controls how much to modify
        strength = request.metadata.get("strength", 0.7) if request.metadata else 0.7
        inputs["prompt_strength"] = strength

        if request.metadata and "negative_prompt" in request.metadata:
            inputs["negative_prompt"] = request.metadata["negative_prompt"]

        start_time = time.time()
        result = await self._run_prediction(model, inputs)
        elapsed_ms = int((time.time() - start_time) * 1000)

        if result.get("error"):
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="img2img",
                timing_ms=elapsed_ms,
                error=result["error"],
            )

        images = []
        urls = self._extract_output_urls(result.get("output"))
        for url in urls:
            image_data = await self._download_image(url)
            if image_data:
                images.append(
                    GeneratedImage(
                        id=str(uuid.uuid4()),
                        mime="image/png",
                        b64=base64.b64encode(image_data).decode(),
                        url=url,
                    )
                )

        return ImageGenerationResult(
            success=bool(images),
            images=images,
            provider=self.name,
            model="img2img",
            timing_ms=elapsed_ms,
            error=None if images else "No images generated",
        )

    def _encode_input_image(self, image: Any) -> str:
        """Encode input image to base64 data URI.

        Args:
            image: Image as bytes, path, or base64 string.

        Returns:
            Data URI string.
        """
        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode()
            return f"data:image/png;base64,{b64}"
        elif isinstance(image, str):
            if image.startswith("data:"):
                return image
            elif image.startswith("http"):
                return image  # Replicate accepts URLs directly
            else:
                # Assume file path
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    return f"data:image/png;base64,{b64}"
        return ""

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Register provider factory
async def _create_replicate_provider(
    config_manager: "ConfigManager",
) -> ReplicateProvider:
    """Factory function for Replicate provider."""
    return ReplicateProvider(config_manager)


register_provider("replicate", _create_replicate_provider)
