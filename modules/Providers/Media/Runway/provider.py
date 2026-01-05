"""
Runway ML Image Generation Provider.

Runway provides AI-powered creative tools including:
- Gen-3 Alpha: Latest image/video generation model
- Text-to-image generation
- Image-to-image transformation
- Style transfer and more

API Documentation: https://docs.runwayml.com
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
    from core.config import ConfigManager


class RunwayProvider(MediaProvider):
    """Runway ML image generation provider.

    Provides access to Runway's AI image generation capabilities
    including Gen-3 Alpha and other creative models.

    Attributes:
        SUPPORTED_MODELS: Available model identifiers.
        DEFAULT_MODEL: Default model for generation requests.
    """

    SUPPORTED_MODELS = [
        "gen3a_turbo",    # Gen-3 Alpha Turbo - fast image generation
        "gen2",           # Gen-2 model
    ]

    DEFAULT_MODEL = "gen3a_turbo"

    BASE_URL = "https://api.runwayml.com/v1"

    # Supported aspect ratios
    ASPECT_RATIOS = [
        "16:9",
        "9:16",
        "1:1",
        "4:3",
        "3:4",
        "21:9",
        "9:21",
    ]

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize Runway provider.

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
                "Runway API key not configured. "
                "Set RUNWAY_API_KEY environment variable."
            )

        self._client = None
        self.logger.debug("Runway provider initialized")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os

        try:
            key = self.config_manager.get_config("RUNWAY_API_KEY")
            if key:
                return key
        except Exception:
            pass

        return os.environ.get("RUNWAY_API_KEY")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp

            self._client = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Runway-Version": "2024-11-06",
                }
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "runway"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Runway API.

        Runway's image generation uses asynchronous task processing.
        This method creates a task and polls for completion.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        start_time = time.time()

        model = request.model or self.DEFAULT_MODEL
        if model not in self.SUPPORTED_MODELS:
            model = self.DEFAULT_MODEL

        # Build request payload
        payload = self._build_payload(request, model)

        images: List[GeneratedImage] = []
        error_msg: Optional[str] = None

        try:
            # Create generation task
            task_id = await self._create_task(model, payload)

            if not task_id:
                error_msg = "Failed to create generation task"
            else:
                # Poll for completion
                result = await self._poll_task(task_id)

                if result.get("error"):
                    error_msg = result["error"]
                elif result.get("output"):
                    # Process outputs
                    for output_url in result["output"]:
                        image_data = await self._download_image(output_url)
                        if image_data:
                            images.append(
                                GeneratedImage(
                                    id=str(uuid.uuid4()),
                                    mime="image/png",
                                    b64=base64.b64encode(image_data).decode(),
                                    url=output_url,
                                )
                            )

        except Exception as exc:
            self.logger.error("Runway generation failed: %s", exc, exc_info=True)
            error_msg = str(exc)

        elapsed_ms = int((time.time() - start_time) * 1000)

        if not images:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=elapsed_ms,
                error=error_msg or "No images generated",
            )

        return ImageGenerationResult(
            success=True,
            images=images,
            provider=self.name,
            model=model,
            timing_ms=elapsed_ms,
            cost_estimate=self._estimate_cost(model, len(images)),
        )

    def _build_payload(
        self, request: ImageGenerationRequest, model: str
    ) -> Dict[str, Any]:
        """Build Runway API request payload.

        Args:
            request: Image generation request.
            model: Runway model identifier.

        Returns:
            API request payload dict.
        """
        payload: Dict[str, Any] = {
            "promptText": request.prompt,
        }

        # Aspect ratio
        if request.aspect_ratio and request.aspect_ratio in self.ASPECT_RATIOS:
            payload["ratio"] = request.aspect_ratio
        elif request.size:
            # Convert size to aspect ratio
            ratio = self._size_to_ratio(request.size)
            if ratio:
                payload["ratio"] = ratio
        else:
            payload["ratio"] = "16:9"  # Default

        # Seed for reproducibility
        if request.seed is not None and request.seed > 0:
            payload["seed"] = request.seed

        # Duration (for video, but some endpoints support)
        duration = request.metadata.get("duration") if request.metadata else None
        if duration:
            payload["duration"] = duration

        # Watermark preference
        if request.metadata and "watermark" in request.metadata:
            payload["watermark"] = request.metadata["watermark"]

        return payload

    def _size_to_ratio(self, size: str) -> Optional[str]:
        """Convert size string to aspect ratio."""
        try:
            w, h = size.lower().split("x")
            width, height = int(w), int(h)

            # Find closest standard ratio
            ratios = {
                (16, 9): "16:9",
                (9, 16): "9:16",
                (1, 1): "1:1",
                (4, 3): "4:3",
                (3, 4): "3:4",
                (21, 9): "21:9",
                (9, 21): "9:21",
            }

            aspect = width / height
            closest = min(
                ratios.keys(),
                key=lambda r: abs((r[0] / r[1]) - aspect)
            )
            return ratios[closest]

        except Exception:
            return None

    async def _create_task(
        self, model: str, payload: Dict[str, Any]
    ) -> Optional[str]:
        """Create a generation task on Runway.

        Args:
            model: Model identifier.
            payload: Request payload.

        Returns:
            Task ID or None on failure.
        """
        client = await self._get_client()

        # Runway uses different endpoints per model
        endpoint = f"{self.BASE_URL}/image_to_video"
        if model == "gen3a_turbo":
            endpoint = f"{self.BASE_URL}/text_to_image"

        try:
            async with client.post(endpoint, json=payload) as resp:
                if resp.status == 200 or resp.status == 201:
                    result = await resp.json()
                    return result.get("id")
                else:
                    error_text = await resp.text()
                    self.logger.error(
                        "Runway task creation failed (%d): %s",
                        resp.status,
                        error_text,
                    )
                    return None

        except Exception as exc:
            self.logger.error("Runway request error: %s", exc)
            return None

    async def _poll_task(
        self, task_id: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """Poll for task completion.

        Args:
            task_id: Runway task ID.
            timeout: Maximum seconds to wait.

        Returns:
            Dict with 'output' list or 'error' message.
        """
        client = await self._get_client()
        url = f"{self.BASE_URL}/tasks/{task_id}"

        poll_interval = 2.0
        waited = 0

        while waited < timeout:
            try:
                async with client.get(url) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(poll_interval)
                        waited += poll_interval
                        continue

                    result = await resp.json()
                    status = result.get("status")

                    if status == "SUCCEEDED":
                        output = result.get("output", [])
                        if isinstance(output, str):
                            output = [output]
                        return {"output": output}

                    elif status in ("FAILED", "CANCELLED"):
                        return {
                            "error": result.get("failure", "Task failed")
                        }

                    # Still processing (PENDING, RUNNING)
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval

                    # Back off polling
                    if waited > 20:
                        poll_interval = min(poll_interval * 1.2, 10.0)

            except Exception as exc:
                self.logger.warning("Poll error: %s", exc)
                await asyncio.sleep(poll_interval)
                waited += poll_interval

        return {"error": "Task timed out"}

    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL.

        Args:
            url: Image URL.

        Returns:
            Image bytes or None on failure.
        """
        client = await self._get_client()

        try:
            # No auth needed for output URLs
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

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Estimate generation cost.

        Runway uses credit-based pricing:
        - Gen-3 Alpha Turbo images: ~$0.05 per image
        - Gen-2: ~$0.03 per image
        """
        costs = {
            "gen3a_turbo": 0.05,
            "gen2": 0.03,
        }

        per_image = costs.get(model, 0.05)

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
            "note": "Actual cost depends on generation parameters",
        }

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using Runway's image-to-image capabilities.

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

        start_time = time.time()

        # Prepare input image
        image_data = self._prepare_image(request.input_images[0])
        if not image_data:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="img2img",
                timing_ms=0,
                error="Failed to prepare input image.",
            )

        # First upload image to get a presigned URL
        # (Runway requires images to be accessible via URL)
        # For now, we'll use base64 encoding if supported

        payload: Dict[str, Any] = {
            "promptText": request.prompt,
            "promptImage": f"data:image/png;base64,{base64.b64encode(image_data).decode()}",
        }

        # Strength parameter controls fidelity to original
        strength = request.metadata.get("strength", 0.5) if request.metadata else 0.5
        payload["strength"] = strength

        try:
            task_id = await self._create_task("gen3a_turbo", payload)

            if not task_id:
                return ImageGenerationResult(
                    success=False,
                    images=[],
                    provider=self.name,
                    model="img2img",
                    timing_ms=int((time.time() - start_time) * 1000),
                    error="Failed to create edit task",
                )

            result = await self._poll_task(task_id)
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
            for output_url in result.get("output", []):
                downloaded = await self._download_image(output_url)
                if downloaded:
                    images.append(
                        GeneratedImage(
                            id=str(uuid.uuid4()),
                            mime="image/png",
                            b64=base64.b64encode(downloaded).decode(),
                            url=output_url,
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

        except Exception as exc:
            self.logger.error("Runway edit failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="img2img",
                timing_ms=int((time.time() - start_time) * 1000),
                error=str(exc),
            )

    def _prepare_image(self, image: Any) -> Optional[bytes]:
        """Prepare image data.

        Args:
            image: Image as bytes, path, or base64 string.

        Returns:
            Image bytes or None.
        """
        if isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            if image.startswith("data:"):
                try:
                    _, data = image.split(",", 1)
                    return base64.b64decode(data)
                except Exception:
                    return None
            else:
                try:
                    with open(image, "rb") as f:
                        return f.read()
                except Exception:
                    return None
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Register provider factory
async def _create_runway_provider(
    config_manager: "ConfigManager",
) -> RunwayProvider:
    """Factory function for Runway provider."""
    return RunwayProvider(config_manager)


register_provider("runway", _create_runway_provider)
