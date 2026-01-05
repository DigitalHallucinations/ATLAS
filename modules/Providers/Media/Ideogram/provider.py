"""
Ideogram Image Generation Provider.

Ideogram specializes in generating images with accurate text rendering,
making it ideal for:
- Logos and typography
- Posters and marketing materials
- Signs and banners
- Any image requiring legible text

API Documentation: https://docs.ideogram.ai
"""

from __future__ import annotations

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


class IdeogramProvider(MediaProvider):
    """Ideogram image generation provider.

    Provides access to Ideogram's text-accurate image generation models.
    Excels at rendering text within images accurately.

    Attributes:
        SUPPORTED_MODELS: Available model versions.
        DEFAULT_MODEL: Default model for generation requests.
    """

    SUPPORTED_MODELS = [
        "V_2",           # Ideogram 2.0 - latest, best quality
        "V_2_TURBO",     # Ideogram 2.0 Turbo - faster, slightly lower quality
        "V_1",           # Ideogram 1.0 - original model
        "V_1_TURBO",     # Ideogram 1.0 Turbo
    ]

    DEFAULT_MODEL = "V_2"

    # Style types
    STYLE_TYPES = [
        "AUTO",          # Let model decide
        "GENERAL",       # General purpose
        "REALISTIC",     # Photo-realistic
        "DESIGN",        # Design-focused (logos, graphics)
        "RENDER_3D",     # 3D renders
        "ANIME",         # Anime/manga style
    ]

    # Magic prompt options
    MAGIC_PROMPT_OPTIONS = ["AUTO", "ON", "OFF"]

    # Resolution presets
    RESOLUTION_PRESETS = {
        "RESOLUTION_1024_1024": (1024, 1024),
        "RESOLUTION_1280_720": (1280, 720),
        "RESOLUTION_720_1280": (720, 1280),
        "RESOLUTION_1408_704": (1408, 704),
        "RESOLUTION_704_1408": (704, 1408),
        "RESOLUTION_1152_896": (1152, 896),
        "RESOLUTION_896_1152": (896, 1152),
        "RESOLUTION_1344_768": (1344, 768),
        "RESOLUTION_768_1344": (768, 1344),
        "RESOLUTION_1536_640": (1536, 640),
        "RESOLUTION_640_1536": (640, 1536),
    }

    BASE_URL = "https://api.ideogram.ai"

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize Ideogram provider.

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
                "Ideogram API key not configured. "
                "Set IDEOGRAM_API_KEY environment variable."
            )

        self._client = None
        self.logger.debug("Ideogram provider initialized")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os

        try:
            key = self.config_manager.get_config("IDEOGRAM_API_KEY")
            if key:
                return key
        except Exception:
            pass

        return os.environ.get("IDEOGRAM_API_KEY")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp

            self._client = aiohttp.ClientSession(
                headers={
                    "Api-Key": self.api_key or "",
                    "Content-Type": "application/json",
                }
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ideogram"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Ideogram API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        start_time = time.time()

        model = request.model or self.DEFAULT_MODEL
        if model not in self.SUPPORTED_MODELS:
            # Try to map common names
            model_map = {
                "ideogram-2": "V_2",
                "ideogram-2-turbo": "V_2_TURBO",
                "ideogram-1": "V_1",
                "ideogram-1-turbo": "V_1_TURBO",
                "v2": "V_2",
                "v2-turbo": "V_2_TURBO",
            }
            model = model_map.get(model.lower(), self.DEFAULT_MODEL)

        # Build request payload
        payload = self._build_payload(request, model)

        images: List[GeneratedImage] = []
        error_msg: Optional[str] = None

        try:
            client = await self._get_client()
            url = f"{self.BASE_URL}/generate"

            async with client.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self.logger.error(
                        "Ideogram API error (%d): %s", resp.status, error_text
                    )
                    error_msg = f"API error ({resp.status}): {error_text}"
                else:
                    result = await resp.json()

                    # Process generated images
                    data = result.get("data", [])
                    for item in data:
                        image_url = item.get("url")
                        if image_url:
                            # Download image
                            image_data = await self._download_image(image_url)
                            if image_data:
                                images.append(
                                    GeneratedImage(
                                        id=str(uuid.uuid4()),
                                        mime="image/png",
                                        b64=base64.b64encode(image_data).decode(),
                                        url=image_url,
                                        seed_used=item.get("seed"),
                                    )
                                )

        except Exception as exc:
            self.logger.error("Ideogram generation failed: %s", exc, exc_info=True)
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
        """Build Ideogram API request payload.

        Args:
            request: Image generation request.
            model: Ideogram model version.

        Returns:
            API request payload dict.
        """
        # Build image request
        image_request: Dict[str, Any] = {
            "prompt": request.prompt,
            "model": model,
        }

        # Number of images (1-8)
        image_request["num_images"] = min(max(request.n, 1), 8)

        # Resolution from aspect ratio or size
        resolution = self._get_resolution(request)
        if resolution:
            image_request["resolution"] = resolution

        # Style type
        style = request.metadata.get("style_type") if request.metadata else None
        if style and style.upper() in self.STYLE_TYPES:
            image_request["style_type"] = style.upper()

        # Magic prompt (enhances prompts)
        magic = request.metadata.get("magic_prompt") if request.metadata else None
        if magic and magic.upper() in self.MAGIC_PROMPT_OPTIONS:
            image_request["magic_prompt_option"] = magic.upper()
        else:
            image_request["magic_prompt_option"] = "AUTO"

        # Seed for reproducibility
        if request.seed is not None and request.seed > 0:
            image_request["seed"] = request.seed

        # Negative prompt
        negative = request.metadata.get("negative_prompt") if request.metadata else None
        if negative:
            image_request["negative_prompt"] = negative

        return {"image_request": image_request}

    def _get_resolution(self, request: ImageGenerationRequest) -> Optional[str]:
        """Get resolution preset from request dimensions.

        Args:
            request: Image generation request.

        Returns:
            Resolution preset string or None.
        """
        # Direct resolution from metadata
        if request.metadata and "resolution" in request.metadata:
            res = request.metadata["resolution"].upper()
            if res.startswith("RESOLUTION_"):
                return res
            # Try to match dimensions
            for preset, dims in self.RESOLUTION_PRESETS.items():
                if f"{dims[0]}x{dims[1]}" == res or f"{dims[0]}_{dims[1]}" == res:
                    return preset

        # From aspect ratio
        if request.aspect_ratio:
            aspect_map = {
                "1:1": "RESOLUTION_1024_1024",
                "16:9": "RESOLUTION_1280_720",
                "9:16": "RESOLUTION_720_1280",
                "4:3": "RESOLUTION_1152_896",
                "3:4": "RESOLUTION_896_1152",
                "3:2": "RESOLUTION_1344_768",
                "2:3": "RESOLUTION_768_1344",
                "21:9": "RESOLUTION_1536_640",
                "9:21": "RESOLUTION_640_1536",
            }
            return aspect_map.get(request.aspect_ratio)

        # From size string
        if request.size:
            try:
                w, h = request.size.lower().split("x")
                width, height = int(w), int(h)
                # Find closest resolution
                closest = min(
                    self.RESOLUTION_PRESETS.items(),
                    key=lambda p: abs(p[1][0] - width) + abs(p[1][1] - height),
                )
                return closest[0]
            except ValueError:
                pass

        return None

    async def _download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL.

        Args:
            url: Image URL.

        Returns:
            Image bytes or None on failure.
        """
        client = await self._get_client()

        try:
            async with client.get(url) as resp:
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

        Ideogram uses credit-based pricing:
        - V_2: ~$0.06-0.08 per image
        - V_2_TURBO: ~$0.03-0.04 per image
        - V_1: ~$0.05 per image
        """
        costs = {
            "V_2": 0.08,
            "V_2_TURBO": 0.04,
            "V_1": 0.05,
            "V_1_TURBO": 0.03,
        }

        per_image = costs.get(model, 0.08)

        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": model,
        }

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit image using Ideogram's edit endpoint.

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
                model="edit",
                timing_ms=0,
                error="Input image required for editing.",
            )

        start_time = time.time()

        # Prepare image for upload
        image_data = self._prepare_image(request.input_images[0])
        if not image_data:
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="edit",
                timing_ms=0,
                error="Failed to prepare input image.",
            )

        # Prepare mask if provided
        mask_data = None
        if request.mask_image:
            mask_data = self._prepare_image(request.mask_image)

        try:
            import aiohttp

            client = await self._get_client()

            # Build multipart form
            form = aiohttp.FormData()
            form.add_field("prompt", request.prompt)
            form.add_field(
                "image_file",
                image_data,
                filename="image.png",
                content_type="image/png",
            )

            if mask_data:
                form.add_field(
                    "mask",
                    mask_data,
                    filename="mask.png",
                    content_type="image/png",
                )

            model = request.model if request.model in self.SUPPORTED_MODELS else "V_2"
            form.add_field("model", model)

            url = f"{self.BASE_URL}/edit"

            # Need to update headers for multipart
            headers = {"Api-Key": self.api_key}

            async with client.post(url, data=form, headers=headers) as resp:
                elapsed_ms = int((time.time() - start_time) * 1000)

                if resp.status != 200:
                    error_text = await resp.text()
                    return ImageGenerationResult(
                        success=False,
                        images=[],
                        provider=self.name,
                        model="edit",
                        timing_ms=elapsed_ms,
                        error=f"Edit API error ({resp.status}): {error_text}",
                    )

                result = await resp.json()
                images = []

                for item in result.get("data", []):
                    image_url = item.get("url")
                    if image_url:
                        downloaded = await self._download_image(image_url)
                        if downloaded:
                            images.append(
                                GeneratedImage(
                                    id=str(uuid.uuid4()),
                                    mime="image/png",
                                    b64=base64.b64encode(downloaded).decode(),
                                    url=image_url,
                                )
                            )

                return ImageGenerationResult(
                    success=bool(images),
                    images=images,
                    provider=self.name,
                    model="edit",
                    timing_ms=elapsed_ms,
                    error=None if images else "No images generated",
                )

        except Exception as exc:
            self.logger.error("Ideogram edit failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model="edit",
                timing_ms=int((time.time() - start_time) * 1000),
                error=str(exc),
            )

    def _prepare_image(self, image: Any) -> Optional[bytes]:
        """Prepare image data for upload.

        Args:
            image: Image as bytes, path, or base64 string.

        Returns:
            Image bytes or None.
        """
        if isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            if image.startswith("data:"):
                # Data URI - extract base64
                try:
                    _, data = image.split(",", 1)
                    return base64.b64decode(data)
                except Exception:
                    return None
            else:
                # File path
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
async def _create_ideogram_provider(
    config_manager: "ConfigManager",
) -> IdeogramProvider:
    """Factory function for Ideogram provider."""
    return IdeogramProvider(config_manager)


register_provider("ideogram", _create_ideogram_provider)
