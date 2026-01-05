"""xAI Aurora provider for Grok image generation.

xAI provides image generation through the grok-2-image model (Aurora).

API Endpoint: https://api.x.ai/v1/images/generations
Documentation: https://docs.x.ai/docs/guides/image-generations

Features:
- Generate 1-10 images per request
- URL or base64 output
- Automatic prompt revision by chat model
- Compatible with OpenAI SDK
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
    from core.config import ConfigManager

logger = setup_logger(__name__)


class XAIImagesProvider(MediaProvider):
    """xAI Aurora image generation provider (Grok-2-Image).

    Uses the xAI API directly for image generation.
    Compatible with OpenAI SDK patterns.
    Requires XAI_API_KEY or GROK_API_KEY environment variable.
    """

    SUPPORTED_MODELS = [
        "grok-2-image",
        "grok-2-image-1212",
    ]

    DEFAULT_MODEL = "grok-2-image"

    # API endpoints
    BASE_URL = "https://api.x.ai/v1"
    IMAGES_ENDPOINT = "/images/generations"

    def __init__(self, config_manager: "ConfigManager"):
        """Initialize xAI Images provider.

        Args:
            config_manager: ATLAS configuration manager.

        Raises:
            ValueError: If API key not configured.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Get API key - try XAI_API_KEY first, fall back to GROK_API_KEY
        self.api_key = self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "xAI API key not configured. "
                "Set XAI_API_KEY or GROK_API_KEY environment variable."
            )

        self._client = None
        self._xai_sdk_client = None
        self.logger.debug("xAI Images provider initialized")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        import os

        # Try XAI_API_KEY first
        key = os.environ.get("XAI_API_KEY")
        if key:
            return key

        # Fall back to GROK_API_KEY (same key works for both)
        try:
            key = self.config_manager.get_grok_api_key()
            if key:
                return key
        except Exception:
            pass

        return os.environ.get("GROK_API_KEY")

    async def _get_client(self) -> Any:
        """Get or create aiohttp client session."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._client

    def _get_xai_sdk_client(self) -> Any:
        """Get or create xai_sdk Client (if available)."""
        if self._xai_sdk_client is None:
            try:
                from xai_sdk import Client
                self._xai_sdk_client = Client(api_key=self.api_key)
            except ImportError:
                self._xai_sdk_client = None
        return self._xai_sdk_client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "xai_aurora"

    @property
    def supported_models(self) -> List[str]:
        """List of supported model identifiers."""
        return self.SUPPORTED_MODELS

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using xAI Grok-2-Image API.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        model = request.model or self.DEFAULT_MODEL
        if model not in self.SUPPORTED_MODELS:
            self.logger.warning("Unknown model %s, using %s", model, self.DEFAULT_MODEL)
            model = self.DEFAULT_MODEL

        # Try using xai_sdk if available (preferred)
        sdk_client = self._get_xai_sdk_client()
        if sdk_client is not None:
            return await self._generate_with_sdk(sdk_client, request, model)

        # Fall back to direct API calls
        return await self._generate_with_api(request, model)

    async def _generate_with_sdk(
        self, client: Any, request: ImageGenerationRequest, model: str
    ) -> ImageGenerationResult:
        """Generate images using xai_sdk Client."""
        import asyncio

        images: List[GeneratedImage] = []
        revised_prompts: List[str] = []

        try:
            # Run sync SDK call in thread pool
            def _generate():
                if request.n > 1:
                    # Batch generation
                    return client.image.sample_batch(
                        model=model,
                        prompt=request.prompt,
                        n=min(request.n, 10),  # Max 10 per request
                        image_format="url",
                    )
                else:
                    # Single image
                    return client.image.sample(
                        model=model,
                        prompt=request.prompt,
                        image_format="url",
                    )

            response = await asyncio.get_event_loop().run_in_executor(
                None, _generate
            )

            # Handle response
            if request.n > 1:
                # Batch response is iterable
                for img in response:
                    images.append(
                        GeneratedImage(
                            id=str(uuid.uuid4()),
                            mime="image/jpeg",
                            url=getattr(img, "url", None),
                            revised_prompt=getattr(img, "prompt", None),
                        )
                    )
                    if hasattr(img, "prompt"):
                        revised_prompts.append(img.prompt)
            else:
                # Single image response
                images.append(
                    GeneratedImage(
                        id=str(uuid.uuid4()),
                        mime="image/jpeg",
                        url=getattr(response, "url", None),
                        revised_prompt=getattr(response, "prompt", None),
                    )
                )
                if hasattr(response, "prompt"):
                    revised_prompts.append(response.prompt)

            return ImageGenerationResult(
                success=True,
                images=images,
                provider=self.name,
                model=model,
                timing_ms=0,
                cost_estimate=self._estimate_cost(len(images)),
            )

        except Exception as exc:
            self.logger.error("xAI SDK generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error=str(exc),
            )

    async def _generate_with_api(
        self, request: ImageGenerationRequest, model: str
    ) -> ImageGenerationResult:
        """Generate images using direct API calls."""
        client = await self._get_client()
        url = f"{self.BASE_URL}{self.IMAGES_ENDPOINT}"

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "n": min(request.n, 10),  # Max 10 per request
            "response_format": "url",
        }

        images: List[GeneratedImage] = []

        try:
            async with client.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return ImageGenerationResult(
                        success=False,
                        images=[],
                        provider=self.name,
                        model=model,
                        timing_ms=0,
                        error=f"xAI API error ({resp.status}): {error_text}",
                    )

                result = await resp.json()

                # Parse response data
                for item in result.get("data", []):
                    images.append(
                        GeneratedImage(
                            id=str(uuid.uuid4()),
                            mime="image/jpeg",
                            url=item.get("url"),
                            revised_prompt=item.get("revised_prompt"),
                        )
                    )

            return ImageGenerationResult(
                success=True,
                images=images,
                provider=self.name,
                model=model,
                timing_ms=0,
                cost_estimate=self._estimate_cost(len(images)),
            )

        except Exception as exc:
            self.logger.error("xAI API generation failed: %s", exc, exc_info=True)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=0,
                error=str(exc),
            )

    def _estimate_cost(self, n: int) -> Dict[str, Any]:
        """Estimate generation cost."""
        # xAI pricing: $0.07 per image
        per_image = 0.07
        return {
            "estimated_usd": per_image * n,
            "per_image_usd": per_image,
            "model": "grok-2-image",
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Register provider factory
async def _create_provider(config_manager: "ConfigManager") -> XAIImagesProvider:
    return XAIImagesProvider(config_manager)


register_provider("xai_aurora", _create_provider)
