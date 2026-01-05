"""Google Gemini and Imagen image generation provider.

Supports:
- Gemini 2.0 Flash with native image generation
- Imagen 3 via Vertex AI
- Imagen 4 via Vertex AI

Requires:
- google-generativeai package for Gemini API access
- google-cloud-aiplatform for Vertex AI/Imagen access

Authentication:
- GOOGLE_API_KEY for Gemini API
- GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS for Vertex AI
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
    # Gemini 2.0 Flash - native image generation
    "gemini-2.0-flash-preview-image-generation": 0.02,
    "gemini-2.0-flash": 0.02,
    # Imagen 3
    "imagen-3.0-generate-002": 0.04,
    "imagen-3.0-generate-001": 0.04,
    "imagen-3.0-fast-generate-001": 0.02,
    # Imagen 4
    "imagen-4.0-generate-001": 0.05,
    "imagen-4.0-ultra-generate-001": 0.08,
    "imagen-4.0-fast-generate-001": 0.03,
}


class GoogleImagesProvider(MediaProvider):
    """Image generation provider using Google Gemini and Imagen.

    Supports both:
    1. Gemini 2.0 Flash with native image generation (via google-generativeai)
    2. Imagen models via Vertex AI (via google-cloud-aiplatform)
    """

    GEMINI_MODELS = [
        "gemini-2.0-flash-preview-image-generation",
        "gemini-2.0-flash",
    ]

    IMAGEN_MODELS = [
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-4.0-fast-generate-001",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        output_dir: Optional[str] = None,
    ):
        """Initialize Google images provider.

        Args:
            api_key: Google API key for Gemini API access.
            project_id: Google Cloud project ID for Vertex AI.
            location: Google Cloud region for Vertex AI.
            output_dir: Directory to save generated images.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location
        self._output_dir = output_dir or os.path.join(
            os.path.expanduser("~"), ".atlas", "media", "generated", "google"
        )
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        self._gemini_client: Optional[Any] = None
        self._vertex_client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "google_images"

    @property
    def supported_models(self) -> List[str]:
        return self.GEMINI_MODELS + self.IMAGEN_MODELS

    async def _ensure_gemini_client(self) -> Any:
        """Lazy initialize Gemini client."""
        if self._gemini_client is not None:
            return self._gemini_client

        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._gemini_client = genai
            return self._gemini_client
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: "
                "pip install google-generativeai"
            )

    async def _ensure_vertex_client(self) -> Any:
        """Lazy initialize Vertex AI client."""
        if self._vertex_client is not None:
            return self._vertex_client

        try:
            from google.cloud import aiplatform

            aiplatform.init(project=self._project_id, location=self._location)
            self._vertex_client = aiplatform
            return self._vertex_client
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform package required for Imagen. Install with: "
                "pip install google-cloud-aiplatform"
            )

    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model."""
        return model in self.GEMINI_MODELS or model.startswith("gemini-")

    def _is_imagen_model(self, model: str) -> bool:
        """Check if model is an Imagen model."""
        return model in self.IMAGEN_MODELS or model.startswith("imagen-")

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Gemini or Imagen.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images.
        """
        model = request.model or "gemini-2.0-flash-preview-image-generation"
        start_time = time.monotonic()

        try:
            if self._is_gemini_model(model):
                result = await self._generate_with_gemini(request, model)
            elif self._is_imagen_model(model):
                result = await self._generate_with_imagen(request, model)
            else:
                # Default to Gemini for unknown models
                result = await self._generate_with_gemini(request, model)

            timing_ms = int((time.monotonic() - start_time) * 1000)

            return ImageGenerationResult(
                success=True,
                images=result["images"],
                provider=self.name,
                model=model,
                timing_ms=timing_ms,
                cost_estimate=self._estimate_cost(model, len(result["images"])),
            )

        except Exception as exc:
            logger.error("Google image generation failed: %s", exc, exc_info=True)
            timing_ms = int((time.monotonic() - start_time) * 1000)
            return ImageGenerationResult(
                success=False,
                images=[],
                provider=self.name,
                model=model,
                timing_ms=timing_ms,
                error=str(exc),
            )

    async def _generate_with_gemini(
        self, request: ImageGenerationRequest, model: str
    ) -> Dict[str, Any]:
        """Generate images using Gemini 2.0 Flash native image generation."""
        genai = await self._ensure_gemini_client()

        # Get the generative model
        gemini_model = genai.GenerativeModel(model)

        # Build the prompt for image generation
        prompt = request.prompt
        if request.style_preset:
            prompt = f"{prompt}, in {request.style_preset} style"

        # Gemini 2.0 Flash generates images when asked explicitly
        generation_prompt = f"Generate an image: {prompt}"

        # Run generation in thread pool (sync API)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(
                generation_prompt,
                generation_config={
                    "response_modalities": ["image", "text"],
                },
            ),
        )

        images: List[GeneratedImage] = []

        # Extract images from response
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    inline_data = part.inline_data
                    if inline_data.mime_type.startswith("image/"):
                        image_id = str(uuid.uuid4())[:8]
                        ext = inline_data.mime_type.split("/")[-1]
                        filename = f"gemini_{image_id}.{ext}"
                        filepath = os.path.join(self._output_dir, filename)

                        # Save image
                        image_bytes = inline_data.data
                        if isinstance(image_bytes, str):
                            image_bytes = base64.b64decode(image_bytes)

                        with open(filepath, "wb") as f:
                            f.write(image_bytes)

                        images.append(
                            GeneratedImage(
                                id=image_id,
                                mime=inline_data.mime_type,
                                path=filepath,
                                revised_prompt=prompt,
                            )
                        )

                        if len(images) >= request.n:
                            break

        return {"images": images}

    async def _generate_with_imagen(
        self, request: ImageGenerationRequest, model: str
    ) -> Dict[str, Any]:
        """Generate images using Imagen via Vertex AI."""
        aiplatform = await self._ensure_vertex_client()

        from vertexai.preview.vision_models import ImageGenerationModel

        # Load the model
        imagen = ImageGenerationModel.from_pretrained(model)

        # Build parameters
        generate_kwargs: Dict[str, Any] = {
            "prompt": request.prompt,
            "number_of_images": min(request.n, 4),
        }

        # Add aspect ratio if specified
        if request.aspect_ratio:
            generate_kwargs["aspect_ratio"] = request.aspect_ratio

        # Add safety settings
        if request.safety:
            generate_kwargs["safety_filter_level"] = request.safety.get(
                "filter_level", "block_few"
            )

        # Run generation in thread pool (sync API)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: imagen.generate_images(**generate_kwargs)
        )

        images: List[GeneratedImage] = []

        for i, image in enumerate(response.images):
            image_id = str(uuid.uuid4())[:8]
            filename = f"imagen_{image_id}.png"
            filepath = os.path.join(self._output_dir, filename)

            # Save the image
            image.save(filepath)

            # Get enhanced prompt if available
            enhanced_prompt = getattr(image, "prompt", None) or request.prompt

            images.append(
                GeneratedImage(
                    id=image_id,
                    mime="image/png",
                    path=filepath,
                    revised_prompt=enhanced_prompt,
                )
            )

        return {"images": images}

    def _estimate_cost(self, model: str, n: int) -> Dict[str, Any]:
        """Calculate cost estimate for generation."""
        per_image = MODEL_PRICING.get(model, 0.04)
        total = per_image * n

        return {
            "estimated_usd": total,
            "per_image_usd": per_image,
            "model": model,
            "count": n,
        }

    async def health_check(self) -> bool:
        """Check provider availability."""
        if not self._api_key and not self._project_id:
            return False

        try:
            if self._api_key:
                await self._ensure_gemini_client()
            return True
        except Exception as exc:
            logger.warning("Google health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Clean up resources."""
        self._gemini_client = None
        self._vertex_client = None


async def _create_google_provider(config_manager: "ConfigManager") -> GoogleImagesProvider:
    """Factory function for creating Google Images provider.
    
    Args:
        config_manager: ATLAS configuration manager.
        
    Returns:
        Initialized GoogleImagesProvider instance.
    """
    return GoogleImagesProvider(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )


def register_with_registry() -> None:
    """Register Google Images provider with the provider registry."""
    from modules.Providers.Media.registry import register_provider
    
    register_provider("google_images", _create_google_provider)
    register_provider("gemini_images", _create_google_provider)
    register_provider("imagen", _create_google_provider)
    register_provider("vertex_imagen", _create_google_provider)
