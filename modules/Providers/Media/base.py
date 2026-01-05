"""Base interfaces for media generation providers.

This module defines the core abstractions for image generation:
- OutputFormat: Enum for output format preferences
- ImageGenerationRequest: Input parameters for generation
- GeneratedImage: Single generated image result
- ImageGenerationResult: Complete generation response
- MediaProvider: Abstract base class for all image providers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OutputFormat(Enum):
    """Output format for generated images."""

    URL = "url"
    BASE64 = "b64"
    FILEPATH = "filepath"


@dataclass
class ImageGenerationRequest:
    """Request parameters for image generation.

    Attributes:
        prompt: Text description of the image to generate.
        model: Optional model identifier (provider-specific).
        n: Number of images to generate (1-4).
        size: Image dimensions (e.g., "1024x1024").
        aspect_ratio: Alternative to size (e.g., "16:9").
        style_preset: Style preset name (provider-specific).
        quality: Quality level ("draft", "standard", "hd").
        seed: Random seed for reproducibility.
        input_images: File paths/URLs for img2img operations.
        mask_image: Mask for inpainting operations.
        output_format: Preferred output format.
        safety: Provider-specific safety settings.
        metadata: Trace information (conversation_id, persona, etc.).
    """

    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    style_preset: Optional[str] = None
    quality: Optional[str] = None
    seed: Optional[int] = None
    input_images: Optional[List[str]] = None
    mask_image: Optional[str] = None
    output_format: OutputFormat = OutputFormat.FILEPATH
    safety: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        self.prompt = self.prompt.strip()
        self.n = max(1, min(self.n, 4))
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GeneratedImage:
    """Single generated image result.

    Attributes:
        id: Unique identifier for this image.
        mime: MIME type (e.g., "image/png").
        path: Local filesystem path (if saved).
        url: Remote URL (if available).
        b64: Base64-encoded image data.
        seed_used: Random seed used for generation (if known).
        revised_prompt: Model-revised prompt (if applicable).
    """

    id: str
    mime: str
    path: Optional[str] = None
    url: Optional[str] = None
    b64: Optional[str] = None
    seed_used: Optional[int] = None
    revised_prompt: Optional[str] = None


@dataclass
class ImageGenerationResult:
    """Complete result from an image generation request.

    Attributes:
        success: Whether generation succeeded.
        images: List of generated images.
        provider: Provider name that handled the request.
        model: Model identifier used.
        timing_ms: Generation time in milliseconds.
        cost_estimate: Estimated cost information.
        error: Error message if success is False.
    """

    success: bool
    images: List[GeneratedImage]
    provider: str
    model: str
    timing_ms: int
    cost_estimate: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MediaProvider(ABC):
    """Abstract base class for image generation providers.

    All image generation providers must implement this interface.
    Follows the same patterns as ATLAS LLM providers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai_images', 'huggingface_images')."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of model identifiers this provider supports."""
        pass

    @abstractmethod
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate image(s) from text prompt.

        Args:
            request: Image generation parameters.

        Returns:
            ImageGenerationResult with generated images or error.
        """
        pass

    async def edit_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Edit an existing image based on prompt and mask.

        Default implementation raises NotImplementedError.
        Override in providers that support image editing.

        Args:
            request: Edit request with input_images and optional mask.

        Returns:
            ImageGenerationResult with edited images or error.
        """
        return ImageGenerationResult(
            success=False,
            images=[],
            provider=self.name,
            model=request.model or "unknown",
            timing_ms=0,
            error=f"Image editing not supported by {self.name}",
        )

    async def health_check(self) -> bool:
        """Check if provider is available and responding.

        Returns:
            True if provider is healthy, False otherwise.
        """
        return True

    async def close(self) -> None:
        """Clean up provider resources.

        Override in providers that need cleanup (e.g., client connections).
        """
        pass

    def get_default_model(self) -> str:
        """Return the default model for this provider.

        Returns:
            First model from supported_models list.
        """
        models = self.supported_models
        return models[0] if models else "unknown"

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports a given model.

        Args:
            model: Model identifier to check.

        Returns:
            True if model is in supported_models.
        """
        return model in self.supported_models
