"""Image generation tool using the MediaProviderManager.

This tool provides a unified interface for generating images via AI models.
It integrates with the ATLAS tool execution framework and supports multiple
image generation providers.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger
from modules.Tools.Base_Tools.prompt_compiler import get_compiler, PromptCompiler

if TYPE_CHECKING:
    from modules.Providers.Media import MediaProviderManager

logger = setup_logger(__name__)


class ImageGenerationTool:
    """Tool for generating images via configured media providers.

    Usage::

        from modules.Providers.Media import get_media_provider_manager

        manager = await get_media_provider_manager(config_manager)
        tool = ImageGenerationTool(manager)
        result = await tool.run(prompt="A sunset over mountains")
    """

    def __init__(
        self,
        media_provider_manager: "MediaProviderManager",
        budget_limiter: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
        prompt_compiler: Optional[PromptCompiler] = None,
    ):
        """Initialize the image generation tool.

        Args:
            media_provider_manager: Media provider manager instance.
            budget_limiter: Optional budget limiter for cost tracking.
            persona_manager: Optional persona manager for capability checking.
            prompt_compiler: Optional prompt compiler for optimization.
        """
        self._manager = media_provider_manager
        self._budget_limiter = budget_limiter
        self._persona_manager = persona_manager
        self._prompt_compiler = prompt_compiler or get_compiler()
        self.logger = setup_logger(__name__)

    def _check_persona_image_capability(
        self,
        persona_name: Optional[str],
        provider: Optional[str] = None,
        n: int = 1,
        cost_estimate: float = 0.0,
    ) -> Dict[str, Any]:
        """Check if the persona allows image generation.

        Args:
            persona_name: Name of the active persona.
            provider: Requested provider name.
            n: Number of images requested.
            cost_estimate: Estimated cost per image.

        Returns:
            Dict with 'allowed' bool and optional 'reason' string.
        """
        # No persona manager means no restrictions
        if self._persona_manager is None or not persona_name:
            return {"allowed": True}

        # Get persona config
        try:
            get_persona = getattr(self._persona_manager, "get_persona", None)
            if callable(get_persona):
                persona_config = get_persona(persona_name)
            else:
                persona_config = None
        except Exception:
            # Fail open if we can't get persona config
            return {"allowed": True}

        if not persona_config or not isinstance(persona_config, dict):
            return {"allowed": True}

        # Check image_generation settings
        img_config = persona_config.get("image_generation", {})
        if not isinstance(img_config, dict):
            img_config = {}

        # Check if enabled (default is False for safety)
        enabled = img_config.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")
        if not enabled:
            return {
                "allowed": False,
                "reason": f"Image generation is not enabled for persona '{persona_name}'.",
            }

        # Check allowed providers
        allowed_providers = img_config.get("allowed_providers", [])
        if allowed_providers and provider:
            provider_lower = provider.lower()
            if provider_lower not in [p.lower() for p in allowed_providers]:
                return {
                    "allowed": False,
                    "reason": f"Provider '{provider}' is not allowed for persona '{persona_name}'.",
                }

        # Check max images per request
        max_images = img_config.get("max_images_per_request", 4)
        if n > max_images:
            return {
                "allowed": False,
                "reason": f"Requested {n} images exceeds max {max_images} for persona '{persona_name}'.",
            }

        # Check max cost per image
        max_cost = img_config.get("max_cost_per_image", 0.10)
        if cost_estimate > max_cost:
            return {
                "allowed": False,
                "reason": f"Estimated cost ${cost_estimate:.2f} exceeds max ${max_cost:.2f} per image for persona '{persona_name}'.",
            }

        return {"allowed": True}

    async def run(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        style: Optional[str] = None,
        quality: Optional[str] = None,
        provider: Optional[str] = None,
        seed: Optional[int] = None,
        reference_images: Optional[List[str]] = None,
        negative_prompt: Optional[str] = None,
        compile_prompt: bool = True,
        enhance_quality: bool = True,
        # Context from tool execution
        conversation_id: Optional[str] = None,
        persona: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Generate images from text prompt.

        Args:
            prompt: Text description of the image to generate.
            model: Optional model identifier.
            n: Number of images to generate (1-4).
            size: Image dimensions (e.g., '1024x1024').
            aspect_ratio: Aspect ratio (e.g., '16:9').
            style: Style preset.
            quality: Quality level ('draft', 'standard', 'hd').
            provider: Explicitly select a provider.
            seed: Random seed for reproducibility.
            reference_images: Paths/URLs for image-to-image.
            negative_prompt: Things to avoid in the image.
            compile_prompt: Whether to optimize prompt for the model.
            enhance_quality: Whether to add quality enhancers.
            conversation_id: Current conversation context.
            persona: Active persona name.
            user: Active user identifier.

        Returns:
            Result dict with success status and generated images or error.
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            return {
                "success": False,
                "error": "A non-empty prompt is required for image generation.",
            }

        # Check persona image generation capability
        estimated_per_image = self._estimate_cost(model, 1, size, quality)
        capability_check = self._check_persona_image_capability(
            persona_name=persona,
            provider=provider,
            n=n,
            cost_estimate=estimated_per_image,
        )
        if not capability_check.get("allowed", True):
            return {
                "success": False,
                "error": capability_check.get("reason", "Image generation not allowed."),
            }

        # Import here to avoid circular imports
        from modules.Providers.Media.base import ImageGenerationRequest, OutputFormat

        # Compile prompt for target model if requested
        target_model = model or "flux"  # Default model for compilation
        if compile_prompt and self._prompt_compiler:
            compiled_prompt = self._prompt_compiler.compile(
                prompt.strip(),
                target_model,
                enhance_quality=enhance_quality,
            )
            # Generate negative prompt if not provided
            if negative_prompt is None:
                compiled_negative = self._prompt_compiler.compile_negative(target_model)
            else:
                compiled_negative = negative_prompt
        else:
            compiled_prompt = prompt.strip()
            compiled_negative = negative_prompt

        # Build request
        request = ImageGenerationRequest(
            prompt=compiled_prompt,
            negative_prompt=compiled_negative,
            model=model,
            n=min(max(1, n), 4),
            size=size,
            aspect_ratio=aspect_ratio,
            style_preset=style,
            quality=quality,
            seed=seed,
            input_images=reference_images,
            output_format=OutputFormat.FILEPATH,
            metadata={
                "conversation_id": conversation_id,
                "persona": persona,
                "user": user,
                "trace_id": self._generate_trace_id(prompt),
            },
        )

        # Check budget if limiter is configured
        if self._budget_limiter:
            estimated_cost = self._estimate_cost(model, n, size, quality)
            try:
                allowed = await self._check_budget(estimated_cost)
                if not allowed:
                    return {
                        "success": False,
                        "error": "Image generation would exceed budget limit.",
                    }
            except Exception as exc:
                self.logger.warning("Budget check failed: %s", exc)

        # Generate image(s)
        try:
            result = await self._manager.generate_image(
                request, provider_override=provider
            )
        except Exception as exc:
            self.logger.error("Image generation failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
            }

        if not result.success:
            return {
                "success": False,
                "error": result.error or "Image generation failed.",
            }

        # Record cost if tracking is enabled
        if self._budget_limiter and result.cost_estimate:
            try:
                await self._record_expense(
                    result.cost_estimate.get("estimated_usd", 0),
                    provider=result.provider,
                    model=result.model,
                )
            except Exception as exc:
                self.logger.warning("Failed to record expense: %s", exc)

        # Build response
        return {
            "success": True,
            "data": {
                "images": [
                    {
                        "id": img.id,
                        "mime": img.mime,
                        "path": img.path,
                        "url": img.url,
                        "revised_prompt": img.revised_prompt,
                    }
                    for img in result.images
                ],
                "provider": result.provider,
                "model": result.model,
                "timing_ms": result.timing_ms,
                "cost_estimate": result.cost_estimate,
            },
        }

    def _generate_trace_id(self, prompt: str) -> str:
        """Generate a trace ID from the prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _estimate_cost(
        self,
        model: Optional[str],
        n: int,
        size: Optional[str],
        quality: Optional[str],
    ) -> float:
        """Estimate generation cost for budget checking.

        Costs are approximate and based on published pricing as of 2024.
        Actual costs may vary based on provider pricing changes.
        """
        # Base costs per image (approximate USD)
        base_costs = {
            # OpenAI models
            "dall-e-2": 0.02,
            "dall-e-3": 0.04,
            "gpt-image-1": 0.04,
            "gpt-image-1-mini": 0.02,
            # xAI Grok models
            "grok-2-image": 0.07,
            "grok-2-image-1212": 0.07,
            # Black Forest Labs FLUX models (per megapixel pricing)
            "flux-pro-1.1": 0.04,
            "flux-pro-1.1-ultra": 0.06,
            "flux-pro": 0.05,
            "flux-dev": 0.025,
            "flux-schnell": 0.003,
            "flux-pro-1.1-canny": 0.05,
            "flux-pro-1.1-depth": 0.05,
            "flux-pro-1.1-redux": 0.05,
            # Stability AI models (credits * ~$0.01/credit)
            "stable-image-ultra": 0.08,
            "stable-image-core": 0.03,
            "sd3.5-large": 0.065,
            "sd3.5-large-turbo": 0.04,
            "sd3.5-medium": 0.035,
            "sd3-large": 0.065,
            "sd3-large-turbo": 0.04,
            "sd3-medium": 0.035,
            "stable-diffusion-xl-1024-v1-0": 0.002,
            # Google Gemini (placeholder)
            "gemini-2.0-flash-preview-image-generation": 0.02,
            "imagen-3.0-generate-002": 0.04,
        }

        per_image = base_costs.get(model or "gpt-image-1", 0.04)

        # HD quality costs more for OpenAI
        if quality == "hd" and model in ("dall-e-3", "gpt-image-1"):
            per_image *= 2

        # Larger sizes cost more for DALL-E 3 / GPT-Image
        if size in ("1024x1792", "1792x1024") and model in ("dall-e-3", "gpt-image-1"):
            per_image *= 2

        # Ultra aspect ratios for FLUX
        if model == "flux-pro-1.1-ultra" and size:
            # Ultra supports up to 4MP
            try:
                w, h = map(int, size.lower().split("x"))
                megapixels = (w * h) / 1_000_000
                if megapixels > 1:
                    per_image *= megapixels
            except (ValueError, AttributeError):
                pass

        return per_image * n

    async def _check_budget(self, estimated_cost: float) -> bool:
        """Check if the operation is within budget."""
        if self._budget_limiter is None:
            return True

        if hasattr(self._budget_limiter, "check_allowance"):
            return await self._budget_limiter.check_allowance(
                operation="image_generation",
                estimated_cost=estimated_cost,
            )

        return True

    async def _record_expense(
        self,
        cost: float,
        provider: str,
        model: str,
    ) -> None:
        """Record expense after successful generation."""
        if self._budget_limiter is None:
            return

        if hasattr(self._budget_limiter, "record_expense"):
            await self._budget_limiter.record_expense(
                operation="image_generation",
                cost=cost,
                metadata={"provider": provider, "model": model},
            )


# Factory function for tool registration
async def create_tool(
    config_manager: Any,
    budget_limiter: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    enable_cost_tracking: bool = True,
    persona_manager: Optional[Any] = None,
) -> ImageGenerationTool:
    """Create an ImageGenerationTool instance.

    Args:
        config_manager: ATLAS configuration manager.
        budget_limiter: Optional budget limiter. If not provided and
            ``enable_cost_tracking`` is True, a CostBudgetTracker will
            be created automatically.
        session_id: Session/conversation ID for cost tracking scope.
        enable_cost_tracking: Whether to enable automatic cost tracking.
            Defaults to True.
        persona_manager: Optional persona manager for capability checking.

    Returns:
        Configured ImageGenerationTool instance.
    """
    from modules.Providers.Media import get_media_provider_manager

    manager = await get_media_provider_manager(config_manager)

    # Create cost tracker if not provided
    if budget_limiter is None and enable_cost_tracking:
        try:
            from modules.orchestration.cost_budget_tracker import create_cost_tracker
            budget_limiter = create_cost_tracker(
                config_manager,
                session_id=session_id,
            )
        except ImportError:
            logger.warning("Cost budget tracker not available; cost tracking disabled")

    return ImageGenerationTool(
        manager,
        budget_limiter=budget_limiter,
        persona_manager=persona_manager,
    )


__all__ = ["ImageGenerationTool", "create_tool"]
