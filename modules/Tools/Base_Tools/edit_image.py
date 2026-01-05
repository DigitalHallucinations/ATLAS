"""Image editing tool using the MediaProviderManager.

This tool provides a unified interface for editing images via AI models.
It supports inpainting (editing with masks), outpainting (extending images),
and generating variations of existing images.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.Providers.Media import MediaProviderManager

logger = setup_logger(__name__)


class ImageEditingTool:
    """Tool for editing images via configured media providers.

    Supports:
    - Inpainting: Edit parts of an image using a mask
    - Outpainting: Extend an image beyond its boundaries
    - Variations: Generate variations of an existing image

    Usage::

        from modules.Providers.Media import get_media_provider_manager

        manager = await get_media_provider_manager(config_manager)
        tool = ImageEditingTool(manager)
        result = await tool.run(
            image_path="/path/to/image.png",
            prompt="Add a rainbow in the sky",
            mask_path="/path/to/mask.png",
            operation="inpaint",
        )
    """

    def __init__(
        self,
        media_provider_manager: "MediaProviderManager",
        budget_limiter: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
    ):
        """Initialize the image editing tool.

        Args:
            media_provider_manager: Media provider manager instance.
            budget_limiter: Optional budget limiter for cost tracking.
            persona_manager: Optional persona manager for capability checking.
        """
        self._manager = media_provider_manager
        self._budget_limiter = budget_limiter
        self._persona_manager = persona_manager
        self.logger = setup_logger(__name__)

    async def run(
        self,
        *,
        image_path: str,
        prompt: str,
        operation: str = "inpaint",
        mask_path: Optional[str] = None,
        model: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        provider: Optional[str] = None,
        strength: Optional[float] = None,
        # Context from tool execution
        conversation_id: Optional[str] = None,
        persona: Optional[str] = None,
        user: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Edit an image based on prompt and optional mask.

        Args:
            image_path: Path to the source image to edit.
            prompt: Text description of the desired edit.
            operation: Type of edit - 'inpaint', 'outpaint', 'variation'.
            mask_path: Path to mask image (white areas are edited).
            model: Optional model identifier.
            n: Number of variations to generate (1-4).
            size: Output dimensions (e.g., '1024x1024').
            provider: Explicitly select a provider.
            strength: How much to change the image (0.0-1.0).
            conversation_id: Current conversation context.
            persona: Active persona name.
            user: Active user identifier.

        Returns:
            Result dict with success status and edited images or error.
        """
        # Validate inputs
        if not image_path or not image_path.strip():
            return {
                "success": False,
                "error": "An image path is required for editing.",
            }

        if not prompt or not prompt.strip():
            return {
                "success": False,
                "error": "A prompt describing the edit is required.",
            }

        operation = (operation or "inpaint").lower().strip()
        valid_operations = {"inpaint", "outpaint", "variation", "img2img"}
        if operation not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation '{operation}'. Must be one of: {', '.join(valid_operations)}",
            }

        # Check persona capability
        estimated_cost = self._estimate_cost(model, n, operation)
        capability_check = self._check_persona_capability(
            persona_name=persona,
            provider=provider,
            cost_estimate=estimated_cost,
        )
        if not capability_check.get("allowed", True):
            return {
                "success": False,
                "error": capability_check.get("reason", "Image editing not allowed."),
            }

        # Import here to avoid circular imports
        from modules.Providers.Media.base import ImageGenerationRequest, OutputFormat

        # Build request
        input_images = [image_path.strip()]
        request = ImageGenerationRequest(
            prompt=prompt.strip(),
            model=model,
            n=min(max(1, n), 4),
            size=size,
            input_images=input_images,
            mask_image=mask_path.strip() if mask_path else None,
            output_format=OutputFormat.FILEPATH,
            metadata={
                "conversation_id": conversation_id,
                "persona": persona,
                "user": user,
                "operation": operation,
                "strength": strength,
                "trace_id": self._generate_trace_id(image_path, prompt),
            },
        )

        # Check budget if limiter is configured
        if self._budget_limiter:
            try:
                allowed = await self._check_budget(estimated_cost)
                if not allowed:
                    return {
                        "success": False,
                        "error": "Image editing would exceed budget limit.",
                    }
            except Exception as exc:
                self.logger.warning("Budget check failed: %s", exc)

        # Perform edit
        try:
            result = await self._manager.edit_image(request, provider_override=provider)
        except Exception as exc:
            self.logger.error("Image editing failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error": str(exc),
            }

        if not result.success:
            return {
                "success": False,
                "error": result.error or "Image editing failed.",
            }

        # Record cost if tracking is enabled
        if self._budget_limiter and result.cost_estimate:
            try:
                await self._record_expense(
                    result.cost_estimate.get("estimated_usd", 0),
                    provider=result.provider,
                    model=result.model,
                    operation=operation,
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
                "operation": operation,
                "timing_ms": result.timing_ms,
                "cost_estimate": result.cost_estimate,
            },
        }

    def _generate_trace_id(self, image_path: str, prompt: str) -> str:
        """Generate a trace ID from the inputs."""
        combined = f"{image_path}:{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _estimate_cost(
        self,
        model: Optional[str],
        n: int,
        operation: str,
    ) -> float:
        """Estimate editing cost for budget checking."""
        # Base costs per edit operation (approximate USD)
        # Editing typically costs more than generation
        base_costs = {
            # OpenAI
            "dall-e-2": 0.025,
            "gpt-image-1": 0.08,
            # Stability AI
            "stable-image-ultra": 0.10,
            "stable-image-core": 0.04,
            "sd3.5-large": 0.08,
        }

        per_image = base_costs.get(model or "gpt-image-1", 0.06)

        # Outpainting costs more (larger canvas)
        if operation == "outpaint":
            per_image *= 1.5

        return per_image * n

    def _check_persona_capability(
        self,
        persona_name: Optional[str],
        provider: Optional[str] = None,
        cost_estimate: float = 0.0,
    ) -> Dict[str, Any]:
        """Check if the persona allows image editing."""
        # Reuse the same logic as image generation
        if self._persona_manager is None or not persona_name:
            return {"allowed": True}

        try:
            get_persona = getattr(self._persona_manager, "get_persona", None)
            if callable(get_persona):
                persona_config = get_persona(persona_name)
            else:
                persona_config = None
        except Exception:
            return {"allowed": True}

        if not persona_config or not isinstance(persona_config, dict):
            return {"allowed": True}

        img_config = persona_config.get("image_generation", {})
        if not isinstance(img_config, dict):
            img_config = {}

        enabled = img_config.get("enabled", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")
        if not enabled:
            return {
                "allowed": False,
                "reason": f"Image editing is not enabled for persona '{persona_name}'.",
            }

        max_cost = img_config.get("max_cost_per_image", 0.10)
        if cost_estimate > max_cost:
            return {
                "allowed": False,
                "reason": f"Estimated cost ${cost_estimate:.2f} exceeds max ${max_cost:.2f} per edit.",
            }

        return {"allowed": True}

    async def _check_budget(self, estimated_cost: float) -> bool:
        """Check if the operation is within budget."""
        if self._budget_limiter is None:
            return True

        if hasattr(self._budget_limiter, "check_allowance"):
            return await self._budget_limiter.check_allowance(
                operation="image_editing",
                estimated_cost=estimated_cost,
            )

        return True

    async def _record_expense(
        self,
        cost: float,
        provider: str,
        model: str,
        operation: str,
    ) -> None:
        """Record expense after successful edit."""
        if self._budget_limiter is None:
            return

        if hasattr(self._budget_limiter, "record_expense"):
            await self._budget_limiter.record_expense(
                operation="image_editing",
                cost=cost,
                metadata={
                    "provider": provider,
                    "model": model,
                    "edit_operation": operation,
                },
            )


# Factory function for tool registration
async def create_tool(
    config_manager: Any,
    budget_limiter: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    enable_cost_tracking: bool = True,
    persona_manager: Optional[Any] = None,
) -> ImageEditingTool:
    """Create an ImageEditingTool instance.

    Args:
        config_manager: ATLAS configuration manager.
        budget_limiter: Optional budget limiter. If not provided and
            ``enable_cost_tracking`` is True, a CostBudgetTracker will
            be created automatically.
        session_id: Session/conversation ID for cost tracking scope.
        enable_cost_tracking: Whether to enable automatic cost tracking.
        persona_manager: Optional persona manager for capability checking.

    Returns:
        Configured ImageEditingTool instance.
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

    return ImageEditingTool(
        manager,
        budget_limiter=budget_limiter,
        persona_manager=persona_manager,
    )


__all__ = ["ImageEditingTool", "create_tool"]
