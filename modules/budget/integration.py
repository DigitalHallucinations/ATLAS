"""Budget integration module for ATLAS.

Provides hooks and wrappers that integrate budget tracking with
ProviderManager and MediaProviderManager.

Usage::

    from modules.budget.integration import (
        wrap_provider_manager,
        wrap_media_provider_manager,
        setup_budget_integration,
    )

    # During startup
    await setup_budget_integration(config_manager)
"""

from __future__ import annotations

import asyncio
import functools
import time
from decimal import Decimal
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ATLAS.provider_manager import ProviderManager
    from modules.Providers.Media.manager import MediaProviderManager
    from modules.Providers.Media.base import ImageGenerationRequest, ImageGenerationResult

logger = setup_logger(__name__)

# Track if integration is already set up
_integration_initialized = False
_original_generate_response: Optional[Callable] = None
_original_generate_image: Optional[Callable] = None


async def setup_budget_integration(config_manager: "ConfigManager") -> bool:
    """Set up budget integration with providers.

    Initializes the budget manager and hooks into provider managers
    to automatically track usage.

    Args:
        config_manager: ATLAS configuration manager.

    Returns:
        True if integration was set up successfully.
    """
    global _integration_initialized

    if _integration_initialized:
        logger.debug("Budget integration already initialized")
        return True

    try:
        # Check if budget is enabled
        budget_config = getattr(config_manager, "budget", None)
        if budget_config and not budget_config.is_enabled:
            logger.info("Budget management is disabled in configuration")
            return False

        # Initialize the budget manager
        from modules.budget import initialize_budget_manager
        budget_manager = await initialize_budget_manager()

        # Set up default policies if configured
        await _setup_default_policies(config_manager, budget_manager)

        _integration_initialized = True
        logger.info("Budget integration initialized successfully")
        return True

    except Exception as exc:
        logger.error("Failed to initialize budget integration: %s", exc, exc_info=True)
        return False


async def _setup_default_policies(
    config_manager: "ConfigManager",
    budget_manager: Any,
) -> None:
    """Set up default budget policies from configuration.

    Args:
        config_manager: Configuration manager.
        budget_manager: Budget manager instance.
    """
    from modules.budget.models import BudgetPolicy, BudgetPeriod, BudgetScope, LimitAction

    budget_config = getattr(config_manager, "budget", None)
    if not budget_config:
        return

    # Check if we already have a global policy
    existing = await budget_manager.get_policies(scope=BudgetScope.GLOBAL)
    if existing:
        logger.debug("Global budget policy already exists, skipping defaults")
        return

    # Create default global monthly policy
    try:
        default_limit = budget_config.default_global_monthly_limit
        soft_limit_pct = budget_config.soft_limit_percent

        policy = BudgetPolicy(
            name="Default Global Monthly Budget",
            scope=BudgetScope.GLOBAL,
            period=BudgetPeriod.MONTHLY,
            limit_amount=default_limit,
            soft_limit_percent=soft_limit_pct / 100.0,  # Convert to 0-1 range
            hard_limit_action=LimitAction.WARN,
            enabled=True,
            metadata={"source": "default_config"},
        )

        await budget_manager.set_budget_policy(policy)
        logger.info("Created default global budget policy: $%.2f/month", default_limit)

    except Exception as exc:
        logger.warning("Failed to create default budget policy: %s", exc)


async def record_llm_usage_from_response(
    provider: str,
    model: str,
    response: Any,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    persona: Optional[str] = None,
    start_time: Optional[float] = None,
) -> None:
    """Extract usage from an LLM response and record it.

    Args:
        provider: Provider name.
        model: Model name.
        response: Response object from the provider.
        user_id: User identifier.
        tenant_id: Tenant identifier.
        conversation_id: Conversation identifier.
        persona: Persona name.
        start_time: Request start time for latency calculation.
    """
    if not _integration_initialized:
        return

    try:
        from modules.budget import record_llm_usage

        # Extract token counts based on response type
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

        # Handle different response formats
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            if hasattr(usage, "completion_tokens"):
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
            if hasattr(usage, "cached_tokens"):
                cached_tokens = getattr(usage, "cached_tokens", 0) or 0
            if hasattr(usage, "reasoning_tokens"):
                reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
            # Handle dict-style usage
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or 0
                cached_tokens = usage.get("cached_tokens", 0) or 0
                reasoning_tokens = usage.get("reasoning_tokens", 0) or 0

        elif isinstance(response, dict):
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or 0
                cached_tokens = usage.get("cached_tokens", 0) or 0
                reasoning_tokens = usage.get("reasoning_tokens", 0) or 0

        # Skip if no tokens recorded
        if input_tokens == 0 and output_tokens == 0:
            return

        # Record the usage
        await record_llm_usage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id=user_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            persona=persona,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        logger.debug(
            "Recorded LLM usage: %s/%s - %d in, %d out",
            provider,
            model,
            input_tokens,
            output_tokens,
        )

    except Exception as exc:
        # Never let budget tracking break the main flow
        logger.warning("Failed to record LLM usage: %s", exc)


async def record_image_usage_from_result(
    provider: str,
    model: str,
    result: "ImageGenerationResult",
    request: "ImageGenerationRequest",
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    persona: Optional[str] = None,
) -> None:
    """Record image generation usage from a result.

    Args:
        provider: Provider name.
        model: Model name.
        result: Image generation result.
        request: Original request.
        user_id: User identifier.
        tenant_id: Tenant identifier.
        conversation_id: Conversation identifier.
        persona: Persona name.
    """
    if not _integration_initialized:
        return

    if not result.success:
        return

    try:
        from modules.budget import record_image_usage

        await record_image_usage(
            provider=provider,
            model=model,
            images_generated=len(result.images) if result.images else 1,
            resolution=request.size,
            quality=request.quality,
            user_id=user_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            persona=persona,
        )

        logger.debug(
            "Recorded image usage: %s/%s - %d images",
            provider,
            model,
            len(result.images) if result.images else 1,
        )

    except Exception as exc:
        logger.warning("Failed to record image usage: %s", exc)


async def check_budget_before_request(
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    estimated_cost: Optional[Decimal] = None,
) -> bool:
    """Check if budget is available before making a request.

    Args:
        user_id: User identifier.
        tenant_id: Tenant identifier.
        provider: Provider name.
        model: Model name.
        estimated_cost: Estimated cost of the operation.

    Returns:
        True if request should proceed, False if blocked by budget.
    """
    if not _integration_initialized:
        return True

    try:
        from modules.budget import check_budget

        result = await check_budget(
            user_id=user_id,
            tenant_id=tenant_id,
            provider=provider,
            model=model,
            estimated_cost=estimated_cost,
        )

        if not result.allowed:
            warning_msg = result.warnings[0] if result.warnings else "Budget limit reached"
            logger.warning(
                "Budget check failed for user=%s: %s (action=%s)",
                user_id,
                warning_msg,
                result.action.value if result.action else "none",
            )

        return result.allowed

    except Exception as exc:
        logger.warning("Budget check failed with error: %s", exc)
        # On error, allow the request to proceed
        return True


def wrap_generate_response(
    original_func: Callable,
    provider_manager: "ProviderManager",
) -> Callable:
    """Wrap generate_response to add budget tracking.

    Args:
        original_func: Original generate_response method.
        provider_manager: ProviderManager instance.

    Returns:
        Wrapped function with budget tracking.
    """
    @functools.wraps(original_func)
    async def wrapped(
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        user: Optional[str] = None,
        conversation_id: Optional[str] = None,
        current_persona: Any = None,
        **kwargs: Any,
    ) -> Union[str, AsyncIterator[str]]:
        start_time = time.time()

        # Resolve provider and model
        resolved_provider = provider or provider_manager.current_llm_provider
        resolved_model = model or provider_manager.current_model

        # Extract persona name if available
        persona_name = None
        if current_persona:
            if hasattr(current_persona, "name"):
                persona_name = current_persona.name
            elif isinstance(current_persona, dict):
                persona_name = current_persona.get("name")

        # Call original function
        response = await original_func(
            messages=messages,
            model=model,
            provider=provider,
            user=user,
            conversation_id=conversation_id,
            current_persona=current_persona,
            **kwargs,
        )

        # Record usage asynchronously (don't block response)
        if resolved_provider:
            asyncio.create_task(
                record_llm_usage_from_response(
                    provider=resolved_provider,
                    model=resolved_model or "unknown",
                    response=response,
                    user_id=user,
                    conversation_id=conversation_id,
                    persona=persona_name,
                    start_time=start_time,
                )
            )

        return response

    return wrapped


def wrap_generate_image(
    original_func: Callable,
    media_manager: "MediaProviderManager",
) -> Callable:
    """Wrap generate_image to add budget tracking.

    Args:
        original_func: Original generate_image method.
        media_manager: MediaProviderManager instance.

    Returns:
        Wrapped function with budget tracking.
    """
    @functools.wraps(original_func)
    async def wrapped(
        request: "ImageGenerationRequest",
        *,
        provider_override: Optional[str] = None,
        **kwargs: Any,
    ) -> "ImageGenerationResult":
        # Call original function
        result = await original_func(
            request,
            provider_override=provider_override,
            **kwargs,
        )

        # Record usage if successful
        if result.success:
            asyncio.create_task(
                record_image_usage_from_result(
                    provider=result.provider,
                    model=result.model,
                    result=result,
                    request=request,
                    user_id=request.metadata.get("user_id") if request.metadata else None,
                    conversation_id=request.metadata.get("conversation_id") if request.metadata else None,
                )
            )

        return result

    return wrapped


async def shutdown_budget_integration() -> None:
    """Shutdown budget integration and cleanup resources."""
    global _integration_initialized

    if not _integration_initialized:
        return

    try:
        from modules.budget import shutdown_budget_manager
        await shutdown_budget_manager()
        _integration_initialized = False
        logger.info("Budget integration shutdown complete")
    except Exception as exc:
        logger.error("Error during budget integration shutdown: %s", exc)
