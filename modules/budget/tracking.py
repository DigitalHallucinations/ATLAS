"""Usage tracking integration for providers.

Provides event handlers that hook into ProviderManager and MediaProviderManager
to capture usage data and record costs in real-time.
"""

from __future__ import annotations

import asyncio
import functools
import time
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING, TypeVar
from uuid import uuid4

from modules.logging.logger import setup_logger

from .models import BudgetCheckResult, LimitAction, OperationType, UsageRecord

if TYPE_CHECKING:
    from core.config import ConfigManager

logger = setup_logger(__name__)

# Module-level singleton
_usage_tracker_instance: Optional["UsageTracker"] = None
_usage_tracker_lock: Optional[asyncio.Lock] = None

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


async def get_usage_tracker(
    config_manager: Optional["ConfigManager"] = None,
) -> "UsageTracker":
    """Get the global UsageTracker singleton.

    Args:
        config_manager: Configuration manager (required on first call).

    Returns:
        Initialized UsageTracker instance.
    """
    global _usage_tracker_instance, _usage_tracker_lock

    if _usage_tracker_instance is not None:
        return _usage_tracker_instance

    if _usage_tracker_lock is None:
        _usage_tracker_lock = asyncio.Lock()

    async with _usage_tracker_lock:
        if _usage_tracker_instance is None:
            _usage_tracker_instance = UsageTracker(config_manager)
            await _usage_tracker_instance.initialize()
            logger.info("UsageTracker singleton created")

    return _usage_tracker_instance


class UsageTracker:
    """Tracks usage across LLM and media providers.

    Integrates with ProviderManager and MediaProviderManager to
    automatically capture and record usage data.

    Usage::

        tracker = await get_usage_tracker(config_manager)

        # Wrap a provider call to track usage
        @tracker.track_llm_call("OpenAI", "gpt-4o")
        async def generate_response(prompt):
            return await openai_client.chat(prompt)

        # Or manually track
        with tracker.track_request("OpenAI", "gpt-4o") as ctx:
            response = await make_request()
            ctx.set_tokens(input=100, output=50)
    """

    def __init__(self, config_manager: Optional["ConfigManager"] = None):
        """Initialize the usage tracker.

        Args:
            config_manager: Optional configuration manager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)

        # Tracking state
        self._enabled = True
        self._pending_requests: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize the tracker."""
        # Load config
        if self.config_manager:
            try:
                budget_config = self.config_manager.get_config("BUDGET")
                if isinstance(budget_config, dict):
                    self._enabled = budget_config.get("tracking_enabled", True)
            except Exception:
                pass

        self.logger.info("UsageTracker initialized, enabled=%s", self._enabled)

    # =========================================================================
    # LLM Tracking
    # =========================================================================

    def track_llm_call(
        self,
        provider: str,
        model: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Callable:
        """Decorator to track LLM API calls.

        Args:
            provider: Provider name.
            model: Model identifier.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.

        Returns:
            Decorator function.

        Example::

            @tracker.track_llm_call("OpenAI", "gpt-4o")
            async def chat(messages):
                return await client.chat.completions.create(...)
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self._enabled:
                    return await func(*args, **kwargs)

                request_id = str(uuid4())
                start_time = time.monotonic()

                try:
                    result = await func(*args, **kwargs)

                    # Extract token counts from result
                    input_tokens, output_tokens = self._extract_token_counts(result)

                    # Record usage
                    await self._record_llm_usage(
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        persona=persona,
                        conversation_id=conversation_id,
                        success=True,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                    )

                    return result

                except Exception as exc:
                    # Record failed request (no tokens billed but track attempt)
                    await self._record_llm_usage(
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        input_tokens=0,
                        output_tokens=0,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        persona=persona,
                        conversation_id=conversation_id,
                        success=False,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                        error=str(exc),
                    )
                    raise

            return wrapper  # type: ignore[return-value]
        return decorator

    def _extract_token_counts(self, result: Any) -> tuple[int, int]:
        """Extract token counts from API response.

        Handles various response formats from different providers.

        Args:
            result: API response object.

        Returns:
            Tuple of (input_tokens, output_tokens).
        """
        input_tokens = 0
        output_tokens = 0

        # OpenAI-style response
        if hasattr(result, "usage"):
            usage = result.usage
            if hasattr(usage, "prompt_tokens"):
                input_tokens = usage.prompt_tokens or 0
            if hasattr(usage, "completion_tokens"):
                output_tokens = usage.completion_tokens or 0

        # Dict-style response
        elif isinstance(result, dict):
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        return input_tokens, output_tokens

    async def _record_llm_usage(
        self,
        request_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str],
        tenant_id: Optional[str],
        persona: Optional[str],
        conversation_id: Optional[str],
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
    ) -> None:
        """Record LLM usage via budget API."""
        try:
            from . import api
            await api.record_llm_usage(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                user_id=user_id,
                tenant_id=tenant_id,
                persona=persona,
                conversation_id=conversation_id,
                request_id=request_id,
                success=success,
                metadata={
                    "duration_ms": duration_ms,
                    "error": error,
                },
            )
        except Exception as exc:
            self.logger.warning("Failed to record LLM usage: %s", exc)

    # =========================================================================
    # Image Generation Tracking
    # =========================================================================

    def track_image_generation(
        self,
        provider: str,
        model: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Callable:
        """Decorator to track image generation calls.

        Args:
            provider: Provider name.
            model: Model identifier.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.

        Returns:
            Decorator function.
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self._enabled:
                    return await func(*args, **kwargs)

                request_id = str(uuid4())
                start_time = time.monotonic()

                # Extract image generation parameters
                count = kwargs.get("n", 1)
                size = kwargs.get("size", "1024x1024")
                quality = kwargs.get("quality", "standard")

                try:
                    result = await func(*args, **kwargs)

                    # Record usage
                    await self._record_image_usage(
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        count=count,
                        size=size,
                        quality=quality,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        persona=persona,
                        conversation_id=conversation_id,
                        success=True,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                    )

                    return result

                except Exception as exc:
                    await self._record_image_usage(
                        request_id=request_id,
                        provider=provider,
                        model=model,
                        count=0,
                        size=size,
                        quality=quality,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        persona=persona,
                        conversation_id=conversation_id,
                        success=False,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                        error=str(exc),
                    )
                    raise

            return wrapper  # type: ignore[return-value]
        return decorator

    async def _record_image_usage(
        self,
        request_id: str,
        provider: str,
        model: str,
        count: int,
        size: str,
        quality: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
        persona: Optional[str],
        conversation_id: Optional[str],
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
    ) -> None:
        """Record image generation usage via budget API."""
        try:
            from . import api
            await api.record_image_usage(
                provider=provider,
                model=model,
                count=count,
                size=size,
                quality=quality,
                user_id=user_id,
                tenant_id=tenant_id,
                persona=persona,
                conversation_id=conversation_id,
                request_id=request_id,
                success=success,
                metadata={
                    "duration_ms": duration_ms,
                    "error": error,
                },
            )
        except Exception as exc:
            self.logger.warning("Failed to record image usage: %s", exc)

    # =========================================================================
    # Manual Tracking Context
    # =========================================================================

    def track_request(
        self,
        provider: str,
        model: str,
        operation_type: OperationType = OperationType.CHAT_COMPLETION,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> "TrackingContext":
        """Create a tracking context for manual usage recording.

        Args:
            provider: Provider name.
            model: Model identifier.
            operation_type: Type of operation.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.

        Returns:
            TrackingContext for use with 'with' statement.

        Example::

            with tracker.track_request("OpenAI", "gpt-4o") as ctx:
                response = await make_request()
                ctx.set_tokens(input=100, output=50)
        """
        return TrackingContext(
            tracker=self,
            provider=provider,
            model=model,
            operation_type=operation_type,
            user_id=user_id,
            tenant_id=tenant_id,
            persona=persona,
            conversation_id=conversation_id,
        )

    # =========================================================================
    # Direct Recording Methods
    # =========================================================================

    async def record_usage(
        self,
        provider: str,
        model: str,
        operation_type: OperationType,
        cost_usd: Decimal,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        images_generated: Optional[int] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[UsageRecord]:
        """Directly record a usage event.

        Args:
            provider: Provider name.
            model: Model identifier.
            operation_type: Type of operation.
            cost_usd: Cost in USD.
            input_tokens: Input token count.
            output_tokens: Output token count.
            images_generated: Image count.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.
            metadata: Additional metadata.

        Returns:
            The recorded UsageRecord, or None if tracking is disabled.
        """
        if not self._enabled:
            return None

        # Build metadata with tenant_id and persona if provided
        record_metadata = metadata.copy() if metadata else {}
        if tenant_id:
            record_metadata["tenant_id"] = tenant_id
        if persona:
            record_metadata["persona"] = persona

        record = UsageRecord(
            provider=provider,
            model=model,
            operation_type=operation_type,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            images_generated=images_generated,
            user_id=user_id,
            conversation_id=conversation_id,
            metadata=record_metadata,
        )

        try:
            from . import api
            # Use appropriate API based on operation type
            if operation_type == OperationType.CHAT_COMPLETION:
                await api.record_llm_usage(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    persona=persona,
                    conversation_id=conversation_id,
                    metadata=metadata,
                )
            elif operation_type == OperationType.IMAGE_GENERATION:
                await api.record_image_usage(
                    provider=provider,
                    model=model,
                    count=images_generated or 1,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    persona=persona,
                    conversation_id=conversation_id,
                    metadata=metadata,
                )
        except Exception as exc:
            self.logger.warning("Failed to record usage: %s", exc)

        return record

    # =========================================================================
    # Pre-request Budget Checks
    # =========================================================================

    async def check_budget(
        self,
        provider: str,
        model: str,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        image_count: int = 0,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> BudgetCheckResult:
        """Check if budget is available before making a request.

        Args:
            provider: Provider name.
            model: Model identifier.
            estimated_input_tokens: Expected input tokens.
            estimated_output_tokens: Expected output tokens.
            image_count: Number of images.
            user_id: User identifier.
            tenant_id: Tenant identifier.

        Returns:
            BudgetCheckResult indicating if request is allowed.
        """
        try:
            from . import api
            return await api.check_budget(
                scope=None,  # Will use GLOBAL scope
                scope_id=tenant_id,
                user_id=user_id,
            )
        except Exception:
            # Return a permissive result when budget check fails
            return BudgetCheckResult(allowed=True, action=LimitAction.WARN)

    @property
    def enabled(self) -> bool:
        """Whether tracking is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable tracking."""
        self._enabled = value


class TrackingContext:
    """Context manager for manual usage tracking.

    Usage::

        with tracker.track_request("OpenAI", "gpt-4o") as ctx:
            response = await make_request()
            ctx.set_tokens(input=100, output=50)
            # Or set cost directly
            ctx.set_cost(Decimal("0.0025"))
    """

    def __init__(
        self,
        tracker: UsageTracker,
        provider: str,
        model: str,
        operation_type: OperationType,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        persona: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        """Initialize tracking context.

        Args:
            tracker: Parent usage tracker.
            provider: Provider name.
            model: Model identifier.
            operation_type: Type of operation.
            user_id: User identifier.
            tenant_id: Tenant identifier.
            persona: Persona name.
            conversation_id: Conversation identifier.
        """
        self.tracker = tracker
        self.provider = provider
        self.model = model
        self.operation_type = operation_type
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.persona = persona
        self.conversation_id = conversation_id

        self.request_id = str(uuid4())
        self.start_time: Optional[float] = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.images_generated = 0
        self.cost_override: Optional[Decimal] = None
        self.success = True
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def __enter__(self) -> "TrackingContext":
        """Enter the context."""
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and record usage."""
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)

        duration_ms = int((time.monotonic() - (self.start_time or 0)) * 1000)
        self.metadata["duration_ms"] = duration_ms

        # Schedule async recording
        asyncio.create_task(self._record())

    async def __aenter__(self) -> "TrackingContext":
        """Async enter."""
        self.start_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async exit and record usage."""
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)

        duration_ms = int((time.monotonic() - (self.start_time or 0)) * 1000)
        self.metadata["duration_ms"] = duration_ms

        await self._record()

    async def _record(self) -> None:
        """Record the usage."""
        if self.cost_override is not None:
            cost = self.cost_override
        else:
            # Calculate cost from tokens/images
            cost = Decimal("0")

        await self.tracker.record_usage(
            provider=self.provider,
            model=self.model,
            operation_type=self.operation_type,
            cost_usd=cost,
            input_tokens=self.input_tokens if self.input_tokens else None,
            output_tokens=self.output_tokens if self.output_tokens else None,
            images_generated=self.images_generated if self.images_generated else None,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            persona=self.persona,
            conversation_id=self.conversation_id,
            metadata=self.metadata,
        )

    def set_tokens(self, input: int = 0, output: int = 0) -> None:
        """Set token counts.

        Args:
            input: Input token count.
            output: Output token count.
        """
        self.input_tokens = input
        self.output_tokens = output

    def set_images(self, count: int) -> None:
        """Set image count.

        Args:
            count: Number of images generated.
        """
        self.images_generated = count

    def set_cost(self, cost: Decimal) -> None:
        """Override automatic cost calculation.

        Args:
            cost: Cost in USD.
        """
        self.cost_override = cost

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value
