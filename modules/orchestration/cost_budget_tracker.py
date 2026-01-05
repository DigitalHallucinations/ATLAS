"""Cost budget tracking utilities for paid API operations.

This module provides cost-based budget tracking for operations like image
generation that incur per-call costs in USD. It complements the runtime-based
tracking in :mod:`~modules.orchestration.budget_tracker`.

The tracker maintains per-session spending limits and integrates with the
``AgentRouter`` session cost tracking when available.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Dict, Optional

from modules.logging.logger import setup_logger


__all__ = [
    "CostBudgetTracker",
    "DEFAULT_SESSION_COST_LIMIT",
    "get_cost_tracker",
    "create_cost_tracker",
]


logger = setup_logger(__name__)


DEFAULT_SESSION_COST_LIMIT: float = 10.0
"""Default per-session cost limit in USD when not configured."""


_tracker_lock = asyncio.Lock()
_session_costs: Dict[str, float] = {}
"""In-memory cost tracking per session."""


class CostBudgetTracker:
    """Async cost budget tracker for paid API operations.

    Provides the interface expected by :class:`~modules.Tools.Base_Tools.generate_image.ImageGenerationTool`
    for budget checking and expense recording.

    Usage::

        tracker = CostBudgetTracker(config_manager, session_id="conv_123")
        
        # Before operation
        allowed = await tracker.check_allowance(
            operation="image_generation",
            estimated_cost=0.04,
        )
        
        # After operation
        await tracker.record_expense(
            operation="image_generation",
            cost=0.04,
            metadata={"provider": "openai", "model": "dall-e-3"},
        )
    """

    def __init__(
        self,
        config_manager: Optional[Any] = None,
        *,
        session_id: Optional[str] = None,
        agent_router: Optional[Any] = None,
    ) -> None:
        """Initialize the cost budget tracker.

        Args:
            config_manager: ATLAS configuration manager for budget settings.
            session_id: Session/conversation ID for cost tracking scope.
            agent_router: Optional AgentRouter for integrated cost tracking.
        """
        self._config_manager = config_manager
        self._session_id = session_id or ""
        self._agent_router = agent_router

    def _resolve_budget_limit(self) -> Optional[float]:
        """Resolve the cost limit from configuration."""
        if self._config_manager is None:
            return DEFAULT_SESSION_COST_LIMIT

        getter = getattr(self._config_manager, "get_config", None)
        if callable(getter):
            try:
                section = getter("tool_defaults", None)
            except TypeError:
                section = getter("tool_defaults")
            if isinstance(section, Mapping):
                candidate = section.get("max_cost_per_session")
                if isinstance(candidate, (int, float)) and candidate >= 0:
                    return float(candidate) if candidate > 0 else None
                if candidate is None:
                    return None
        
        return DEFAULT_SESSION_COST_LIMIT

    async def get_current_spend(self) -> float:
        """Get the current session spending total."""
        # Try AgentRouter first if available
        if self._agent_router is not None:
            router_costs = getattr(self._agent_router, "_session_costs", None)
            if isinstance(router_costs, dict):
                return router_costs.get(self._session_id, 0.0)

        # Fall back to local tracking
        async with _tracker_lock:
            return _session_costs.get(self._session_id, 0.0)

    async def check_allowance(
        self,
        *,
        operation: str,
        estimated_cost: float,
    ) -> bool:
        """Check if an operation is within budget.

        Args:
            operation: Type of operation (e.g., 'image_generation').
            estimated_cost: Estimated cost in USD.

        Returns:
            True if the operation is allowed within budget, False otherwise.
        """
        budget = self._resolve_budget_limit()
        
        # No budget limit configured - always allow
        if budget is None:
            logger.debug(
                "No cost budget configured; allowing %s (est. $%.4f)",
                operation,
                estimated_cost,
            )
            return True

        current_spend = await self.get_current_spend()
        projected_total = current_spend + estimated_cost

        if projected_total > budget + 1e-9:  # Small epsilon for float comparison
            logger.warning(
                "Operation %s would exceed budget: current=$%.4f + estimated=$%.4f = $%.4f > limit=$%.4f",
                operation,
                current_spend,
                estimated_cost,
                projected_total,
                budget,
            )
            return False

        logger.debug(
            "Budget check passed for %s: $%.4f + $%.4f = $%.4f <= $%.4f",
            operation,
            current_spend,
            estimated_cost,
            projected_total,
            budget,
        )
        return True

    async def record_expense(
        self,
        *,
        operation: str,
        cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Record an expense after an operation completes.

        Args:
            operation: Type of operation (e.g., 'image_generation').
            cost: Actual cost in USD.
            metadata: Additional context (provider, model, etc.).

        Returns:
            Updated session spending total.
        """
        if cost <= 0:
            return await self.get_current_spend()

        meta_str = ""
        if metadata:
            provider = metadata.get("provider", "unknown")
            model = metadata.get("model", "unknown")
            meta_str = f" ({provider}/{model})"

        # Update AgentRouter if available
        if self._agent_router is not None:
            router_costs = getattr(self._agent_router, "_session_costs", None)
            if isinstance(router_costs, dict):
                current = router_costs.get(self._session_id, 0.0)
                updated = current + cost
                router_costs[self._session_id] = updated
                logger.info(
                    "Recorded %s expense: $%.4f%s (session total: $%.4f)",
                    operation,
                    cost,
                    meta_str,
                    updated,
                )
                return updated

        # Fall back to local tracking
        async with _tracker_lock:
            current = _session_costs.get(self._session_id, 0.0)
            updated = current + cost
            _session_costs[self._session_id] = updated
            logger.info(
                "Recorded %s expense: $%.4f%s (session total: $%.4f)",
                operation,
                cost,
                meta_str,
                updated,
            )
            return updated

    async def reset(self) -> None:
        """Reset the session cost tracking."""
        # Reset in AgentRouter if available
        if self._agent_router is not None:
            router_costs = getattr(self._agent_router, "_session_costs", None)
            if isinstance(router_costs, dict):
                router_costs.pop(self._session_id, None)
                return

        # Fall back to local tracking
        async with _tracker_lock:
            _session_costs.pop(self._session_id, None)

    async def get_budget_status(self) -> Dict[str, Any]:
        """Get the current budget status.

        Returns:
            Dict with current spend, limit, and remaining budget.
        """
        budget = self._resolve_budget_limit()
        current = await self.get_current_spend()
        remaining = None if budget is None else max(0.0, budget - current)

        return {
            "session_id": self._session_id,
            "current_spend_usd": current,
            "budget_limit_usd": budget,
            "remaining_usd": remaining,
            "budget_exceeded": budget is not None and current >= budget,
        }


def create_cost_tracker(
    config_manager: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
    agent_router: Optional[Any] = None,
) -> CostBudgetTracker:
    """Factory function to create a cost budget tracker.

    Args:
        config_manager: ATLAS configuration manager.
        session_id: Session/conversation ID for tracking scope.
        agent_router: Optional AgentRouter for integrated tracking.

    Returns:
        Configured CostBudgetTracker instance.
    """
    return CostBudgetTracker(
        config_manager,
        session_id=session_id,
        agent_router=agent_router,
    )


async def get_cost_tracker(
    config_manager: Optional[Any] = None,
    *,
    session_id: Optional[str] = None,
) -> CostBudgetTracker:
    """Async factory function to create a cost budget tracker.

    This is the preferred way to create trackers in async contexts.

    Args:
        config_manager: ATLAS configuration manager.
        session_id: Session/conversation ID for tracking scope.

    Returns:
        Configured CostBudgetTracker instance.
    """
    return create_cost_tracker(config_manager, session_id=session_id)
