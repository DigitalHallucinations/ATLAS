"""Tool for inspecting and manipulating conversation runtime budgets."""

from __future__ import annotations

from typing import Mapping, Optional

from modules.logging.logger import setup_logger
from modules.orchestration import budget_tracker

__all__ = ["BudgetLimiterError", "BudgetLimiterTool", "budget_limiter"]


logger = setup_logger(__name__)


class BudgetLimiterError(RuntimeError):
    """Raised when a budget limiter request cannot be fulfilled."""


_VALID_OPERATIONS = {"inspect", "reserve", "release", "reset"}


def _normalize_operation(operation: Optional[str]) -> str:
    if not isinstance(operation, str):
        raise BudgetLimiterError("Operation must be a string")
    op = operation.strip().lower()
    if op not in _VALID_OPERATIONS:
        raise BudgetLimiterError(f"Unsupported budget limiter operation '{operation}'")
    return op


def _ensure_conversation_id(conversation_id: Optional[str]) -> str:
    if not conversation_id or not isinstance(conversation_id, str):
        raise BudgetLimiterError("conversation_id is required for this operation")
    return conversation_id


def _ensure_duration(duration_ms: Optional[float]) -> float:
    if duration_ms is None:
        raise BudgetLimiterError("duration_ms must be provided for this operation")
    try:
        duration = float(duration_ms)
    except (TypeError, ValueError):
        raise BudgetLimiterError("duration_ms must be a number") from None
    if duration < 0:
        raise BudgetLimiterError("duration_ms cannot be negative")
    return duration


def _build_budget_payload(conversation_id: Optional[str], consumed: float, budget: Optional[float]):
    remaining = None if budget is None else max(0.0, budget - consumed)
    payload = {
        "operation": None,
        "conversation_id": conversation_id,
        "consumed_ms": consumed,
        "budget_ms": budget,
        "remaining_ms": remaining,
        "budget_exceeded": budget is not None and consumed >= budget,
    }
    return payload


class BudgetLimiterTool:
    """Async tool exposing the shared runtime budget tracker."""

    def __init__(self, *, config_manager=None):
        self._config_manager = config_manager

    def _resolve_budget(self) -> Optional[float]:
        return budget_tracker.resolve_conversation_budget_ms(self._config_manager)

    async def run(
        self,
        *,
        operation: str,
        conversation_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        **_: Mapping[str, object],
    ):
        op = _normalize_operation(operation)
        budget_ms = self._resolve_budget()

        if op == "inspect":
            if conversation_id:
                consumed = await budget_tracker.get_consumed_runtime_ms(conversation_id)
                payload = _build_budget_payload(conversation_id, consumed, budget_ms)
            else:
                snapshot = await budget_tracker.get_runtime_snapshot()
                payload = {
                    "operation": op,
                    "snapshot": snapshot,
                    "budget_ms": budget_ms,
                }
                return payload

            payload["operation"] = op
            return payload

        if op == "reset":
            await budget_tracker.reset_runtime(conversation_id)
            if conversation_id:
                consumed = await budget_tracker.get_consumed_runtime_ms(conversation_id)
                payload = _build_budget_payload(conversation_id, consumed, budget_ms)
                payload["operation"] = op
                return payload
            snapshot = await budget_tracker.get_runtime_snapshot()
            return {"operation": op, "snapshot": snapshot, "budget_ms": budget_ms}

        # reserve/release require both conversation_id and duration
        conversation = _ensure_conversation_id(conversation_id)
        amount = _ensure_duration(duration_ms)

        consumed = await budget_tracker.get_consumed_runtime_ms(conversation)
        payload = _build_budget_payload(conversation, consumed, budget_ms)
        payload.update({"operation": op, "accepted": True})

        if budget_ms is not None and consumed >= budget_ms and op == "reserve":
            payload["accepted"] = False
            payload["reason"] = "budget_exhausted"
            return payload

        if op == "reserve":
            updated = await budget_tracker.reserve_runtime_ms(conversation, amount)
            payload.update(
                {
                    "consumed_ms": updated,
                    "remaining_ms": None
                    if budget_ms is None
                    else max(0.0, budget_ms - updated),
                    "budget_exceeded": budget_ms is not None and updated >= budget_ms,
                    "reserved_ms": amount,
                }
            )
            if payload["budget_exceeded"]:
                logger.debug(
                    "Conversation %s has exceeded its budget: %.2f >= %.2f",
                    conversation,
                    updated,
                    budget_ms,
                )
            return payload

        updated = await budget_tracker.release_runtime_ms(conversation, amount)
        payload.update(
            {
                "consumed_ms": updated,
                "remaining_ms": None
                if budget_ms is None
                else max(0.0, budget_ms - updated),
                "budget_exceeded": budget_ms is not None and updated >= budget_ms,
                "released_ms": amount,
            }
        )
        return payload


async def budget_limiter(**kwargs):
    tool = BudgetLimiterTool()
    return await tool.run(**kwargs)
