"""Runtime budget tracking utilities shared across the orchestration layer.

This module centralises the accounting previously embedded in
``ATLAS.ToolManager`` so that other components (for example runtime tools)
can introspect or manipulate the tracked conversation runtime budget using a
consistent API.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Dict, Optional

__all__ = [
    "DEFAULT_CONVERSATION_TOOL_BUDGET_MS",
    "resolve_conversation_budget_ms",
    "get_consumed_runtime_ms",
    "reserve_runtime_ms",
    "release_runtime_ms",
    "reset_runtime",
    "get_runtime_snapshot",
]


DEFAULT_CONVERSATION_TOOL_BUDGET_MS = 120000.0
"""Default per-conversation runtime budget in milliseconds."""

_CONFIG_SECTION = "conversation"
_CONFIG_KEY = "max_tool_duration_ms"

_runtime_lock = asyncio.Lock()
_conversation_runtime_ms: Dict[str, float] = {}


def resolve_conversation_budget_ms(config_manager) -> Optional[float]:
    """Return the configured per-conversation tool runtime budget."""

    if config_manager is not None:
        getter = getattr(config_manager, "get_config", None)
        if callable(getter):
            section = getter(_CONFIG_SECTION)
            if isinstance(section, Mapping):
                candidate = section.get(_CONFIG_KEY)
                if isinstance(candidate, (int, float)):
                    if candidate <= 0:
                        return None
                    return float(candidate)

    return DEFAULT_CONVERSATION_TOOL_BUDGET_MS


async def get_consumed_runtime_ms(conversation_id: Optional[str]) -> float:
    """Return the accumulated runtime for ``conversation_id``."""

    if not conversation_id:
        return 0.0

    async with _runtime_lock:
        return _conversation_runtime_ms.get(conversation_id, 0.0)


async def reserve_runtime_ms(conversation_id: Optional[str], duration_ms: float) -> float:
    """Add ``duration_ms`` to the tracked runtime for ``conversation_id``."""

    if not conversation_id:
        return 0.0

    try:
        increment = float(duration_ms)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return await get_consumed_runtime_ms(conversation_id)

    if increment <= 0:
        return await get_consumed_runtime_ms(conversation_id)

    async with _runtime_lock:
        previous = _conversation_runtime_ms.get(conversation_id, 0.0)
        updated = previous + increment
        _conversation_runtime_ms[conversation_id] = updated
        return updated


async def release_runtime_ms(conversation_id: Optional[str], duration_ms: float) -> float:
    """Subtract ``duration_ms`` from the tracked runtime for ``conversation_id``."""

    if not conversation_id:
        return 0.0

    try:
        decrement = float(duration_ms)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return await get_consumed_runtime_ms(conversation_id)

    if decrement <= 0:
        return await get_consumed_runtime_ms(conversation_id)

    async with _runtime_lock:
        previous = _conversation_runtime_ms.get(conversation_id, 0.0)
        updated = max(0.0, previous - decrement)
        if updated:
            _conversation_runtime_ms[conversation_id] = updated
        else:
            _conversation_runtime_ms.pop(conversation_id, None)
        return updated


async def reset_runtime(conversation_id: Optional[str] = None) -> None:
    """Clear tracked runtime for ``conversation_id`` or all conversations."""

    async with _runtime_lock:
        if conversation_id:
            _conversation_runtime_ms.pop(conversation_id, None)
        else:
            _conversation_runtime_ms.clear()


async def get_runtime_snapshot(conversation_id: Optional[str] = None) -> Dict[str, float]:
    """Return a snapshot of tracked runtimes."""

    async with _runtime_lock:
        if conversation_id:
            value = _conversation_runtime_ms.get(conversation_id, 0.0)
            return {conversation_id: value}
        return dict(_conversation_runtime_ms)
