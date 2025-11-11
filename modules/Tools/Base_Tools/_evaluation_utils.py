"""Shared helpers for evaluation-oriented tools."""

from __future__ import annotations

from typing import Optional

__all__ = ["_normalize_event_name"]


def _normalize_event_name(event_name: Optional[str], *, default: str) -> str:
    """Return a normalized analytics event name.

    Args:
        event_name: Optional override provided by the caller.
        default: Fallback event name to use when *event_name* is not provided or
            empty.

    Returns:
        A stripped event topic string suitable for publishing analytics events.
    """

    if not isinstance(default, str) or not default.strip():
        raise ValueError("default event name must be a non-empty string")

    if isinstance(event_name, str) and event_name.strip():
        return event_name.strip()
    return default.strip()
