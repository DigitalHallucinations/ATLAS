"""Shared utilities for provider adapters."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import guard for type checkers only
    from ATLAS.provider_manager import ProviderManager

ResultPayload = Dict[str, Any]
ProviderInvoker = Callable[["ProviderManager", Callable[..., Awaitable[Any]], Dict[str, Any]], Awaitable[Any]]


_PROVIDER_INVOKERS: Dict[str, ProviderInvoker] = {}


def register_invoker(name: str, invoker: ProviderInvoker) -> None:
    """Register an invocation strategy for the given provider."""

    if not callable(invoker):  # pragma: no cover - defensive validation
        raise ValueError("invoker must be callable")
    _PROVIDER_INVOKERS[name] = invoker


def get_invoker(name: str) -> Optional[ProviderInvoker]:
    """Return the invocation strategy for ``name`` if one is registered."""

    return _PROVIDER_INVOKERS.get(name)


def build_result(
    success: bool,
    *,
    message: str = "",
    error: str = "",
    data: Any = None,
) -> ResultPayload:
    """Create a structured result payload for provider actions."""

    payload: ResultPayload = {"success": success}
    if success:
        if message:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
    else:
        payload["error"] = error or message or "Unknown error"
    return payload
