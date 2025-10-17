"""Utilities for routing capability requests to concrete tools."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional

from modules.logging.logger import setup_logger

try:  # ConfigManager is optional in some test contexts
    from ATLAS.config import ConfigManager
except Exception:  # pragma: no cover - fallback for isolated tests
    ConfigManager = None  # type: ignore


logger = setup_logger(__name__)


def _format_cost(value: float) -> str:
    """Return a human readable representation of a numeric cost."""

    formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _normalize_cost(value: Any) -> Optional[float]:
    """Coerce ``value`` to a non-negative float when possible."""

    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    return None


def _normalize_capabilities(metadata: Mapping[str, Any]) -> Iterable[str]:
    """Yield normalized capability tokens from ``metadata``."""

    if not metadata:
        return []

    candidates = metadata.get("capabilities")
    if isinstance(candidates, str):
        candidates = [candidates]
    elif not isinstance(candidates, IterableABC):
        candidates = []

    required_capabilities = metadata.get("required_capabilities")
    if required_capabilities:
        if isinstance(required_capabilities, str):
            candidates = list(candidates) + [required_capabilities]
        elif isinstance(required_capabilities, IterableABC):
            candidates = list(candidates) + list(required_capabilities)
        else:
            candidates = list(candidates) + [required_capabilities]

    normalized = []
    for candidate in candidates:
        if candidate is None:
            continue
        token = str(candidate).strip().lower()
        if token:
            normalized.append(token)
    return normalized


def _normalize_allowlist(allowed_tools: Any) -> Optional[Iterable[str]]:
    """Return a deduplicated iterable of tool names from ``allowed_tools``."""

    if allowed_tools is None:
        return None

    names: List[str] = []

    if isinstance(allowed_tools, str):
        candidate = allowed_tools.strip()
        if candidate:
            names.append(candidate)
    elif isinstance(allowed_tools, IterableABC):
        for item in allowed_tools:
            if isinstance(item, MappingABC):
                raw_name = item.get("name")
                candidate = str(raw_name).strip() if raw_name is not None else ""
            else:
                candidate = str(item).strip() if item is not None else ""

            if candidate and candidate not in names:
                names.append(candidate)
    else:
        candidate = str(allowed_tools).strip()
        if candidate:
            names.append(candidate)

    return names


@dataclass(frozen=True)
class RouterDecision:
    """Represents the outcome of a routing decision."""

    allowed: bool
    tool_name: Optional[str] = None
    reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


class AgentRouter:
    """Route capability requests to the most appropriate tool implementation."""

    def __init__(self, *, config_manager: Optional[Any] = None) -> None:
        self._config_manager = config_manager
        self._session_costs: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_tool(
        self,
        capability: str,
        function_map: Optional[Mapping[str, Any]],
        *,
        session_id: Optional[str] = None,
        persona_context: Optional[Mapping[str, Any]] = None,
        allowed_tools: Optional[Iterable[str]] = None,
    ) -> RouterDecision:
        """Select the cheapest tool that satisfies ``capability``."""

        if not capability:
            return RouterDecision(False, reason="No capability provided.")

        if not isinstance(function_map, MappingABC) or not function_map:
            return RouterDecision(False, reason="No tools are available.")

        requested_capability = str(capability).strip().lower()
        if not requested_capability:
            return RouterDecision(False, reason="No capability provided.")

        persona_name: Optional[str] = None
        allowlist_source: Any = allowed_tools
        if isinstance(persona_context, MappingABC):
            persona_name_candidate = persona_context.get("persona_name")
            if persona_name_candidate is not None:
                persona_name = str(persona_name_candidate)
            if allowlist_source is None and "allowed_tools" in persona_context:
                allowlist_source = persona_context.get("allowed_tools")

        normalized_allowlist = _normalize_allowlist(allowlist_source)
        allowlist_defined = allowlist_source is not None
        allowlist_set = None
        if normalized_allowlist is not None:
            allowlist_set = {name for name in normalized_allowlist}

        candidates = []
        matching_capability = []
        disallowed_candidates = []
        for name, entry in function_map.items():
            metadata = {}
            if isinstance(entry, MappingABC):
                metadata = entry.get("metadata") or {}
                if not isinstance(metadata, MappingABC):
                    metadata = {}
            capabilities = set(_normalize_capabilities(metadata))
            if requested_capability not in capabilities:
                continue
            matching_capability.append((name, metadata))
            if allowlist_set is not None and name not in allowlist_set:
                disallowed_candidates.append((name, metadata))
                continue
            cost = _normalize_cost(metadata.get("cost_per_call"))
            candidates.append((name, metadata, cost))

        if not candidates:
            if allowlist_defined and matching_capability:
                persona_label = f"Persona '{persona_name}'" if persona_name else "The current persona"
                reason = (
                    f"{persona_label} is not permitted to use tools for the "
                    f"'{capability}' capability."
                )
                disallowed_tools = [
                    name for name, _ in (disallowed_candidates or matching_capability)
                ]
                logger.info(
                    "%s (persona=%s capability=%s session=%s disallowed=%s)",
                    reason,
                    persona_name or "<unknown>",
                    requested_capability,
                    session_id or "<none>",
                    disallowed_tools,
                )
                metadata = {
                    "persona_name": persona_name,
                    "disallowed_tools": tuple(
                        name for name, _ in (disallowed_candidates or matching_capability)
                    ),
                }
                return RouterDecision(
                    False,
                    reason=reason,
                    metadata=MappingProxyType(metadata),
                )
            return RouterDecision(
                False,
                reason=f"No registered tool exposes the '{capability}' capability.",
            )

        candidates.sort(
            key=lambda item: (
                0 if item[2] is not None else 1,
                item[2] if item[2] is not None else float("inf"),
                item[0],
            )
        )
        selected_name, selected_metadata, selected_cost = candidates[0]

        budget = self._resolve_budget()
        tracked_cost = selected_cost or 0.0
        session_key = session_id or ""
        current_spend = self._session_costs.get(session_key, 0.0)
        if budget is not None and session_key:
            projected_total = current_spend + tracked_cost
            if projected_total > budget + 1e-9:
                unit = selected_metadata.get("cost_unit") if isinstance(selected_metadata, MappingABC) else None
                unit_text = str(unit).strip() if unit else "credits"
                reason = (
                    f"Selecting tool '{selected_name}' would exceed the session budget of "
                    f"{_format_cost(budget)} {unit_text}."
                )
                return RouterDecision(
                    False,
                    reason=reason,
                    metadata=MappingProxyType(dict(selected_metadata)),
                )

        if session_key:
            self._session_costs[session_key] = current_spend + tracked_cost

        return RouterDecision(
            True,
            tool_name=selected_name,
            metadata=MappingProxyType(dict(selected_metadata)),
        )

    def reset_session(self, session_id: str) -> None:
        """Reset the tracked cost for ``session_id``."""

        self._session_costs.pop(session_id or "", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_budget(self) -> Optional[float]:
        manager = self._ensure_config_manager()
        if manager is None:
            return None

        section: Optional[Any] = None
        getter = getattr(manager, "get_config", None)
        if callable(getter):
            try:
                section = getter("tool_defaults", None)
            except TypeError:
                section = getter("tool_defaults")
        elif isinstance(getattr(manager, "config", None), MappingABC):
            section = manager.config.get("tool_defaults")  # type: ignore[attr-defined]

        if isinstance(section, MappingABC):
            candidate = section.get("max_cost_per_session")
            if isinstance(candidate, (int, float)):
                if candidate < 0:
                    return None
                return float(candidate)
            if candidate is None:
                return None
        return None

    def _ensure_config_manager(self) -> Optional[Any]:
        if self._config_manager is not None:
            return self._config_manager
        if ConfigManager is None:
            return None
        try:
            self._config_manager = ConfigManager()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to instantiate ConfigManager: %s", exc)
            self._config_manager = None
        return self._config_manager

