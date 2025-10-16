"""Lightweight server routing helpers for tool metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

from modules.Tools.manifest_loader import ToolManifestEntry, load_manifest_entries
from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


class AtlasServer:
    """Expose read-only endpoints for tool metadata."""

    def get_tools(
        self,
        *,
        capability: Optional[Any] = None,
        safety_level: Optional[Any] = None,
        persona: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return merged tool metadata, optionally filtered."""

        entries = load_manifest_entries()
        filtered = _filter_entries(
            entries,
            capability_tokens=_normalize_filters(capability),
            safety_tokens=_normalize_filters(safety_level),
            persona_tokens=_normalize_filters(persona),
        )

        return {
            "count": len(filtered),
            "tools": [_serialize_entry(entry) for entry in filtered],
        }

    def handle_request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simple request dispatcher supporting ``GET /tools``."""

        if method.upper() != "GET":
            raise ValueError(f"Unsupported method: {method}")

        if path != "/tools":
            raise ValueError(f"Unsupported path: {path}")

        query = query or {}
        return self.get_tools(
            capability=query.get("capability"),
            safety_level=query.get("safety_level"),
            persona=query.get("persona"),
        )


def _filter_entries(
    entries: Iterable[ToolManifestEntry],
    *,
    capability_tokens: List[str],
    safety_tokens: List[str],
    persona_tokens: List[str],
) -> List[ToolManifestEntry]:
    filtered: List[ToolManifestEntry] = []
    for entry in entries:
        if capability_tokens and not _capabilities_match(entry, capability_tokens):
            continue
        if safety_tokens and not _safety_matches(entry, safety_tokens):
            continue
        if persona_tokens and not _persona_matches(entry, persona_tokens):
            continue
        filtered.append(entry)
    return filtered


def _capabilities_match(entry: ToolManifestEntry, tokens: List[str]) -> bool:
    capabilities = {cap.lower() for cap in entry.capabilities}
    return all(token in capabilities for token in tokens)


def _safety_matches(entry: ToolManifestEntry, tokens: List[str]) -> bool:
    safety = (entry.safety_level or "").lower()
    return bool(safety) and safety in tokens


def _persona_matches(entry: ToolManifestEntry, tokens: List[str]) -> bool:
    persona_token = (entry.persona or "shared").lower()
    if persona_token == "shared":
        persona_aliases = {"shared", "default", "global", ""}
        return any(token in persona_aliases for token in tokens)
    return persona_token in tokens


def _normalize_filters(values: Optional[Any]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    elif isinstance(values, Mapping):
        values = list(values.values())
    elif not isinstance(values, Iterable):
        values = [values]

    tokens: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip().lower()
        else:
            text = str(value).strip().lower()
        if text:
            tokens.append(text)
    return tokens


def _serialize_entry(entry: ToolManifestEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "persona": entry.persona,
        "description": entry.description,
        "version": entry.version,
        "capabilities": entry.capabilities,
        "auth": entry.auth,
        "auth_required": entry.auth_required,
        "safety_level": entry.safety_level,
        "requires_consent": entry.requires_consent,
        "allow_parallel": entry.allow_parallel,
        "idempotency_key": entry.idempotency_key,
        "default_timeout": entry.default_timeout,
        "side_effects": entry.side_effects,
        "cost_per_call": entry.cost_per_call,
        "cost_unit": entry.cost_unit,
        "persona_allowlist": entry.persona_allowlist,
        "providers": entry.providers,
        "source": entry.source,
    }


atlas_server = AtlasServer()

__all__ = ["AtlasServer", "atlas_server"]
