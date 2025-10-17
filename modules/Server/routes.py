"""Lightweight server routing helpers for tool metadata and analytics."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional

from modules.Personas import (
    PersonaBundleError,
    PersonaValidationError,
    load_persona_definition,
    load_tool_metadata,
    normalize_allowed_tools,
    persist_persona_definition,
    export_persona_bundle_bytes,
    import_persona_bundle_bytes,
    _validate_persona_payload,
)
from modules.Tools.manifest_loader import ToolManifestEntry, load_manifest_entries
from modules.analytics.persona_metrics import get_persona_metrics
from modules.logging.audit import (
    get_persona_audit_logger,
    get_persona_review_logger,
    get_persona_review_queue,
    parse_persona_timestamp,
)
from modules.logging.logger import setup_logger
from modules.persona_review import REVIEW_INTERVAL_DAYS, compute_review_status

logger = setup_logger(__name__)


def _parse_query_timestamp(raw_value: Optional[Any]) -> Optional[datetime]:
    """Return a normalized UTC timestamp parsed from query parameters."""

    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class AtlasServer:
    """Expose read-only endpoints for tool metadata."""

    def __init__(self, *, config_manager: Optional[object] = None) -> None:
        self._config_manager = config_manager

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

    def get_persona_metrics(
        self,
        persona_name: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Return aggregated persona metrics for the requested persona."""

        if not persona_name:
            raise ValueError("Persona name is required for analytics")

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 20

        limit_value = max(1, min(limit_value, 200))

        return get_persona_metrics(
            persona_name,
            start=start,
            end=end,
            limit_recent=limit_value,
            config_manager=self._config_manager,
        )

    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]:
        """Return the current review status for ``persona_name``."""

        if not persona_name:
            raise ValueError("Persona name is required for review status")

        status = compute_review_status(
            persona_name,
            audit_logger=get_persona_audit_logger(),
            review_logger=get_persona_review_logger(),
            review_queue=get_persona_review_queue(),
        )

        payload = asdict(status)
        payload["success"] = True
        return payload

    def attest_persona_review(
        self,
        persona_name: str,
        *,
        reviewer: str,
        expires_at: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a persona review attestation for ``persona_name``."""

        if not persona_name:
            raise ValueError("Persona name is required for review attestation")

        reviewer_name = reviewer.strip() if reviewer else ""
        if not reviewer_name:
            reviewer_name = "unknown"

        validity_days = (
            int(expires_in_days)
            if expires_in_days is not None
            else REVIEW_INTERVAL_DAYS
        )
        validity_days = max(1, validity_days)
        validity = timedelta(days=validity_days)

        parsed_expires = (
            parse_persona_timestamp(expires_at)
            if expires_at
            else None
        )

        review_logger = get_persona_review_logger()
        queue = get_persona_review_queue()

        attestation = review_logger.record_attestation(
            persona_name,
            reviewer=reviewer_name,
            expires_at=parsed_expires,
            notes=notes,
            validity=validity,
        )

        now = datetime.now(timezone.utc)
        queue.mark_completed(persona_name, timestamp=now)

        status = compute_review_status(
            persona_name,
            audit_logger=get_persona_audit_logger(),
            review_logger=review_logger,
            review_queue=queue,
            now=now,
            interval_days=validity_days,
        )

        return {
            "success": True,
            "attestation": asdict(attestation),
            "status": asdict(status),
        }

    def handle_request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simple request dispatcher supporting persona metadata endpoints."""

        method_upper = method.upper()
        query = query or {}

        if method_upper == "GET":
            if path.startswith("/personas/") and path.endswith("/analytics"):
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) != 3:
                    raise ValueError(f"Unsupported path: {path}")
                persona_name = components[1]
                start = _parse_query_timestamp(query.get("start"))
                end = _parse_query_timestamp(query.get("end"))
                limit = query.get("limit")
                return self.get_persona_metrics(
                    persona_name,
                    start=start,
                    end=end,
                    limit=limit,
                )
            if path.startswith("/personas/") and path.endswith("/review"):
                components = [part for part in path.strip("/").split("/") if part]
                if len(components) != 3:
                    raise ValueError(f"Unsupported path: {path}")
                persona_name = components[1]
                return self.get_persona_review_status(persona_name)
            if path != "/tools":
                raise ValueError(f"Unsupported path: {path}")
            return self.get_tools(
                capability=query.get("capability"),
                safety_level=query.get("safety_level"),
                persona=query.get("persona"),
            )

        if method_upper == "POST":
            return self._handle_post(path, query)

        raise ValueError(f"Unsupported method: {method}")

    def _handle_post(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if path.startswith("/personas/") and path.endswith("/tools"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            return self.update_persona_tools(
                persona_name,
                tools=payload.get("tools"),
                rationale=str(payload.get("rationale") or "Server route persona update"),
            )

        if path.startswith("/personas/") and path.endswith("/export"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            return self.export_persona_bundle(
                persona_name,
                signing_key=str(payload.get("signing_key") or ""),
            )

        if path == "/personas/import":
            return self.import_persona_bundle(
                bundle_base64=str(payload.get("bundle") or ""),
                signing_key=str(payload.get("signing_key") or ""),
                rationale=str(payload.get("rationale") or "Imported via server route"),
            )

        if path.startswith("/personas/") and path.endswith("/review"):
            components = [part for part in path.strip("/").split("/") if part]
            if len(components) != 3:
                raise ValueError(f"Unsupported path: {path}")
            persona_name = components[1]
            expires_at = payload.get("expires_at")
            expires_in = payload.get("expires_in_days")
            reviewer = payload.get("reviewer")
            notes = payload.get("notes")
            return self.attest_persona_review(
                persona_name,
                reviewer=str(reviewer or ""),
                expires_at=str(expires_at) if expires_at else None,
                expires_in_days=int(expires_in) if expires_in is not None else None,
                notes=str(notes) if notes is not None else None,
            )

        raise ValueError(f"Unsupported path: {path}")

    def update_persona_tools(
        self,
        persona_name: str,
        *,
        tools: Optional[Any],
        rationale: str = "Server route persona update",
    ) -> Dict[str, Any]:
        """Update the allowed tools for a persona via server APIs."""

        if not persona_name:
            raise ValueError("Persona name is required")

        persona = load_persona_definition(
            persona_name,
            config_manager=self._config_manager,
        )
        if persona is None:
            raise ValueError(f"Persona '{persona_name}' could not be loaded.")

        metadata_order, metadata_lookup = load_tool_metadata(
            config_manager=self._config_manager
        )
        normalised_tools = normalize_allowed_tools(
            self._normalise_tool_payload(tools),
            metadata_order=metadata_order,
        )

        persona_for_validation = dict(persona)
        persona_for_validation["allowed_tools"] = normalised_tools

        known_tools: set[str] = {str(name) for name in metadata_order}
        known_tools.update(str(name) for name in metadata_lookup.keys())
        existing_tools = persona.get("allowed_tools") or []
        known_tools.update(str(name) for name in existing_tools if str(name))

        try:
            _validate_persona_payload(
                {"persona": [persona_for_validation]},
                persona_name=persona_name,
                tool_ids=known_tools,
                config_manager=self._config_manager,
            )
        except PersonaValidationError as exc:
            message = str(exc)
            logger.warning(
                "Rejected persona tool update for '%s': %s",
                persona_name,
                message,
            )
            return {"success": False, "error": message, "errors": [message]}

        persona["allowed_tools"] = normalised_tools

        persist_persona_definition(
            persona_name,
            persona,
            config_manager=self._config_manager,
            rationale=rationale,
        )

        return {
            "success": True,
            "persona": {
                "name": persona.get("name", persona_name),
                "allowed_tools": normalised_tools,
            },
        }

    @staticmethod
    def _normalise_tool_payload(raw: Optional[Any]) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, Mapping):
            return list(raw.values())
        if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
            return list(raw)
        return [str(raw)]

    def export_persona_bundle(
        self,
        persona_name: str,
        *,
        signing_key: str,
    ) -> Dict[str, Any]:
        if not persona_name:
            raise ValueError("Persona name is required for export")

        try:
            bundle_bytes, persona = export_persona_bundle_bytes(
                persona_name,
                signing_key=signing_key,
                config_manager=self._config_manager,
            )
        except PersonaBundleError as exc:
            return {"success": False, "error": str(exc)}

        encoded = base64.b64encode(bundle_bytes).decode("ascii")
        return {
            "success": True,
            "persona": persona,
            "bundle": encoded,
        }

    def import_persona_bundle(
        self,
        *,
        bundle_base64: str,
        signing_key: str,
        rationale: str = "Imported via server route",
    ) -> Dict[str, Any]:
        if not bundle_base64:
            raise ValueError("Bundle payload is required for import")

        try:
            bundle_bytes = base64.b64decode(bundle_base64)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Bundle payload is not valid base64 data") from exc

        try:
            result = import_persona_bundle_bytes(
                bundle_bytes,
                signing_key=signing_key,
                config_manager=self._config_manager,
                rationale=rationale,
            )
        except PersonaBundleError as exc:
            return {"success": False, "error": str(exc)}

        result.setdefault("success", True)
        return result


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
    shared_exclusions = {
        "-shared",
        "!shared",
        "no-shared",
        "without-shared",
        "shared=false",
        "shared:false",
    }
    exclude_shared = any(token in shared_exclusions for token in tokens)
    positive_tokens = [token for token in tokens if token not in shared_exclusions]
    if persona_token == "shared":
        return not exclude_shared
    if not positive_tokens:
        return True
    return persona_token in positive_tokens


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
