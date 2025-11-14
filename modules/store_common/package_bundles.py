"""Bundle helpers for packaging multiple capability assets together.

This module coordinates export and import flows for capability assets (tools,
skills, jobs, personas, and future types). It piggybacks on the existing
per-asset bundle helpers while providing a higher level package that can be
transported as a single signed payload.

The package format is JSON and contains:

    {
        "metadata": {...},
        "assets": [
            {
                "type": "tool",
                "name": "example",
                "persona": null,
                "encoding": "base64",
                "format": "json",
                "bundle": "..."  # base64 encoded bundle bytes
            },
            ...
        ],
        "signature": {
            "algorithm": "HS256",
            "value": "..."
        }
    }

Each asset bundle is the same payload that the per-asset bundlers already
produce. The outer package is signed so the consumer can verify integrity
before applying the individual imports.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from modules.Jobs import export_job_bundle_bytes, import_job_bundle_bytes
from modules.Personas import export_persona_bundle_bytes, import_persona_bundle_bytes
from modules.Skills import export_skill_bundle_bytes, import_skill_bundle_bytes
from modules.Tools import export_tool_bundle_bytes, import_tool_bundle_bytes
from modules.store_common.bundle_utils import (
    BUNDLE_ALGORITHM,
    sign_payload,
    utcnow_isoformat,
    verify_signature,
)

__all__ = [
    "AssetPackageError",
    "export_asset_package_bytes",
    "import_asset_package_bytes",
]


class AssetPackageError(ValueError):
    """Raised when asset package export or import fails."""


@dataclass(frozen=True)
class _AssetHandler:
    """Container for per-asset export/import helpers."""

    export: Callable[..., Tuple[bytes, Mapping[str, Any]]]
    importer: Callable[..., Mapping[str, Any]]
    supports_persona: bool = False


# Registry of supported asset handlers. Tasks are intentionally omitted for now
# but the structure allows future expansion without changing consumers.
_ASSET_HANDLERS: Mapping[str, _AssetHandler] = {
    "tool": _AssetHandler(export_tool_bundle_bytes, import_tool_bundle_bytes, True),
    "skill": _AssetHandler(export_skill_bundle_bytes, import_skill_bundle_bytes, True),
    "job": _AssetHandler(export_job_bundle_bytes, import_job_bundle_bytes, True),
    "persona": _AssetHandler(export_persona_bundle_bytes, import_persona_bundle_bytes, False),
}

# Import priority ensures personas run after their dependent capability types so
# validation sees the freshly restored tools/skills/jobs.
_ASSET_TYPE_PRIORITY: Mapping[str, int] = {
    "tool": 1,
    "skill": 2,
    "task": 3,  # placeholder priority for future support
    "job": 4,
    "persona": 5,
}


@dataclass
class _PackageAsset:
    """Normalized representation for a decoded package asset."""

    index: int
    asset_type: str
    name: str
    persona: Optional[str]
    bundle_bytes: bytes


def _normalize_persona(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value).strip() or None


def _prepare_export_kwargs(
    *,
    handler: _AssetHandler,
    name: str,
    persona: Optional[str],
    signing_key: str,
    config_manager,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Build keyword arguments for per-asset export helpers."""

    kwargs: Dict[str, Any] = {
        "signing_key": signing_key,
        "config_manager": config_manager,
    }
    normalized_persona = _normalize_persona(persona)
    if normalized_persona and not handler.supports_persona:
        raise AssetPackageError(
            f"Asset type does not support persona scoping: {handler}"
        )
    if handler.supports_persona and normalized_persona:
        kwargs["persona"] = normalized_persona
    return kwargs, normalized_persona


def _prepare_import_kwargs(
    *,
    handler: _AssetHandler,
    signing_key: str,
    config_manager,
    rationale: str,
) -> Dict[str, Any]:
    """Build keyword arguments for per-asset import helpers."""

    kwargs: Dict[str, Any] = {
        "signing_key": signing_key,
        "config_manager": config_manager,
        "rationale": rationale,
    }
    return kwargs


def _encode_bundle_bytes(bundle_bytes: bytes) -> str:
    return base64.b64encode(bundle_bytes).decode("ascii")


def _decode_bundle_entry(asset: Mapping[str, Any], *, index: int) -> _PackageAsset:
    asset_type = str(asset.get("type") or "").strip().lower()
    if not asset_type:
        raise AssetPackageError(f"Asset entry #{index + 1} is missing the 'type' field")
    handler = _ASSET_HANDLERS.get(asset_type)
    if handler is None:
        raise AssetPackageError(f"Unsupported asset type: {asset_type}")

    name = str(asset.get("name") or "").strip()
    if not name:
        raise AssetPackageError(f"Asset entry #{index + 1} is missing the 'name' field")

    persona = _normalize_persona(asset.get("persona"))

    encoding = str(asset.get("encoding") or "base64").strip().lower()
    if encoding != "base64":
        raise AssetPackageError(
            f"Asset '{name}' of type '{asset_type}' uses unsupported encoding: {encoding}"
        )

    bundle_data = asset.get("bundle")
    if not isinstance(bundle_data, str) or not bundle_data.strip():
        raise AssetPackageError(
            f"Asset '{name}' of type '{asset_type}' is missing bundle data"
        )
    try:
        bundle_bytes = base64.b64decode(bundle_data.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AssetPackageError(
            f"Asset '{name}' of type '{asset_type}' contains invalid base64 data"
        ) from exc

    return _PackageAsset(
        index=index,
        asset_type=asset_type,
        name=name,
        persona=persona,
        bundle_bytes=bundle_bytes,
    )


def export_asset_package_bytes(
    assets: Sequence[Mapping[str, Any]],
    *,
    signing_key: str,
    config_manager=None,
) -> Tuple[bytes, List[Mapping[str, Any]]]:
    """Export multiple capability assets into a single signed package."""

    if not isinstance(assets, Sequence) or isinstance(assets, (bytes, str)):
        raise AssetPackageError("Assets must be provided as a sequence of mappings")

    normalized_exports: List[Mapping[str, Any]] = []
    asset_entries: List[Mapping[str, Any]] = []

    for asset in assets:
        if not isinstance(asset, Mapping):
            raise AssetPackageError("Each asset entry must be a mapping")
        asset_type = str(asset.get("type") or "").strip().lower()
        if not asset_type:
            raise AssetPackageError("Asset entry is missing the 'type' field")
        handler = _ASSET_HANDLERS.get(asset_type)
        if handler is None:
            raise AssetPackageError(f"Unsupported asset type: {asset_type}")

        name = str(asset.get("name") or "").strip()
        if not name:
            raise AssetPackageError(f"Asset entry for type '{asset_type}' is missing the name")

        persona = asset.get("persona")
        export_kwargs, normalized_persona = _prepare_export_kwargs(
            handler=handler,
            name=name,
            persona=persona,
            signing_key=signing_key,
            config_manager=config_manager,
        )

        try:
            bundle_bytes, manifest_entry = handler.export(name, **export_kwargs)
        except Exception as exc:
            raise AssetPackageError(
                f"Failed to export {asset_type} '{name}': {exc}"
            ) from exc

        asset_entries.append(
            {
                "type": asset_type,
                "name": name,
                "persona": normalized_persona,
                "encoding": "base64",
                "format": "json",
                "bundle": _encode_bundle_bytes(bundle_bytes),
            }
        )
        normalized_exports.append(
            {
                "type": asset_type,
                "name": name,
                "persona": normalized_persona,
                "manifest": dict(manifest_entry),
            }
        )

    metadata: Dict[str, Any] = {
        "version": 1,
        "exported_at": utcnow_isoformat(),
        "asset_count": len(asset_entries),
    }

    payload: Dict[str, Any] = {
        "metadata": metadata,
        "assets": asset_entries,
    }

    signature = sign_payload(payload, signing_key=signing_key, error_cls=AssetPackageError)
    package_payload = {
        **payload,
        "signature": {
            "algorithm": BUNDLE_ALGORITHM,
            "value": signature,
        },
    }

    return json.dumps(package_payload, indent=2).encode("utf-8"), normalized_exports


def import_asset_package_bytes(
    package_bytes: bytes,
    *,
    signing_key: str,
    config_manager=None,
    rationale: str = "Imported asset package",
) -> Dict[str, Any]:
    """Import a previously exported asset package."""

    try:
        payload = json.loads(package_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise AssetPackageError("Asset package is not valid UTF-8 data.") from exc
    except json.JSONDecodeError as exc:
        raise AssetPackageError("Asset package payload is not valid JSON.") from exc

    if not isinstance(payload, MutableMapping):
        raise AssetPackageError("Asset package payload must be a JSON object.")

    metadata = payload.get("metadata")
    assets = payload.get("assets")
    signature_info = payload.get("signature")

    if not isinstance(metadata, Mapping):
        raise AssetPackageError("Asset package metadata is missing or invalid.")
    if not isinstance(assets, list):
        raise AssetPackageError("Asset package assets must be provided as a list.")
    if not isinstance(signature_info, Mapping):
        raise AssetPackageError("Asset package signature block is missing or invalid.")

    algorithm = signature_info.get("algorithm")
    signature_value = signature_info.get("value")
    if algorithm != BUNDLE_ALGORITHM:
        raise AssetPackageError(f"Unsupported asset package algorithm: {algorithm!r}")
    if not isinstance(signature_value, str) or not signature_value.strip():
        raise AssetPackageError("Asset package signature is missing.")

    payload_for_signature = {
        "metadata": dict(metadata),
        "assets": [dict(item) for item in assets],
    }

    verify_signature(
        payload_for_signature,
        signature=signature_value,
        signing_key=signing_key,
        error_cls=AssetPackageError,
    )

    decoded_assets: List[_PackageAsset] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise AssetPackageError(f"Asset entry #{index + 1} must be a JSON object")
        decoded_assets.append(_decode_bundle_entry(asset, index=index))

    ordered_assets = sorted(
        decoded_assets,
        key=lambda item: (
            _ASSET_TYPE_PRIORITY.get(item.asset_type, 100),
            item.index,
        ),
    )

    results: List[Dict[str, Any]] = []
    overall_success = True
    for asset in ordered_assets:
        handler = _ASSET_HANDLERS[asset.asset_type]
        import_kwargs = _prepare_import_kwargs(
            handler=handler,
            signing_key=signing_key,
            config_manager=config_manager,
            rationale=rationale,
        )
        try:
            result = handler.importer(asset.bundle_bytes, **import_kwargs)
        except Exception as exc:
            raise AssetPackageError(
                f"Failed to import {asset.asset_type} '{asset.name}': {exc}"
            ) from exc

        if isinstance(result, Mapping):
            success_value = result.get("success")
            if success_value is not None:
                overall_success = overall_success and bool(success_value)
        results.append(
            {
                "type": asset.asset_type,
                "name": asset.name,
                "persona": asset.persona,
                "result": result,
            }
        )

    return {
        "success": overall_success,
        "metadata": dict(metadata),
        "results": results,
    }
