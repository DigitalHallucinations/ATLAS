"""Helpers for translating MCP tool metadata into ATLAS manifest entries."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional


DEFAULT_SIDE_EFFECTS = "network"
DEFAULT_TIMEOUT = 30
DEFAULT_ALLOW_PARALLEL = False
DEFAULT_IDEMPOTENCY_KEY = False
DEFAULT_REQUIRES_CONSENT = True


def _coerce_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        result = int(float(value))
        return result if result >= 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_mapping(value: Any) -> MutableMapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_schema(schema: Any) -> MutableMapping[str, Any]:
    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}

    normalized: MutableMapping[str, Any] = dict(schema)
    schema_type = str(normalized.get("type") or "object").strip().lower()
    normalized["type"] = "object" if schema_type != "object" else "object"

    properties = normalized.get("properties")
    if isinstance(properties, Mapping):
        normalized["properties"] = dict(properties)
    else:
        normalized["properties"] = {}

    required = normalized.get("required")
    if isinstance(required, Iterable) and not isinstance(required, (str, bytes)):
        normalized["required"] = [
            str(item).strip()
            for item in required
            if isinstance(item, (str, bytes)) and str(item).strip()
        ]
    elif isinstance(required, str) and required.strip():
        normalized["required"] = [required.strip()]

    return normalized


def _collect_scopes(*values: Any) -> list[str]:
    scopes: list[str] = []
    for value in values:
        candidate = value
        if isinstance(candidate, Mapping):
            candidate = candidate.get("scopes") or candidate.get("scope")
        if candidate is None:
            continue
        if isinstance(candidate, str):
            tokens = [token.strip() for token in candidate.split() if token.strip()]
        elif isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
            tokens = [str(item).strip() for item in candidate if str(item).strip()]
        else:
            continue
        for token in tokens:
            if token not in scopes:
                scopes.append(token)
    return scopes


def _resolve_auth(
    tool_metadata: Mapping[str, Any],
    server_config: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    tool_auth = _coerce_mapping(
        tool_metadata.get("auth") or tool_metadata.get("authentication")
    )
    server_auth = _coerce_mapping(
        server_config.get("auth") or server_config.get("authentication")
    )

    auth: MutableMapping[str, Any] = {}

    tool_required = _coerce_bool(
        tool_metadata.get("requires_authentication"),
        default=_coerce_bool(tool_auth.get("required")),
    )
    server_required = _coerce_bool(server_config.get("auth_required"))
    required = (
        tool_required
        if tool_required is not None
        else _coerce_bool(server_auth.get("required"), default=server_required)
    )

    scopes = _collect_scopes(tool_auth, server_auth, server_config.get("scopes"))
    if required is None and scopes:
        required = True
    auth["required"] = bool(required) if required is not None else False

    auth_type = (
        tool_auth.get("type")
        or tool_auth.get("scheme")
        or server_auth.get("type")
        or server_auth.get("scheme")
        or server_config.get("auth_type")
    )
    if isinstance(auth_type, str) and auth_type.strip():
        auth["type"] = auth_type.strip()

    if scopes:
        auth["scopes"] = scopes

    env_block = server_auth.get("env") or server_config.get("auth_env")
    if isinstance(env_block, Mapping):
        auth["env"] = dict(env_block)

    return auth


def translate_mcp_tool_to_manifest(
    tool_metadata: Mapping[str, Any],
    *,
    server_name: str,
    server_config: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    """Convert MCP tool metadata into a manifest entry."""

    server_config = server_config or {}
    defaults = defaults or {}

    tool_name = str(tool_metadata.get("name") or "").strip()
    if not tool_name:
        raise ValueError("MCP tool metadata is missing a name")

    normalized_server = str(server_name or "").strip()
    manifest_name = f"mcp.{normalized_server}.{tool_name}" if normalized_server else f"mcp.{tool_name}"

    description = str(tool_metadata.get("description") or "").strip()
    if not description:
        description = f"MCP tool '{tool_name}' from server '{normalized_server or 'default'}'"

    parameters = _normalize_schema(
        tool_metadata.get("input_schema")
        or tool_metadata.get("inputSchema")
        or tool_metadata.get("parameters")
    )

    timeout_default = server_config.get("timeout_seconds") or server_config.get("timeout")
    timeout_default = timeout_default or defaults.get("timeout_seconds") or DEFAULT_TIMEOUT
    default_timeout = _coerce_int(timeout_default, DEFAULT_TIMEOUT)

    side_effects = str(
        server_config.get("side_effects")
        or defaults.get("side_effects")
        or DEFAULT_SIDE_EFFECTS
    ).strip() or DEFAULT_SIDE_EFFECTS

    allow_parallel = _coerce_bool(
        server_config.get("allow_parallel"),
        default=_coerce_bool(defaults.get("allow_parallel"), DEFAULT_ALLOW_PARALLEL),
    )
    requires_consent = _coerce_bool(
        server_config.get("requires_consent"),
        default=_coerce_bool(defaults.get("requires_consent"), DEFAULT_REQUIRES_CONSENT),
    )
    idempotency_key = server_config.get("idempotency_key", DEFAULT_IDEMPOTENCY_KEY)

    auth_block = _resolve_auth(tool_metadata, server_config)

    entry: MutableMapping[str, Any] = {
        "name": manifest_name,
        "description": description,
        "parameters": parameters,
        "version": str(
            tool_metadata.get("version")
            or server_config.get("version")
            or defaults.get("version")
            or "1.0.0"
        ),
        "side_effects": side_effects,
        "default_timeout": default_timeout,
        "auth": auth_block,
        "allow_parallel": bool(allow_parallel) if allow_parallel is not None else DEFAULT_ALLOW_PARALLEL,
        "idempotency_key": idempotency_key if idempotency_key is not None else DEFAULT_IDEMPOTENCY_KEY,
        "requires_consent": bool(requires_consent) if requires_consent is not None else DEFAULT_REQUIRES_CONSENT,
        "providers": [
            {
                "name": "mcp",
                "config": {
                    "server": normalized_server or server_name,
                    "tool": tool_name,
                },
            }
        ],
        "capabilities": list(tool_metadata.get("capabilities") or []),
    }
    return entry


__all__ = ["translate_mcp_tool_to_manifest"]
