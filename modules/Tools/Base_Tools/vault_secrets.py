"""Vault secrets management tool."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

__all__ = ["VaultSecretsTool"]

logger = setup_logger(__name__)


class VaultSecretsTool:
    """Expose masked credential snapshots and persistence helpers."""

    def __init__(self, *, config_manager: Optional[ConfigManager] = None) -> None:
        self._config_manager = config_manager or ConfigManager()

    async def run(
        self,
        *,
        operation: str,
        tool_name: Optional[str] = None,
        tool_names: Optional[Iterable[str]] = None,
        manifest_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
        manifest_auth: Optional[Mapping[str, Any]] = None,
        credentials: Optional[Mapping[str, Any]] = None,
        env_keys: Optional[Iterable[str]] = None,
        persona: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Dispatch vault operations for credentials."""

        if not isinstance(operation, str):
            raise TypeError("operation must be provided as a string")

        normalized_operation = operation.strip().lower()
        if not normalized_operation:
            raise ValueError("operation must be a non-empty string")

        if normalized_operation == "get_snapshot":
            return self._handle_snapshot(
                tool_names=tool_names,
                manifest_lookup=manifest_lookup,
                persona=persona,
            )

        if normalized_operation == "store_credentials":
            return self._handle_store(
                tool_name=tool_name,
                credentials=credentials,
                manifest_auth=manifest_auth,
                persona=persona,
            )

        if normalized_operation == "clear_credentials":
            return self._handle_clear(
                tool_name=tool_name,
                credentials=credentials,
                env_keys=env_keys,
                manifest_auth=manifest_auth,
                persona=persona,
            )

        raise ValueError(f"Unsupported vault secrets operation: {operation!r}")

    def _handle_snapshot(
        self,
        *,
        tool_names: Optional[Iterable[str]],
        manifest_lookup: Optional[Mapping[str, Mapping[str, Any]]],
        persona: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        snapshot = self._config_manager.get_tool_config_snapshot(
            manifest_lookup=manifest_lookup,
            tool_names=tool_names,
        )

        logger.info(
            "Vault secrets snapshot retrieved.",
            extra={
                "event": "vault_secrets.snapshot",
                "persona": persona,
                "tools": tuple(sorted(snapshot.keys())),
            },
        )

        return snapshot

    def _handle_store(
        self,
        *,
        tool_name: Optional[str],
        credentials: Optional[Mapping[str, Any]],
        manifest_auth: Optional[Mapping[str, Any]],
        persona: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        normalized_name = self._normalize_tool_name(tool_name)
        normalized_credentials = self._normalize_credentials(credentials)
        if not normalized_credentials:
            raise ValueError("credentials payload must include at least one entry")

        logger.info(
            "Persisting vault secrets.",
            extra={
                "event": "vault_secrets.store",
                "persona": persona,
                "tool_name": normalized_name,
                "keys": tuple(sorted(normalized_credentials.keys())),
            },
        )

        return self._config_manager.set_tool_credentials(
            normalized_name,
            normalized_credentials,
            manifest_auth=manifest_auth,
        )

    def _handle_clear(
        self,
        *,
        tool_name: Optional[str],
        credentials: Optional[Mapping[str, Any]],
        env_keys: Optional[Iterable[str]],
        manifest_auth: Optional[Mapping[str, Any]],
        persona: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        normalized_name = self._normalize_tool_name(tool_name)
        sanitized_keys = self._normalize_keys(env_keys)

        if credentials:
            sanitized_keys.extend(
                key for key in self._normalize_credentials(credentials).keys() if key not in sanitized_keys
            )

        if not sanitized_keys:
            raise ValueError("env_keys or credentials must supply keys to clear")

        payload = {key: None for key in sanitized_keys}

        logger.info(
            "Clearing vault secrets.",
            extra={
                "event": "vault_secrets.clear",
                "persona": persona,
                "tool_name": normalized_name,
                "keys": tuple(sorted(sanitized_keys)),
            },
        )

        return self._config_manager.set_tool_credentials(
            normalized_name,
            payload,
            manifest_auth=manifest_auth,
        )

    @staticmethod
    def _normalize_tool_name(name: Optional[str]) -> str:
        token = (name or "").strip()
        if not token:
            raise ValueError("tool_name is required for this operation")
        return token

    @staticmethod
    def _normalize_credentials(
        credentials: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        if not credentials:
            return normalized
        for raw_key, value in credentials.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip()
            if not key:
                continue
            normalized[key] = value
        return normalized

    @staticmethod
    def _normalize_keys(keys: Optional[Iterable[Any]]) -> list[str]:
        normalized: list[str] = []
        if not keys:
            return normalized
        for raw in keys:
            if not isinstance(raw, str):
                continue
            key = raw.strip()
            if key and key not in normalized:
                normalized.append(key)
        return normalized
