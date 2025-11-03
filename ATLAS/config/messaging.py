"""Messaging configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping
from collections.abc import Mapping


@dataclass
class MessagingConfigSection:
    """Normalize messaging backend configuration."""

    config: MutableMapping[str, Any]
    yaml_config: MutableMapping[str, Any]
    env_config: Mapping[str, Any]
    logger: Any
    write_yaml_callback: Callable[[], None]

    def apply(self) -> None:
        messaging_block = self.config.get("messaging")
        if not isinstance(messaging_block, Mapping):
            messaging_block = {}
        else:
            messaging_block = dict(messaging_block)

        backend_name = str(messaging_block.get("backend") or "in_memory").lower()
        messaging_block["backend"] = backend_name
        if backend_name == "redis":
            default_url = self.env_config.get("REDIS_URL", "redis://localhost:6379/0")
            messaging_block.setdefault("redis_url", default_url)
            messaging_block.setdefault("stream_prefix", "atlas_bus")

        self.config["messaging"] = messaging_block

    def get_settings(self) -> dict[str, Any]:
        configured = self.config.get("messaging")
        if isinstance(configured, Mapping):
            return dict(configured)
        return {"backend": "in_memory"}

    def set_settings(
        self,
        *,
        backend: str,
        redis_url: str | None = None,
        stream_prefix: str | None = None,
    ) -> dict[str, Any]:
        sanitized_backend = str(backend or "in_memory").strip().lower()
        if sanitized_backend not in {"in_memory", "redis"}:
            sanitized_backend = "in_memory"

        block: dict[str, Any] = {"backend": sanitized_backend}
        if sanitized_backend == "redis":
            if redis_url:
                block["redis_url"] = str(redis_url).strip()
            if stream_prefix:
                block["stream_prefix"] = str(stream_prefix).strip()

        self.yaml_config["messaging"] = dict(block)
        self.config["messaging"] = dict(block)
        self.write_yaml_callback()
        return dict(block)
