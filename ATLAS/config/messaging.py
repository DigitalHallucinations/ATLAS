"""Messaging configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, Tuple
from collections.abc import Mapping

from modules.orchestration.message_bus import (
    InMemoryQueueBackend,
    MessageBus,
    RedisStreamBackend,
    configure_message_bus,
)


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
            raw_offset = messaging_block.get("initial_offset")
            if raw_offset is None:
                raw_offset = messaging_block.get("initial_stream_id")
            messaging_block["initial_offset"] = _normalize_initial_offset(raw_offset)
            if "initial_stream_id" in messaging_block:
                messaging_block["initial_stream_id"] = messaging_block["initial_offset"]

        self.config["messaging"] = messaging_block

    def get_settings(self) -> dict[str, Any]:
        configured = self.config.get("messaging")
        if isinstance(configured, Mapping):
            block = dict(configured)
            if block.get("backend") == "redis":
                raw_offset = block.get("initial_offset")
                if raw_offset is None:
                    raw_offset = block.get("initial_stream_id")
                block["initial_offset"] = _normalize_initial_offset(raw_offset)
            return block
        return {"backend": "in_memory", "initial_offset": "$"}

    def set_settings(
        self,
        *,
        backend: str,
        redis_url: str | None = None,
        stream_prefix: str | None = None,
        initial_offset: str | None = None,
        initial_stream_id: str | None = None,
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
            normalized_offset = _normalize_initial_offset(
                initial_offset if initial_offset is not None else initial_stream_id
            )
            block["initial_offset"] = normalized_offset
            block["initial_stream_id"] = normalized_offset

        self.yaml_config["messaging"] = dict(block)
        self.config["messaging"] = dict(block)
        self.write_yaml_callback()
        return dict(block)


def setup_message_bus(settings: Mapping[str, Any], *, logger: Any) -> Tuple[Any, MessageBus]:
    """Instantiate a message bus backend according to the provided settings."""

    backend_name = str(settings.get("backend", "in_memory") or "in_memory").lower()

    backend: Any
    if backend_name == "redis":
        redis_url = settings.get("redis_url")
        stream_prefix = settings.get("stream_prefix", "atlas_bus")
        initial_offset = _normalize_initial_offset(
            settings.get("initial_offset") or settings.get("initial_stream_id")
        )
        try:
            backend = RedisStreamBackend(
                str(redis_url),
                stream_prefix=str(stream_prefix),
                initial_offset=str(initial_offset),
            )
        except Exception as exc:  # pragma: no cover - Redis optional dependency
            logger.warning(
                "Falling back to in-memory message bus backend due to Redis configuration error: %s",
                exc,
            )
            backend = InMemoryQueueBackend()
    else:
        backend = InMemoryQueueBackend()

    bus = configure_message_bus(backend)
    return backend, bus


_STREAM_ID_PATTERN = re.compile(r"^\d+-\d+$")


def _normalize_initial_offset(value: Any | None) -> str:
    candidate = (str(value).strip() if value is not None else "") or "$"
    if candidate == "$":
        return "$"
    if _STREAM_ID_PATTERN.match(candidate):
        return candidate
    return "$"
