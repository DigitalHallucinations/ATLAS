"""Messaging configuration helpers for :mod:`ATLAS.config`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, MutableMapping, Sequence, Tuple
from collections.abc import Mapping

from modules.orchestration.message_bus import (
    InMemoryQueueBackend,
    MessageBus,
    RedisStreamGroupBackend,
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
            raw_replay = messaging_block.get("replay_start")
            if raw_replay is None:
                raw_replay = messaging_block.get("initial_offset")
            if raw_replay is None:
                raw_replay = messaging_block.get("initial_stream_id")
            messaging_block["replay_start"] = _normalize_replay_start(raw_replay)
            messaging_block["initial_offset"] = messaging_block["replay_start"]
            messaging_block["initial_stream_id"] = messaging_block["replay_start"]
            messaging_block["batch_size"] = _coerce_positive_int(messaging_block.get("batch_size"), default=1)
            messaging_block["blocking_timeout_ms"] = _coerce_positive_int(
                messaging_block.get("blocking_timeout_ms"), default=1000
            )
            messaging_block["auto_claim_idle_ms"] = _coerce_positive_int(
                messaging_block.get("auto_claim_idle_ms"), default=60_000, allow_zero=True
            )
            messaging_block["auto_claim_count"] = _coerce_positive_int(
                messaging_block.get("auto_claim_count"), default=10
            )
            delete_ack = messaging_block.get("delete_acknowledged")
            messaging_block["delete_acknowledged"] = True if delete_ack is None else bool(delete_ack)
            if "trim_max_length" in messaging_block:
                messaging_block["trim_max_length"] = _coerce_positive_int(
                    messaging_block.get("trim_max_length"), default=None
                )

        kafka_block = _normalize_kafka_block(messaging_block.get("kafka"), env_config=self.env_config)
        messaging_block["kafka"] = kafka_block

        self.config["messaging"] = messaging_block

    def get_settings(self) -> dict[str, Any]:
        configured = self.config.get("messaging")
        if isinstance(configured, Mapping):
            block = dict(configured)
            if block.get("backend") == "redis":
                raw_offset = block.get("replay_start") or block.get("initial_offset")
                if raw_offset is None:
                    raw_offset = block.get("initial_stream_id")
                normalized_replay = _normalize_replay_start(raw_offset)
                block["replay_start"] = normalized_replay
                block["initial_offset"] = normalized_replay
                block["initial_stream_id"] = normalized_replay
            if block.get("kafka") is not None:
                block["kafka"] = _normalize_kafka_block(block.get("kafka"), env_config=self.env_config)
            return block
        return {"backend": "in_memory", "replay_start": "$", "initial_offset": "$"}

    def set_settings(
        self,
        *,
        backend: str,
        redis_url: str | None = None,
        stream_prefix: str | None = None,
        initial_offset: str | None = None,
        initial_stream_id: str | None = None,
        replay_start: str | None = None,
        batch_size: int | None = None,
        blocking_timeout_ms: int | None = None,
        auto_claim_idle_ms: int | None = None,
        auto_claim_count: int | None = None,
        delete_acknowledged: bool | None = None,
        trim_max_length: int | None = None,
        kafka: Mapping[str, Any] | None = None,
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
            normalized_replay = _normalize_replay_start(
                replay_start if replay_start is not None else initial_offset or initial_stream_id
            )
            block["replay_start"] = normalized_replay
            block["initial_offset"] = normalized_replay
            block["initial_stream_id"] = normalized_replay
            if batch_size is not None:
                block["batch_size"] = _coerce_positive_int(batch_size, default=1)
            if blocking_timeout_ms is not None:
                block["blocking_timeout_ms"] = _coerce_positive_int(blocking_timeout_ms, default=1000)
            if auto_claim_idle_ms is not None:
                block["auto_claim_idle_ms"] = _coerce_positive_int(
                    auto_claim_idle_ms, default=60_000, allow_zero=True
                )
            if auto_claim_count is not None:
                block["auto_claim_count"] = _coerce_positive_int(auto_claim_count, default=10)
            if delete_acknowledged is not None:
                block["delete_acknowledged"] = bool(delete_acknowledged)
            if trim_max_length is not None:
                block["trim_max_length"] = _coerce_positive_int(trim_max_length, default=None)
        if kafka is not None:
            block["kafka"] = _normalize_kafka_block(kafka, env_config=self.env_config)

        self.yaml_config["messaging"] = dict(block)
        self.config["messaging"] = dict(block)
        self.write_yaml_callback()
        return dict(block)


def setup_message_bus(settings: Mapping[str, Any], *, logger: Any) -> Tuple[Any, MessageBus]:
    """Instantiate a message bus backend according to the provided settings.

    Recognized Redis settings include:
    ``redis_url``, ``stream_prefix``, ``replay_start``/``initial_offset``,
    ``batch_size``, ``blocking_timeout_ms``, ``auto_claim_idle_ms``,
    ``auto_claim_count``, ``delete_acknowledged``, and ``trim_max_length``.
    """

    backend_name = str(settings.get("backend", "in_memory") or "in_memory").lower()

    backend: Any
    if backend_name == "redis":
        redis_url = settings.get("redis_url")
        stream_prefix = settings.get("stream_prefix", "atlas_bus")
        replay_start = _normalize_replay_start(
            settings.get("replay_start") or settings.get("initial_offset") or settings.get("initial_stream_id")
        )
        batch_size = _coerce_positive_int(settings.get("batch_size"), default=1)
        blocking_timeout_ms = _coerce_positive_int(settings.get("blocking_timeout_ms"), default=1000)
        auto_claim_idle_ms = _coerce_positive_int(
            settings.get("auto_claim_idle_ms"), default=60_000, allow_zero=True
        )
        auto_claim_count = _coerce_positive_int(settings.get("auto_claim_count"), default=10)
        delete_acknowledged = True if settings.get("delete_acknowledged") is None else bool(
            settings.get("delete_acknowledged")
        )
        trim_max_length = settings.get("trim_max_length")
        if trim_max_length is not None:
            trim_max_length = _coerce_positive_int(trim_max_length, default=None)
        try:
            backend = RedisStreamGroupBackend(
                str(redis_url),
                stream_prefix=str(stream_prefix),
                replay_start=str(replay_start),
                batch_size=batch_size,
                blocking_timeout=blocking_timeout_ms,
                auto_claim_idle_ms=auto_claim_idle_ms,
                auto_claim_count=auto_claim_count,
                delete_acknowledged=delete_acknowledged,
                trim_max_length=trim_max_length,
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


def _normalize_replay_start(value: Any | None) -> str:
    candidate = (str(value).strip() if value is not None else "") or "$"
    if candidate == "$":
        return "$"
    if _STREAM_ID_PATTERN.match(candidate):
        return candidate
    return "$"


def _normalize_kafka_block(value: Any | None, *, env_config: Mapping[str, Any]) -> dict[str, Any]:
    block = dict(value) if isinstance(value, Mapping) else {}
    enabled = _coerce_bool(block.get("enabled"), default=False)
    block["enabled"] = enabled

    bootstrap_servers = block.get("bootstrap_servers") or env_config.get("KAFKA_BOOTSTRAP_SERVERS")
    block["bootstrap_servers"] = str(bootstrap_servers).strip() if bootstrap_servers else None

    topic_prefix = block.get("topic_prefix")
    block["topic_prefix"] = str(topic_prefix).strip() if topic_prefix else "atlas.bus"

    client_id = block.get("client_id")
    block["client_id"] = str(client_id).strip() if client_id else "atlas-message-bridge"

    driver = block.get("driver") or block.get("preferred_driver")
    block["driver"] = str(driver).strip().lower() if driver else None

    extra_config = block.get("producer_config")
    block["producer_config"] = dict(extra_config) if isinstance(extra_config, Mapping) else {}

    block["delivery_timeout"] = _coerce_positive_float(block.get("delivery_timeout"), default=10.0)
    block["bridge"] = _normalize_kafka_bridge_block(block.get("bridge"))
    return block


def _normalize_kafka_bridge_block(value: Any | None) -> dict[str, Any]:
    bridge = dict(value) if isinstance(value, Mapping) else {}
    bridge["enabled"] = _coerce_bool(bridge.get("enabled"), default=False)
    source_prefix = bridge.get("source_prefix")
    bridge["source_prefix"] = str(source_prefix).strip() if source_prefix else "redis_kafka"

    topics_raw = bridge.get("topics")
    topics_value: Sequence[Any] = (
        topics_raw if isinstance(topics_raw, Sequence) and not isinstance(topics_raw, (str, bytes, bytearray)) else []
    )
    topics: list[str] = []
    for topic in topics_value or []:
        cleaned = str(topic or "").strip()
        if cleaned:
            topics.append(cleaned)
    bridge["topics"] = topics

    topic_map = bridge.get("topic_map")
    if isinstance(topic_map, Mapping):
        bridge["topic_map"] = {str(key).strip(): str(value).strip() for key, value in topic_map.items()}
    else:
        bridge["topic_map"] = {}

    dlq_topic = bridge.get("dlq_topic")
    bridge["dlq_topic"] = str(dlq_topic).strip() if dlq_topic else "atlas.bridge.dlq"

    bridge["max_attempts"] = _coerce_positive_int(bridge.get("max_attempts"), default=3)
    bridge["backoff_seconds"] = _coerce_positive_float(bridge.get("backoff_seconds"), default=1.0)
    return bridge


def _coerce_positive_int(value: Any | None, *, default: int | None, allow_zero: bool = False) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed > 0 or (allow_zero and parsed == 0):
        return parsed
    return default


def _coerce_positive_float(value: Any | None, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed > 0:
        return parsed
    return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return bool(value)
