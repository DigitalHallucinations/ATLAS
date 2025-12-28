"""Messaging utilities for publishing to external transports."""

from __future__ import annotations

from typing import Any

from .idempotency import IdempotencyStore

__all__ = ["KafkaSink", "KafkaSinkUnavailable", "RedisToKafkaBridge", "build_bridge_from_settings", "IdempotencyStore"]


def __getattr__(name: str) -> Any:
    if name in {"KafkaSink", "KafkaSinkUnavailable"}:
        from .kafka_sink import KafkaSink, KafkaSinkUnavailable  # pylint: disable=import-outside-toplevel

        globals().update({"KafkaSink": KafkaSink, "KafkaSinkUnavailable": KafkaSinkUnavailable})
        return globals()[name]

    if name in {"RedisToKafkaBridge", "build_bridge_from_settings"}:
        from .bridge_redis_to_kafka import (  # pylint: disable=import-outside-toplevel
            RedisToKafkaBridge,
            build_bridge_from_settings,
        )

        globals().update(
            {
                "RedisToKafkaBridge": RedisToKafkaBridge,
                "build_bridge_from_settings": build_bridge_from_settings,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
