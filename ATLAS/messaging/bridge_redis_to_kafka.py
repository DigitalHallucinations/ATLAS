"""Bridge Redis stream topics to Kafka."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Iterable, Mapping, Sequence

from modules.orchestration.message_bus import BusMessage, RedisStreamGroupBackend

from .kafka_sink import KafkaSink

LOGGER = logging.getLogger(__name__)


class RedisToKafkaBridge:
    """Consume Redis-backed bus topics and publish them to Kafka."""

    def __init__(
        self,
        *,
        redis_backend: RedisStreamGroupBackend,
        kafka_sink: KafkaSink,
        source_topics: Sequence[str],
        source_prefix: str = "redis_kafka",
        topic_map: Mapping[str, str] | None = None,
        dlq_topic: str = "atlas.bridge.dlq",
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._redis_backend = redis_backend
        self._kafka_sink = kafka_sink
        self._source_prefix = source_prefix.strip() or "redis_kafka"
        self._source_topics = self._normalize_topics(source_topics)
        self._topic_map = {str(key).strip(): str(value).strip() for key, value in dict(topic_map or {}).items()}
        self._dlq_topic = dlq_topic
        self._max_attempts = max(int(max_attempts), 1)
        self._backoff_seconds = max(float(backoff_seconds), 0.0)
        self._logger = logger or LOGGER
        self._running = False
        self._tasks: set[asyncio.Task[Any]] = set()

    def start(self) -> None:
        """Begin consuming configured topics."""

        if self._running:
            return
        self._running = True
        for topic in self._source_topics:
            task = asyncio.create_task(self._worker(topic))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        """Stop consuming and flush producers."""

        self._running = False
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self._kafka_sink.close()

    async def process_one(self, topic: str) -> None:
        """Consume a single message from *topic* (intended for tests and tooling)."""

        await self._handle_message(topic)

    async def _worker(self, topic: str) -> None:
        while self._running:
            try:
                await self._handle_message(topic)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive loop logging
                self._logger.exception("Bridge worker for topic '%s' failed: %s", topic, exc)
                await asyncio.sleep(self._backoff_seconds or 0.1)

    async def _handle_message(self, topic: str) -> None:
        message = await self._redis_backend.get(topic)
        await self._publish_with_retries(topic, message)

    async def _publish_with_retries(self, source_topic: str, message: BusMessage) -> None:
        attempts = max(int(message.delivery_attempts or 0), 0)

        while attempts < self._max_attempts:
            attempts += 1
            message.delivery_attempts = attempts
            try:
                target_topic = self._resolve_target_topic(source_topic)
                await self._kafka_sink.publish_message(message, topic_override=target_topic)
                self._redis_backend.acknowledge(message)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.exception(
                    "Kafka publish attempt %d/%d failed for topic '%s': %s",
                    attempts,
                    self._max_attempts,
                    source_topic,
                    exc,
                )
                if attempts >= self._max_attempts:
                    await self._emit_dlq(source_topic, message, error=exc)
                    self._redis_backend.acknowledge(message)
                    return
                await asyncio.sleep(self._backoff_seconds * attempts)

    async def _emit_dlq(self, source_topic: str, message: BusMessage, *, error: Exception) -> None:
        payload = {
            "source_topic": source_topic,
            "payload": message.payload,
            "metadata": message.metadata,
            "correlation_id": message.correlation_id,
            "tracing": message.tracing or {},
            "delivery_attempts": message.delivery_attempts,
            "backend_id": message.backend_id,
            "error": str(error),
            "failed_at": time.time(),
        }
        try:
            await self._kafka_sink.publish_event(
                self._dlq_topic,
                payload,
                correlation_id=message.correlation_id,
                tracing=message.tracing or {},
                metadata=message.metadata,
            )
        except Exception:  # pragma: no cover - logging only path
            self._logger.exception(
                "Failed to publish DLQ entry for topic '%s' after %d attempts.", source_topic, message.delivery_attempts
            )

    def _resolve_target_topic(self, source_topic: str) -> str:
        mapped = self._topic_map.get(source_topic)
        if mapped:
            return self._kafka_sink.resolve_topic(mapped)

        suffix = self._strip_prefix(source_topic)
        mapped_suffix = self._topic_map.get(suffix, suffix)
        return self._kafka_sink.resolve_topic(mapped_suffix)

    def _strip_prefix(self, topic: str) -> str:
        prefix = f"{self._source_prefix}."
        if topic.startswith(prefix):
            return topic[len(prefix) :]
        if topic == self._source_prefix:
            return ""
        return topic

    def _normalize_topics(self, topics: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        prefix = f"{self._source_prefix}."
        for topic in topics:
            cleaned = str(topic or "").strip()
            if not cleaned:
                continue
            if not cleaned.startswith(prefix) and cleaned != self._source_prefix:
                cleaned = f"{prefix}{cleaned}"
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized


def build_bridge_from_settings(
    settings: Mapping[str, Any],
    redis_backend: RedisStreamGroupBackend,
    *,
    logger: logging.Logger | None = None,
) -> RedisToKafkaBridge | None:
    """Construct a Redis-to-Kafka bridge using messaging settings."""

    logger = logger or LOGGER
    kafka_block = settings.get("kafka") if isinstance(settings, Mapping) else None
    if not isinstance(kafka_block, Mapping) or not kafka_block.get("enabled"):
        logger.info("Kafka sink is disabled; Redis-to-Kafka bridge will not start.")
        return None

    bridge_block = kafka_block.get("bridge") if isinstance(kafka_block.get("bridge"), Mapping) else None
    if not bridge_block or not bridge_block.get("enabled"):
        logger.info("Kafka bridge is disabled; skipping Redis forwarding.")
        return None

    sink = KafkaSink.build(kafka_block, logger=logger)
    if sink is None:
        return None

    topics = _normalize_topics_with_prefix(
        bridge_block.get("topics") or [],
        source_prefix=bridge_block.get("source_prefix") or "redis_kafka",
    )
    if not topics:
        logger.warning(
            "Kafka bridge is enabled but no topics were configured under the 'bridge.topics' key; skipping startup."
        )
        return None

    return RedisToKafkaBridge(
        redis_backend=redis_backend,
        kafka_sink=sink,
        source_topics=topics,
        source_prefix=bridge_block.get("source_prefix") or "redis_kafka",
        topic_map=bridge_block.get("topic_map") or {},
        dlq_topic=bridge_block.get("dlq_topic") or "atlas.bridge.dlq",
        max_attempts=bridge_block.get("max_attempts") or 3,
        backoff_seconds=bridge_block.get("backoff_seconds") or 1.0,
        logger=logger,
    )


def _normalize_topics_with_prefix(topics: Iterable[str], *, source_prefix: str) -> list[str]:
    prefix = source_prefix.strip() or "redis_kafka"
    normalized: list[str] = []
    for topic in topics:
        cleaned = str(topic or "").strip()
        if not cleaned:
            continue
        if not cleaned.startswith(f"{prefix}.") and cleaned != prefix:
            cleaned = f"{prefix}.{cleaned}"
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized
