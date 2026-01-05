"""Kafka publishing helpers for message bus envelopes."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from .messages import AgentMessage

LOGGER = logging.getLogger(__name__)


class KafkaSinkUnavailable(RuntimeError):
    """Raised when no supported Kafka client libraries are available."""


@dataclass
class KafkaSink:
    """Publish message bus envelopes to Kafka topics."""

    bootstrap_servers: str
    topic_prefix: str | None = None
    client_id: str = "atlas-message-bridge"
    preferred_driver: str | None = None
    extra_config: Mapping[str, Any] = field(default_factory=dict)
    delivery_timeout: float = 10.0
    enable_idempotence: bool = True
    acks: str = "all"
    max_in_flight: int = 5
    producer_factory: Any | None = None
    logger: logging.Logger = LOGGER

    def __post_init__(self) -> None:
        driver, producer = self._create_producer()
        self._driver = driver
        self._producer = producer

    def _create_producer(self) -> tuple[str, Any]:
        preferred = (self.preferred_driver or "auto").strip().lower()
        attempts: list[tuple[str, str]] = []

        if preferred in {"auto", "confluent"}:
            if self._has_module("confluent_kafka"):
                confluent_kafka = importlib.import_module("confluent_kafka")
                config = {
                    "bootstrap.servers": self.bootstrap_servers,
                    "client.id": self.client_id,
                    "enable.idempotence": bool(self.enable_idempotence),
                    "acks": self.acks or "all",
                    "max.in.flight.requests.per.connection": max(int(self.max_in_flight), 1),
                }
                config.update(dict(self.extra_config or {}))
                return "confluent", confluent_kafka.Producer(config)
            attempts.append(("confluent", "Install 'confluent-kafka' to enable this driver."))

        if preferred in {"auto", "kafka_python"}:
            if self._has_module("kafka"):
                kafka_module = importlib.import_module("kafka")
                config = {
                    "bootstrap_servers": self.bootstrap_servers,
                    "client_id": self.client_id,
                    "acks": self.acks or "all",
                    "enable_idempotence": bool(self.enable_idempotence),
                    "retries": 5,
                }
                config.update(dict(self.extra_config or {}))
                return "kafka_python", kafka_module.KafkaProducer(**config)
            attempts.append(("kafka-python", "Install 'kafka-python' to enable this driver."))

        if self.producer_factory is not None:
            driver, producer = self.producer_factory(self.bootstrap_servers, self.client_id, self.extra_config)
            return driver, producer

        self._log_unavailable_drivers(attempts)
        raise KafkaSinkUnavailable(
            "Kafka publishing is unavailable. Install either 'confluent-kafka' or 'kafka-python' to enable the sink."
        )

    @staticmethod
    def _has_module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def _log_unavailable_drivers(self, attempts: list[tuple[str, str]]) -> None:
        if not attempts:
            return
        messages = "; ".join(f"{driver}: {hint}" for driver, hint in attempts)
        self.logger.warning("Kafka sink dependencies missing or unavailable (%s).", messages)

    def resolve_topic(self, topic: str) -> str:
        cleaned = str(topic or "").strip()
        if not cleaned:
            return self.topic_prefix or ""
        if self.topic_prefix:
            if cleaned.startswith(self.topic_prefix):
                return cleaned
            trimmed = cleaned.lstrip(".")
            return f"{self.topic_prefix}.{trimmed}" if trimmed else self.topic_prefix
        return cleaned

    async def publish_message(self, message: AgentMessage, *, topic_override: str | None = None) -> None:
        target_topic = self.resolve_topic(topic_override or message.channel)
        payload = self._serialize_agent_message(message)
        await self._send(target_topic, payload)

    async def publish_event(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
        tracing: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        target_topic = self.resolve_topic(topic)
        record = {
            "payload": payload,
            "correlation_id": correlation_id,
            "tracing": tracing or {},
            "metadata": metadata or {},
        }
        await self._send(target_topic, self._encode(record))

    async def flush(self) -> None:
        if self._driver == "confluent":
            await asyncio.get_running_loop().run_in_executor(None, self._producer.flush, self.delivery_timeout)
        elif self._driver == "kafka_python":
            await asyncio.get_running_loop().run_in_executor(None, self._producer.flush, self.delivery_timeout)

    async def close(self) -> None:
        await self.flush()

    async def _send(self, topic: str, payload: bytes) -> None:
        if self._driver == "confluent":
            await self._send_confluent(topic, payload)
            return
        if self._driver == "kafka_python":
            await self._send_kafka_python(topic, payload)
            return
        raise KafkaSinkUnavailable("Kafka producer is not initialized.")

    async def _send_confluent(self, topic: str, payload: bytes) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _delivery(err: Any, msg: Any) -> None:
            if err:
                loop.call_soon_threadsafe(future.set_exception, RuntimeError(str(err)))
            else:
                loop.call_soon_threadsafe(future.set_result, msg)

        self._producer.produce(topic, payload, callback=_delivery)
        self._producer.poll(0)
        await future

    async def _send_kafka_python(self, topic: str, payload: bytes) -> None:
        loop = asyncio.get_running_loop()
        future = self._producer.send(topic, payload)
        await loop.run_in_executor(None, future.get, self.delivery_timeout)

    @staticmethod
    def _serialize_agent_message(message: AgentMessage) -> bytes:
        record = {
            "channel": message.channel,
            "payload": message.payload,
            "priority": message.priority,
            "trace_id": message.trace_id,
            "headers": message.headers or {},
            "agent_id": message.agent_id,
            "conversation_id": message.conversation_id,
            "request_id": message.request_id,
            "user_id": message.user_id,
            "ts": message.ts,
        }
        return KafkaSink._encode(record)

    @staticmethod
    def _encode(record: Mapping[str, Any]) -> bytes:
        return json.dumps(record, default=str).encode("utf-8")

    @classmethod
    def build(cls, kafka_settings: Mapping[str, Any], *, logger: logging.Logger | None = None) -> KafkaSink | None:
        logger = logger or LOGGER
        enabled = bool(kafka_settings.get("enabled"))
        if not enabled:
            return None

        bootstrap_servers = kafka_settings.get("bootstrap_servers")
        if not bootstrap_servers:
            logger.warning("Kafka sink is enabled but no bootstrap servers were provided; skipping sink initialization.")
            return None

        topic_prefix = kafka_settings.get("topic_prefix")
        preferred_driver = kafka_settings.get("driver") or kafka_settings.get("preferred_driver")
        extra_config = kafka_settings.get("producer_config") if isinstance(kafka_settings.get("producer_config"), Mapping) else {}

        try:
            return cls(
                bootstrap_servers=str(bootstrap_servers),
                topic_prefix=str(topic_prefix) if topic_prefix else None,
                client_id=str(kafka_settings.get("client_id") or "atlas-message-bridge"),
                preferred_driver=str(preferred_driver) if preferred_driver else None,
                extra_config=extra_config,
                delivery_timeout=float(kafka_settings.get("delivery_timeout", 10.0) or 10.0),
                enable_idempotence=bool(kafka_settings.get("enable_idempotence", True)),
                acks=str(kafka_settings.get("acks") or "all"),
                max_in_flight=int(kafka_settings.get("max_in_flight", 5) or 5),
                logger=logger,
            )
        except KafkaSinkUnavailable:
            logger.warning(
                "Kafka sink is disabled because no Kafka client libraries are installed. "
                "Install 'confluent-kafka' or 'kafka-python' to enable publishing."
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialise Kafka sink: %s", exc)
        return None
