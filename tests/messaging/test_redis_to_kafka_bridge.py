import asyncio

import pytest

from ATLAS.messaging import bridge_redis_to_kafka as bridge_module
from ATLAS.messaging.bridge_redis_to_kafka import RedisToKafkaBridge
from modules.orchestration.message_bus import BusMessage


class StubRedisBackend:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[BusMessage] = asyncio.Queue()
        self.acknowledged: list[BusMessage] = []

    async def publish(self, message: BusMessage) -> None:
        await self.queue.put(message)

    async def get(self, _topic: str) -> BusMessage:
        return await self.queue.get()

    async def requeue(self, message: BusMessage) -> None:  # pragma: no cover - bridge does not requeue
        await self.publish(message)

    def acknowledge(self, message: BusMessage) -> None:
        self.acknowledged.append(message)

    async def close(self) -> None:  # pragma: no cover - unused in tests
        return None


class RecordingSink:
    def __init__(self, *, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.published: list[tuple[str, str]] = []
        self.dlq: list[tuple[str, dict]] = []

    def resolve_topic(self, topic: str) -> str:
        return f"resolved.{topic}" if topic else "resolved"

    async def publish_message(self, message: BusMessage, *, topic_override: str | None = None) -> None:
        target = topic_override or message.topic
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("simulated publish failure")
        self.published.append((target, message.correlation_id))

    async def publish_event(self, topic: str, payload, **_kwargs) -> None:
        self.dlq.append((topic, dict(payload)))

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_bridge_retries_before_acknowledging():
    backend = StubRedisBackend()
    sink = RecordingSink(failures_before_success=1)
    bridge = RedisToKafkaBridge(
        redis_backend=backend,
        kafka_sink=sink,
        source_topics=["redis_kafka.events"],
        max_attempts=3,
        backoff_seconds=0,
    )

    message = BusMessage(topic="redis_kafka.events", payload={"foo": "bar"})
    await backend.publish(message)

    await bridge.process_one("redis_kafka.events")

    assert backend.acknowledged == [message]
    assert sink.published and sink.published[0][0] == "resolved.events"
    assert message.delivery_attempts == 2


@pytest.mark.asyncio
async def test_bridge_emits_dlq_after_max_attempts():
    backend = StubRedisBackend()
    sink = RecordingSink(failures_before_success=5)
    bridge = RedisToKafkaBridge(
        redis_backend=backend,
        kafka_sink=sink,
        source_topics=["redis_kafka.metrics"],
        dlq_topic="bridge.dlq",
        max_attempts=2,
        backoff_seconds=0,
    )

    message = BusMessage(topic="redis_kafka.metrics", payload={"value": 10})
    await backend.publish(message)

    await bridge.process_one("redis_kafka.metrics")

    assert backend.acknowledged == [message]
    assert sink.published == []
    assert sink.dlq and sink.dlq[0][0] == "bridge.dlq"
    dlq_payload = sink.dlq[0][1]
    assert dlq_payload["source_topic"] == "redis_kafka.metrics"
    assert dlq_payload["delivery_attempts"] == 2


def test_build_bridge_from_settings_normalizes_topics(monkeypatch):
    backend = StubRedisBackend()

    monkeypatch.setattr(
        "ATLAS.messaging.bridge_redis_to_kafka.KafkaSink.build",
        staticmethod(lambda _settings, logger=None: RecordingSink()),
    )

    bridge = bridge_module.build_bridge_from_settings(
        {
            "kafka": {
                "enabled": True,
                "bootstrap_servers": "kafka:9092",
                "bridge": {"enabled": True, "source_prefix": "redis_kafka", "topics": ["alpha", "redis_kafka.beta"]},
            }
        },
        backend,
    )

    assert isinstance(bridge, RedisToKafkaBridge)
    assert sorted(bridge._source_topics) == ["redis_kafka.alpha", "redis_kafka.beta"]
