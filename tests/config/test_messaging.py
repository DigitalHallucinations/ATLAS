from typing import Any

from ATLAS.config import messaging


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        if args:
            try:
                message = message % args
            except Exception:
                message = str(message)
        self.messages.append(str(message))


def test_setup_message_bus_redis_backend(monkeypatch):
    created: dict[str, Any] = {}

    class StubRedisBackend:
        def __init__(
            self,
            url: str,
            stream_prefix: str,
            replay_start: str | None = None,
            **kwargs: Any,
        ) -> None:
            created["url"] = url
            created["prefix"] = stream_prefix
            created["replay_start"] = replay_start
            created["batch_size"] = kwargs.get("batch_size")
            created["blocking_timeout"] = kwargs.get("blocking_timeout")
            created["auto_claim_idle_ms"] = kwargs.get("auto_claim_idle_ms")

    sentinel_bus = object()

    def fake_configure(backend: Any) -> Any:
        created["configured"] = backend
        return sentinel_bus

    monkeypatch.setattr(messaging, "RedisStreamGroupBackend", StubRedisBackend)
    monkeypatch.setattr(messaging, "configure_message_bus", fake_configure)

    backend, bus = messaging.setup_message_bus(
        {
            "backend": "redis",
            "redis_url": "redis://localhost:6379/0",
            "stream_prefix": "atlas",
            "initial_stream_id": "0-0",
        },
        logger=DummyLogger(),
    )

    assert isinstance(backend, StubRedisBackend)
    assert created["configured"] is backend
    assert created["url"] == "redis://localhost:6379/0"
    assert created["prefix"] == "atlas"
    assert created["replay_start"] == "0-0"
    assert created["batch_size"] == 1
    assert created["blocking_timeout"] == 1000
    assert created["auto_claim_idle_ms"] == 60000
    assert bus is sentinel_bus


def test_setup_message_bus_falls_back_to_memory(monkeypatch):
    warnings = DummyLogger()

    class RaisingRedisBackend:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

    class StubMemoryBackend:
        pass

    sentinel_bus = object()

    monkeypatch.setattr(messaging, "RedisStreamGroupBackend", RaisingRedisBackend)
    monkeypatch.setattr(messaging, "InMemoryQueueBackend", StubMemoryBackend)
    monkeypatch.setattr(messaging, "configure_message_bus", lambda backend: sentinel_bus)

    backend, bus = messaging.setup_message_bus({"backend": "redis"}, logger=warnings)

    assert isinstance(backend, StubMemoryBackend)
    assert bus is sentinel_bus
    assert warnings.messages and "Falling back" in warnings.messages[0]


def test_messaging_apply_includes_kafka_defaults():
    config: dict[str, Any] = {}
    yaml_config: dict[str, Any] = {}

    section = messaging.MessagingConfigSection(
        config=config,
        yaml_config=yaml_config,
        env_config={},
        logger=DummyLogger(),
        write_yaml_callback=lambda: None,
    )
    section.apply()
    settings = section.get_settings()
    kafka_block = settings.get("kafka")

    assert kafka_block is not None
    assert kafka_block["enabled"] is False
    assert kafka_block["topic_prefix"] == "atlas.bus"
    assert kafka_block["bridge"]["source_prefix"] == "redis_kafka"
    assert kafka_block["bridge"]["dlq_topic"] == "atlas.bridge.dlq"


def test_set_settings_normalizes_kafka_block():
    writes: list[bool] = []
    section = messaging.MessagingConfigSection(
        config={},
        yaml_config={},
        env_config={"KAFKA_BOOTSTRAP_SERVERS": "env.kafka:9092"},
        logger=DummyLogger(),
        write_yaml_callback=lambda: writes.append(True),
    )

    settings = section.set_settings(
        backend="in_memory",
        kafka={
            "enabled": True,
            "topic_prefix": "custom.bus",
            "bridge": {
                "topics": ["audit", "metrics"],
                "topic_map": {"audit": "external.audit"},
                "max_attempts": 9,
                "backoff_seconds": 0.25,
            },
        },
    )

    kafka_block = settings["kafka"]
    assert kafka_block["enabled"] is True
    assert kafka_block["bootstrap_servers"] == "env.kafka:9092"
    assert kafka_block["topic_prefix"] == "custom.bus"
    assert kafka_block["bridge"]["topics"] == ["audit", "metrics"]
    assert kafka_block["bridge"]["topic_map"]["audit"] == "external.audit"
    assert kafka_block["bridge"]["max_attempts"] == 9
    assert kafka_block["bridge"]["backoff_seconds"] == 0.25
    assert writes
