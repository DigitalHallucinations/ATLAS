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
            self, url: str, stream_prefix: str, initial_offset: str | None = None, **_kwargs: Any
        ) -> None:
            created["url"] = url
            created["prefix"] = stream_prefix
            created["initial_id"] = initial_offset

    sentinel_bus = object()

    def fake_configure(backend: Any) -> Any:
        created["configured"] = backend
        return sentinel_bus

    monkeypatch.setattr(messaging, "RedisStreamBackend", StubRedisBackend)
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
    assert created["initial_id"] == "0-0"
    assert bus is sentinel_bus


def test_setup_message_bus_falls_back_to_memory(monkeypatch):
    warnings = DummyLogger()

    class RaisingRedisBackend:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("boom")

    class StubMemoryBackend:
        pass

    sentinel_bus = object()

    monkeypatch.setattr(messaging, "RedisStreamBackend", RaisingRedisBackend)
    monkeypatch.setattr(messaging, "InMemoryQueueBackend", StubMemoryBackend)
    monkeypatch.setattr(messaging, "configure_message_bus", lambda backend: sentinel_bus)

    backend, bus = messaging.setup_message_bus({"backend": "redis"}, logger=warnings)

    assert isinstance(backend, StubMemoryBackend)
    assert bus is sentinel_bus
    assert warnings.messages and "Falling back" in warnings.messages[0]
