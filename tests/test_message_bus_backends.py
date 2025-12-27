import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from modules.orchestration.message_bus import (
    BusMessage,
    MessagePriority,
    RedisStreamBackend,
)


class _FakeRedisStream:
    """Minimal Redis Streams stand-in to exercise backend sequencing."""

    def __init__(self) -> None:
        self._streams: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
        self._latest: Dict[str, str] = {}
        self._events: Dict[str, asyncio.Event] = defaultdict(asyncio.Event)

    async def xadd(self, stream_name: str, payload: dict) -> str:
        entry_id = f"{int(time.time() * 1000)}-{len(self._streams[stream_name]) + 1}"
        self._streams[stream_name].append((entry_id, dict(payload)))
        self._latest[stream_name] = entry_id
        self._events[stream_name].set()
        return entry_id

    async def xread(self, streams: dict, count: int = 1, block: int = 0):
        stream_name, last_id = next(iter(streams.items()))
        baseline = self._latest.get(stream_name)
        start_after = baseline if last_id == "$" else last_id
        deadline = time.time() + (block / 1000.0 if block else 0.0)

        while True:
            entries = [
                (entry_id, payload)
                for entry_id, payload in self._streams.get(stream_name, [])
                if self._greater(entry_id, start_after)
            ]
            if entries:
                return [(stream_name, entries[:count])]

            if block <= 0:
                return []

            timeout = deadline - time.time()
            if timeout <= 0:
                return []

            try:
                await asyncio.wait_for(self._events[stream_name].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return []
            finally:
                self._events[stream_name].clear()

    async def xdel(self, stream_name: str, entry_id: str):
        items = self._streams.get(stream_name, [])
        self._streams[stream_name] = [(eid, payload) for eid, payload in items if eid != entry_id]

    async def close(self):
        return None

    @staticmethod
    def _greater(candidate: str, floor: str | None) -> bool:
        if floor is None:
            return True
        try:
            cand_ts, cand_seq = candidate.split("-", maxsplit=1)
            floor_ts, floor_seq = floor.split("-", maxsplit=1)
            return (int(cand_ts), int(cand_seq)) > (int(floor_ts), int(floor_seq))
        except ValueError:
            return False


async def _roundtrip_requeue() -> tuple[BusMessage, BusMessage, BusMessage]:
    fake_redis = _FakeRedisStream()
    backend = RedisStreamBackend(
        "redis://example",
        stream_prefix="test_bus",
        blocking_timeout=50,
        redis_client=fake_redis,
    )

    original = BusMessage(
        topic="alerts",
        payload={"status": "open"},
        priority=MessagePriority.NORMAL,
        tracing={"source": "test"},
    )

    initial_task = asyncio.create_task(backend.get("alerts"))
    await asyncio.sleep(0)
    await backend.publish(original)
    initial = await asyncio.wait_for(initial_task, timeout=0.5)
    await backend.requeue(initial)
    backend.acknowledge(initial)

    replayed = await asyncio.wait_for(backend.get("alerts"), timeout=0.5)
    await backend.close()
    return original, initial, replayed


def test_redis_backend_requeues_without_skipping_entries():
    original, initial, replayed = asyncio.run(_roundtrip_requeue())

    assert replayed.payload == original.payload
    assert replayed.backend_id != initial.backend_id
