"""Streaming telemetry aggregator used to monitor ingestion pipelines."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, MutableMapping, Sequence

from modules.logging.logger import setup_logger

__all__ = ["StreamMonitor", "StreamMonitorError"]


logger = setup_logger(__name__)


class StreamMonitorError(RuntimeError):
    """Raised when stream events cannot be processed."""


@dataclass(frozen=True)
class StreamSummary:
    stream_id: str
    total_events: int
    status_counts: Mapping[str, int]
    error_events: Sequence[Mapping[str, object]]
    latency_ms: float | None


def _normalise_events(events: Iterable[Mapping[str, object]]) -> Sequence[Mapping[str, object]]:
    normalised = []
    for item in events:
        if not isinstance(item, Mapping):
            raise StreamMonitorError("All stream events must be mappings")
        normalised.append(dict(item))
    return tuple(normalised)


class StreamMonitor:
    """Aggregate status, error, and latency information from event streams."""

    async def run(
        self,
        *,
        stream_id: str,
        events: Iterable[Mapping[str, object]],
    ) -> Mapping[str, object]:
        if not stream_id:
            raise StreamMonitorError("stream_id is required")

        payload = _normalise_events(events)
        await asyncio.sleep(0)

        status_counts: MutableMapping[str, int] = Counter()
        latencies = []
        error_events = []

        for event in payload:
            status = str(event.get("status", "unknown")).lower()
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in {"error", "failed", "failure"}:
                error_events.append({"status": status, "details": event.get("details")})
            latency = event.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

        summary = StreamSummary(
            stream_id=stream_id,
            total_events=len(payload),
            status_counts=dict(status_counts),
            error_events=tuple(error_events),
            latency_ms=mean(latencies) if latencies else None,
        )
        logger.info(
            "Stream %s processed %d events", stream_id, summary.total_events
        )
        return summary.__dict__
