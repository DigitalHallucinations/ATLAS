"""Lightweight analytics dashboard helper for cohort and metric tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)

from .utils import coerce_metadata, dedupe_strings, normalize_metrics


@dataclass(frozen=True)
class AnalyticsSnapshot:
    """Structured representation of an analytics dashboard refresh."""

    dashboard_id: str
    summary: str
    metrics: Mapping[str, float]
    segments: tuple[Mapping[str, object], ...]
    tags: tuple[str, ...]
    metadata: Mapping[str, object]
    refreshed_at: str


def _normalize_segments(segments: Optional[Sequence[Mapping[str, object]]]) -> tuple[Mapping[str, object], ...]:
    if not segments:
        return tuple()
    normalized: list[Mapping[str, object]] = []
    for entry in segments:
        if not isinstance(entry, Mapping):
            continue
        normalized.append({str(key): value for key, value in entry.items()})
    return tuple(normalized)


class AnalyticsDashboardClient:
    """Capture analytics rollups for downstream reporting.

    The client stores data in-memory so repeated calls within a session can
    enrich or review the most recent snapshot for a dashboard.
    """

    def __init__(self) -> None:
        self._snapshots: MutableMapping[str, AnalyticsSnapshot] = {}

    async def run(
        self,
        *,
        dashboard_id: str,
        summary: str,
        metrics: Mapping[str, object],
        segments: Optional[Sequence[Mapping[str, object]]] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        if not isinstance(dashboard_id, str) or not dashboard_id.strip():
            raise ValueError("Dashboard identifier must be provided.")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("A summary of the analytics update is required.")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("At least one metric must be provided.")

        await asyncio.sleep(0)

        normalized_metrics = normalize_metrics(metrics)
        if not normalized_metrics:
            raise ValueError("Provided metrics did not contain numeric values.")

        snapshot = AnalyticsSnapshot(
            dashboard_id=dashboard_id.strip(),
            summary=summary.strip(),
            metrics=normalized_metrics,
            segments=_normalize_segments(segments),
            tags=dedupe_strings(tags, lower=True),
            metadata=coerce_metadata(metadata),
            refreshed_at=datetime.now(timezone.utc).isoformat(),
        )

        self._snapshots[snapshot.dashboard_id] = snapshot
        logger.debug("Analytics dashboard %s refreshed", snapshot.dashboard_id)

        return {
            "dashboard_id": snapshot.dashboard_id,
            "summary": snapshot.summary,
            "metrics": snapshot.metrics,
            "segments": snapshot.segments,
            "tags": snapshot.tags,
            "metadata": snapshot.metadata,
            "refreshed_at": snapshot.refreshed_at,
        }

    async def snapshot(self, dashboard_id: str) -> Mapping[str, object]:
        await asyncio.sleep(0)
        snapshot = self._snapshots.get(dashboard_id)
        if snapshot is None:
            raise KeyError(f"Dashboard '{dashboard_id}' has no recorded snapshot.")
        return asdict(snapshot)


__all__ = ["AnalyticsDashboardClient", "AnalyticsSnapshot"]
