"""Minimal dashboard aggregation helper used in performance reports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardSnapshot:
    """Aggregated metrics stored for a dashboard."""

    dashboard_id: str
    metrics: Mapping[str, float]
    notes: str
    published_at: str


def _normalize_metrics(metrics: Mapping[str, object]) -> Mapping[str, float]:
    normalized: MutableMapping[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return dict(normalized)


class DashboardService:
    """Persist aggregate performance metrics for reporting."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, DashboardSnapshot] = {}

    async def run(
        self,
        *,
        dashboard_id: str,
        metrics: Mapping[str, object],
        notes: Optional[str] = None,
        publish: bool = True,
    ) -> Mapping[str, object]:
        if not isinstance(dashboard_id, str) or not dashboard_id.strip():
            raise ValueError("Dashboard identifier must be provided.")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("At least one metric must be provided.")

        await asyncio.sleep(0)

        normalized_id = dashboard_id.strip()
        normalized_metrics = _normalize_metrics(metrics)
        if not normalized_metrics:
            raise ValueError("Provided metrics did not contain numeric values.")

        snapshot = DashboardSnapshot(
            dashboard_id=normalized_id,
            metrics=normalized_metrics,
            notes=(notes or "").strip(),
            published_at=datetime.now(timezone.utc).isoformat() if publish else "",
        )

        self._store[normalized_id] = snapshot

        logger.debug("Dashboard %s updated with %d metrics", normalized_id, len(normalized_metrics))

        return {
            "dashboard_id": normalized_id,
            "metrics": normalized_metrics,
            "notes": snapshot.notes,
            "published_at": snapshot.published_at,
        }

    async def snapshot(self, dashboard_id: str) -> Mapping[str, object]:
        await asyncio.sleep(0)
        snapshot = self._store.get(dashboard_id)
        if snapshot is None:
            raise KeyError(f"Dashboard '{dashboard_id}' has no recorded snapshot.")
        return asdict(snapshot)


__all__ = ["DashboardService", "DashboardSnapshot"]

