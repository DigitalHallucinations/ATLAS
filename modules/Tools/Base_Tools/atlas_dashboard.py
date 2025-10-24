"""Persona specific dashboard helper for ATLAS operators."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtlasDashboardUpdate:
    """Structured representation of an initiative update."""

    initiative: str
    health: str
    summary: str
    metrics: Mapping[str, float]
    stakeholders: tuple[str, ...]
    captured_at: str


def _normalize_metrics(metrics: Optional[Mapping[str, object]]) -> Mapping[str, float]:
    normalized: MutableMapping[str, float] = {}
    if not metrics:
        return {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return dict(normalized)


def _normalize_stakeholders(stakeholders: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not stakeholders:
        return tuple()
    normalized: list[str] = []
    for entry in stakeholders:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if candidate:
            normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


class AtlasDashboardClient:
    """Collect initiative level signals for ATLAS specific dashboards."""

    def __init__(self) -> None:
        self._updates: MutableMapping[str, AtlasDashboardUpdate] = {}

    async def run(
        self,
        *,
        initiative: str,
        health: str,
        summary: str,
        metrics: Optional[Mapping[str, object]] = None,
        stakeholders: Optional[Sequence[str]] = None,
    ) -> Mapping[str, object]:
        if not initiative.strip():
            raise ValueError("Initiative name is required.")
        if not health.strip():
            raise ValueError("Health rating must be provided.")
        if not summary.strip():
            raise ValueError("Summary must be provided.")

        await asyncio.sleep(0)

        update = AtlasDashboardUpdate(
            initiative=initiative.strip(),
            health=health.strip().lower(),
            summary=summary.strip(),
            metrics=_normalize_metrics(metrics),
            stakeholders=_normalize_stakeholders(stakeholders),
            captured_at=datetime.now(timezone.utc).isoformat(),
        )

        self._updates[update.initiative] = update
        logger.debug("Atlas dashboard updated for %s", update.initiative)

        return {
            "initiative": update.initiative,
            "health": update.health,
            "summary": update.summary,
            "metrics": update.metrics,
            "stakeholders": update.stakeholders,
            "captured_at": update.captured_at,
        }

    async def snapshot(self) -> Mapping[str, object]:
        await asyncio.sleep(0)
        return {key: asdict(value) for key, value in self._updates.items()}


__all__ = ["AtlasDashboardClient", "AtlasDashboardUpdate"]

