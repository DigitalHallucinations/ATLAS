"""Synthetic labour market feed used for research capabilities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import hashlib

import logging

logger = logging.getLogger(__name__)


def _normalize_list(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not values:
        return tuple()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if candidate:
            normalized.append(candidate)
    return tuple(dict.fromkeys(normalized))


def _score_signal(region: str, skill: str, timeframe: str) -> float:
    components = (
        region.strip().lower(),
        skill.strip().lower(),
        timeframe.strip().lower(),
    )
    hasher = hashlib.sha256()
    for component in components:
        hasher.update(component.encode("utf-8"))
        hasher.update(b"\0")
    digest_value = int.from_bytes(hasher.digest(), "big")
    base = digest_value % 100
    return round(50 + base / 2, 2)


async def fetch_labor_market_signals(
    *,
    regions: Sequence[str],
    skills: Sequence[str],
    timeframe: str = "30d",
) -> Mapping[str, object]:
    if not regions or not skills:
        raise ValueError("At least one region and skill must be provided.")

    await asyncio.sleep(0)

    normalized_regions = _normalize_list(regions)
    normalized_skills = _normalize_list(skills)
    if not normalized_regions or not normalized_skills:
        raise ValueError("Unable to normalise regions or skills.")

    timeframe_value = timeframe.strip() or "30d"
    payload: dict[str, object] = {
        "timeframe": timeframe_value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regions": normalized_regions,
        "skills": normalized_skills,
        "signals": {},
    }

    signals: dict[str, dict[str, float]] = {}
    for region in normalized_regions:
        region_scores: dict[str, float] = {}
        for skill in normalized_skills:
            region_scores[skill] = _score_signal(region, skill, timeframe_value)
        signals[region] = region_scores

    payload["signals"] = signals

    logger.debug(
        "Generated labour market snapshot for %s regions and %s skills",
        len(normalized_regions),
        len(normalized_skills),
    )

    return payload


__all__ = ["fetch_labor_market_signals"]

