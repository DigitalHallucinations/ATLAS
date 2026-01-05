"""Shared normalization helpers for Base Tools modules."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from core.utils.collections import dedupe_strings

__all__ = [
    "dedupe_strings",
    "coerce_metadata",
    "normalize_metrics",
    "normalize_mapping_keys",
]


def coerce_metadata(mapping: Mapping[Any, Any] | None) -> Mapping[str, Any]:
    """Coerce arbitrary mapping types into a ``dict`` with string keys."""

    if mapping is None:
        return {}
    if isinstance(mapping, MutableMapping):
        return dict(mapping)
    return {str(key): value for key, value in mapping.items()}


def normalize_metrics(
    metrics: Mapping[Any, Any] | None,
) -> Mapping[str, float]:
    """Return numeric metrics from an optional mapping.

    Args:
        metrics: Mapping of potential metric values. Non-string keys or values
            that cannot be coerced to ``float`` are ignored. ``None`` inputs are
            treated as empty mappings so callers handling optional metrics do
            not need to special-case the absence of values.

    Returns:
        A dictionary containing only the metrics that could be converted to
        ``float`` values.
    """

    if not metrics:
        return {}

    normalized: MutableMapping[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

    return dict(normalized)


def normalize_mapping_keys(mapping: Mapping[Any, Any] | None) -> Mapping[str, Any]:
    """Return a ``dict`` with string keys for the provided mapping.

    Args:
        mapping: Mapping containing arbitrary key types. ``None`` inputs are
            treated as empty mappings so callers do not need to special-case the
            absence of values.

    Returns:
        A dictionary whose keys have been coerced to strings.
    """

    if not mapping:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        normalized[str(key)] = value
    return normalized
