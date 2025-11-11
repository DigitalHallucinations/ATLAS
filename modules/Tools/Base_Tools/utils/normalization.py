"""Shared normalization helpers for Base Tools modules."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

__all__ = ["dedupe_strings", "coerce_metadata", "normalize_metrics"]


def dedupe_strings(sequence: Sequence[Any] | None, *, lower: bool = False) -> tuple[str, ...]:
    """Return a tuple of unique, cleaned string values.

    Args:
        sequence: Potential sequence of string-like values to normalise.
        lower: When ``True`` the resulting strings are coerced to lowercase.

    Returns:
        A tuple containing the unique cleaned strings in order of first
        appearance. Non-string or empty entries are ignored.
    """

    if not sequence:
        return tuple()

    normalised: list[str] = []
    for value in sequence:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        if lower:
            candidate = candidate.lower()
        normalised.append(candidate)

    return tuple(dict.fromkeys(normalised))


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
