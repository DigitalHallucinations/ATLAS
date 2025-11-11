"""Shared normalization helpers for Base Tools modules."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

__all__ = ["dedupe_strings", "coerce_metadata"]


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
