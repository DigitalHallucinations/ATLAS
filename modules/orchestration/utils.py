"""Shared orchestration helpers."""

from __future__ import annotations

from typing import Any, Optional


def normalize_persona_identifier(persona: Optional[Any]) -> Optional[str]:
    """Return a canonical key for a persona identifier.

    Persona identifiers are stored as lowercase strings so that lookups across
    the orchestration layer behave consistently regardless of the caller's
    casing.  Non-string inputs are coerced to strings, whitespace is stripped,
    and empty results are treated as ``None``.
    """

    if persona is None:
        return None

    text = str(persona).strip()
    if not text:
        return None

    return text.lower()
