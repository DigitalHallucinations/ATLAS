"""Shared orchestration helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence


SHARED_PERSONA_EXCLUSION_TOKENS = {
    "-shared",
    "!shared",
    "no-shared",
    "without-shared",
    "shared=false",
    "shared:false",
}


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


def persona_matches_filter(
    persona: Optional[Any], tokens: Sequence[Any]
) -> bool:
    """Return ``True`` when ``persona`` matches the provided ``tokens``.

    The helper centralizes the shared logic for evaluating persona filters
    across the API and capability registry surfaces.  Tokens are normalized to
    lowercase strings, with a dedicated exclusion list for opt-out of shared
    personas (``SHARED_PERSONA_EXCLUSION_TOKENS``).
    """

    if not tokens:
        return True

    exclude_shared = False
    positive_tokens: list[str] = []
    for token in tokens:
        if token is None:
            continue
        text = str(token).strip().lower()
        if not text:
            continue
        if text in SHARED_PERSONA_EXCLUSION_TOKENS:
            exclude_shared = True
            continue
        positive_tokens.append(text)

    persona_token = normalize_persona_identifier(persona) or "shared"

    if persona_token == "shared":
        return not exclude_shared

    if not positive_tokens:
        return True

    return persona_token in positive_tokens


__all__ = [
    "SHARED_PERSONA_EXCLUSION_TOKENS",
    "normalize_persona_identifier",
    "persona_matches_filter",
]
