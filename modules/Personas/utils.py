"""Shared persona normalization helpers."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Tuple


def normalize_persona_allowlist(raw_allowlist: Any) -> Optional[set[str]]:
    """Return a normalized set of persona names from ``raw_allowlist``."""

    if raw_allowlist is None:
        return None

    if isinstance(raw_allowlist, str):
        candidate = raw_allowlist.strip()
        return {candidate} if candidate else None

    if isinstance(raw_allowlist, Mapping):
        values = raw_allowlist.values()
    elif isinstance(raw_allowlist, (list, tuple, set)):
        values = raw_allowlist
    else:
        return None

    names = {str(item).strip() for item in values if str(item).strip()}
    return names or None


def join_with_and(items: Iterable[str]) -> str:
    sequence = [item for item in items if item]
    if not sequence:
        return ""
    if len(sequence) == 1:
        return sequence[0]
    return ", ".join(sequence[:-1]) + f", and {sequence[-1]}"


def normalize_requires_flags(raw_value: Any) -> Dict[str, Tuple[str, ...]]:
    """Coerce metadata flag requirements into a normalized mapping."""

    normalized: Dict[str, Tuple[str, ...]] = {}
    if not isinstance(raw_value, Mapping):
        return normalized

    for raw_operation, raw_flags in raw_value.items():
        operation = str(raw_operation or "").strip().lower()
        if not operation:
            continue

        if isinstance(raw_flags, (list, tuple, set)):
            candidates = list(raw_flags)
        else:
            candidates = [raw_flags]

        flags: List[str] = []
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                flags.append(text)

        if flags:
            deduped = list(dict.fromkeys(flags))
            normalized[operation] = tuple(deduped)

    return normalized


def coerce_persona_flag(value: Any) -> bool:
    """Interpret serialized persona toggles as booleans."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "enabled"}:
            return True
        if lowered in {"false", "0", "no", "off", "disabled"}:
            return False
    return bool(value)


def persona_flag_enabled(persona: Mapping[str, Any], flag_path: str) -> bool:
    """Resolve dotted persona paths against ``persona`` and coerce to ``bool``."""

    target: Any = persona
    for segment in str(flag_path).split("."):
        key = segment.strip()
        if not key:
            return False
        if isinstance(target, Mapping):
            target = target.get(key)
        else:
            target = getattr(target, key, None)
        if target is None:
            return False
    return coerce_persona_flag(target)


def collect_missing_flag_requirements(
    requires_flags: Mapping[str, Tuple[str, ...]],
    persona: Mapping[str, Any],
) -> Dict[str, Tuple[str, ...]]:
    """Return operations whose required persona flags are missing."""

    missing: Dict[str, Tuple[str, ...]] = {}
    for operation, flags in requires_flags.items():
        missing_flags = tuple(
            flag for flag in flags if not persona_flag_enabled(persona, flag)
        )
        if missing_flags:
            missing[operation] = missing_flags
    return missing


def format_denied_operations_summary(
    function_name: str,
    denied_operations: Mapping[str, Tuple[str, ...]],
) -> Optional[str]:
    """Summarize the operations disabled by missing persona flags."""

    if not denied_operations:
        return None

    operations = sorted({op for op in denied_operations.keys() if op})
    if not operations:
        return None

    flags = sorted({flag for flags in denied_operations.values() for flag in flags})
    if not flags:
        return None

    if set(operations) == {"create", "update", "delete"}:
        operations_phrase = "Write operations (create, update, delete)"
    else:
        operations_phrase = ", ".join(operations)

    flag_phrase = join_with_and([f"'{flag}'" for flag in flags])
    plural = "s" if len(flags) > 1 else ""
    return (
        f"{operations_phrase} for tool '{function_name}' require{plural} persona flag"
        f"{plural} {flag_phrase} to be enabled."
    )


__all__ = [
    "normalize_persona_allowlist",
    "join_with_and",
    "normalize_requires_flags",
    "coerce_persona_flag",
    "persona_flag_enabled",
    "collect_missing_flag_requirements",
    "format_denied_operations_summary",
]
