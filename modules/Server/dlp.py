"""Data loss prevention helpers for request payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


_DEFAULT_PATTERNS: tuple[str, ...] = (
    r"(?i)ssn[^0-9]*\d{3}-\d{2}-\d{4}",
    r"(?i)passport[^A-Za-z0-9]*[A-Za-z0-9]{6,9}",
    r"(?i)secret_key\s*[:=]\s*\S+",
    r"(?i)api_key\s*[:=]\s*\S+",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
)


@dataclass(frozen=True)
class DlpPolicy:
    """Normalized DLP policy."""

    enabled: bool
    replacement: str
    patterns: tuple[re.Pattern[str], ...]

    @staticmethod
    def from_mapping(settings: Mapping[str, Any]) -> "DlpPolicy":
        replacement = str(settings.get("replacement") or "<redacted>")
        pattern_values: Iterable[str] = settings.get("patterns") or _DEFAULT_PATTERNS
        compiled: list[re.Pattern[str]] = []
        for value in pattern_values:
            try:
                compiled.append(re.compile(value))
            except re.error:
                continue
        if not compiled:
            compiled = [re.compile(pattern) for pattern in _DEFAULT_PATTERNS]
        return DlpPolicy(enabled=bool(settings.get("enabled", True)), replacement=replacement, patterns=tuple(compiled))


class DataLossPreventionEnforcer:
    """Redact sensitive values before they reach persistence layers."""

    def __init__(self, config_manager: Any | None) -> None:
        self._config_manager = config_manager

    def _load_policy(self, tenant_id: Optional[str]) -> DlpPolicy:
        getter = getattr(self._config_manager, "get_dlp_policy", None)
        settings: Mapping[str, Any] = {}
        if callable(getter):
            try:
                settings = getter(tenant_id)
            except Exception:
                settings = {}
        return DlpPolicy.from_mapping(settings)

    def apply_to_payload(self, payload: Mapping[str, Any], tenant_id: Optional[str]) -> dict[str, Any]:
        policy = self._load_policy(tenant_id)
        if not policy.enabled:
            return dict(payload)

        return {
            key: self._sanitize_value(value, policy)
            for key, value in payload.items()
        }

    def _sanitize_value(self, value: Any, policy: DlpPolicy) -> Any:
        if isinstance(value, str):
            return self._redact_text(value, policy)
        if isinstance(value, Mapping):
            return {k: self._sanitize_value(v, policy) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            sequence = [self._sanitize_value(item, policy) for item in value]
            return type(value)(sequence)
        return value

    def _redact_text(self, text: str, policy: DlpPolicy) -> str:
        redacted = text
        for pattern in policy.patterns:
            redacted = pattern.sub(policy.replacement, redacted)
        return redacted
