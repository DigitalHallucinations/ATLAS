"""Heuristics for ranking potential security threats inside event streams."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["ThreatScanner", "ThreatFinding"]

_SEVERITY_WEIGHTS: Mapping[str, float] = {
    "critical": 1.0,
    "alert": 0.95,
    "emergency": 1.0,
    "error": 0.75,
    "warning": 0.55,
    "notice": 0.4,
    "info": 0.2,
    "debug": 0.1,
}


@dataclass(frozen=True)
class ThreatFinding:
    """A ranked threat finding derived from the supplied events."""

    score: float
    severity: str
    timestamp: str | None
    reason: str
    indicator_hits: Sequence[str] = field(default_factory=tuple)
    event: Mapping[str, Any] | None = None


class ThreatScanner:
    """Score events against signatures and severity heuristics."""

    def run(
        self,
        *,
        events: Sequence[Mapping[str, Any]],
        indicators: Sequence[str] | None = None,
        min_score: float = 0.5,
        max_findings: int | None = None,
        include_context: bool = True,
    ) -> Mapping[str, Any]:
        """Return the highest scoring threat findings."""

        if not isinstance(events, Sequence):
            raise TypeError("events must be a sequence of mapping objects")

        normalized_indicators = [
            item.strip().lower()
            for item in (indicators or [])
            if isinstance(item, str) and item.strip()
        ]
        indicator_set = set(normalized_indicators)

        findings: list[ThreatFinding] = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            normalized = _normalize_event(event)
            severity = normalized["severity"]
            score = _severity_score(severity)
            reason_bits: list[str] = []

            indicator_hits: list[str] = []
            if indicator_set:
                payload_blob = " ".join(str(value).lower() for value in normalized.values())
                for candidate in indicator_set:
                    if candidate and candidate in payload_blob:
                        indicator_hits.append(candidate)
                if indicator_hits:
                    score += 0.15 * len(indicator_hits)
                    reason_bits.append(
                        "matched indicators: " + ", ".join(sorted(set(indicator_hits)))
                    )

            anomaly_score = normalized.get("anomaly_score")
            if isinstance(anomaly_score, (int, float)):
                score += min(max(anomaly_score, 0.0), 1.0) * 0.25
                reason_bits.append("included anomaly score signal")

            if normalized.get("failed") is True or normalized.get("status") == "failed":
                score += 0.2
                reason_bits.append("explicit failure status")

            if normalized.get("source_ip") and normalized.get("destination_ip"):
                reason_bits.append("network pivot detected")

            if not reason_bits:
                reason_bits.append("severity-derived score")

            score = max(0.0, min(score, 1.0))
            if score < min_score:
                continue

            finding = ThreatFinding(
                score=round(score, 4),
                severity=severity,
                timestamp=normalized.get("timestamp"),
                reason="; ".join(reason_bits),
                indicator_hits=tuple(sorted(set(indicator_hits))) if indicator_hits else tuple(),
                event=event if include_context else None,
            )
            findings.append(finding)

        findings.sort(key=lambda item: item.score, reverse=True)
        if max_findings is not None and max_findings > 0:
            findings = findings[:max_findings]

        response = {
            "scanned": len(events),
            "findings": [asdict(item) for item in findings],
            "threshold": min_score,
            "indicators": sorted(indicator_set),
        }
        return response


def _normalize_event(event: Mapping[str, Any]) -> MutableMapping[str, Any]:
    payload: MutableMapping[str, Any] = {}
    for key, value in event.items():
        payload[str(key)] = value
    severity = payload.get("severity")
    if isinstance(severity, str):
        payload["severity"] = severity.strip().lower() or "info"
    elif isinstance(severity, (int, float)):
        payload["severity"] = _numeric_severity(float(severity))
    else:
        payload["severity"] = "info"

    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        payload["timestamp"] = timestamp.strip() or None
    elif isinstance(timestamp, (int, float)):
        payload["timestamp"] = str(timestamp)
    else:
        payload["timestamp"] = None
    return payload


def _severity_score(severity: str) -> float:
    return _SEVERITY_WEIGHTS.get(severity.strip().lower(), 0.2)


def _numeric_severity(value: float) -> str:
    if value >= 50:
        return "critical"
    if value >= 40:
        return "error"
    if value >= 30:
        return "warning"
    if value >= 20:
        return "notice"
    return "info"
