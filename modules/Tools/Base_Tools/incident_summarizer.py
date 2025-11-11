"""Produce structured incident summaries for operational reviews."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping, Sequence

from modules.Tools.Base_Tools.utils.normalization import normalize_mapping_keys

__all__ = ["IncidentSummarizer", "IncidentEvent"]


@dataclass(frozen=True)
class IncidentEvent:
    """Normalized event included in an incident timeline."""

    timestamp: str
    description: str
    metadata: Mapping[str, Any]


class IncidentSummarizer:
    """Assemble executive-ready incident briefs."""

    def run(
        self,
        *,
        timeline: Sequence[Mapping[str, Any]],
        impact: Mapping[str, Any] | None = None,
        actions: Sequence[Mapping[str, Any]] | None = None,
        include_recommendations: bool = True,
    ) -> Mapping[str, Any]:
        """Return a structured incident summary."""

        if not isinstance(timeline, Sequence) or not timeline:
            raise ValueError("timeline must be a non-empty sequence of mappings")

        normalized_timeline = [_normalize_event(item) for item in timeline]
        normalized_timeline.sort(key=lambda event: event.timestamp)

        normalized_impact = normalize_mapping_keys(impact)
        normalized_actions = [normalize_mapping_keys(action) for action in (actions or [])]

        headline = _derive_headline(normalized_timeline, normalized_impact)
        recommendations = (
            _generate_recommendations(normalized_timeline, normalized_impact, normalized_actions)
            if include_recommendations
            else []
        )

        summary = {
            "headline": headline,
            "timeline": [asdict(event) for event in normalized_timeline],
            "impact": normalized_impact,
            "actions": normalized_actions,
            "recommendations": recommendations,
        }
        return summary


def _normalize_event(event: Mapping[str, Any]) -> IncidentEvent:
    if isinstance(event, IncidentEvent):
        return event

    timestamp = str(event.get("timestamp") or event.get("time") or "").strip()
    if not timestamp:
        raise ValueError("incident events must include a timestamp")

    description = str(event.get("description") or event.get("message") or "").strip()
    if not description:
        description = "Event recorded"

    metadata = normalize_mapping_keys(event)
    metadata.setdefault("description", description)
    return IncidentEvent(timestamp=timestamp, description=description, metadata=metadata)


def _derive_headline(timeline: Sequence[IncidentEvent], impact: Mapping[str, Any]) -> str:
    latest = timeline[-1]
    severity = str(impact.get("severity", "")).strip().capitalize()
    affected = impact.get("affected_assets")
    if isinstance(affected, Iterable) and not isinstance(affected, (str, bytes)):
        affected_list = ", ".join(str(item) for item in affected)
    else:
        affected_list = str(affected) if affected else "environment"

    headline_parts = []
    if severity:
        headline_parts.append(f"{severity} incident")
    else:
        headline_parts.append("Incident update")
    headline_parts.append(f"latest at {latest.timestamp}")
    headline_parts.append(f"impacting {affected_list}")
    return ", ".join(headline_parts)


def _generate_recommendations(
    timeline: Sequence[IncidentEvent],
    impact: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> list[str]:
    recommendations: list[str] = []

    containment_done = any(
        str(action.get("status", "")).lower() == "complete"
        and str(action.get("category", "")).lower() == "containment"
        for action in actions
    )
    if not containment_done:
        recommendations.append("Confirm containment owners and validate blocking controls remain active.")

    eradication_step = next(
        (action for action in actions if str(action.get("category", "")).lower() == "eradication"),
        None,
    )
    if eradication_step is None:
        recommendations.append("Schedule eradication tasks and assign follow-up owners.")

    if not any("postmortem" in str(event.description).lower() for event in timeline):
        recommendations.append("Plan a post-incident review with contributing teams.")

    if impact.get("customer_visible"):
        recommendations.append("Coordinate customer comms with approved messaging templates.")

    return recommendations
