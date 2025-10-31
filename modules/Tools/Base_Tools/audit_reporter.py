"""Generate governance-friendly snapshots from security events."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["AuditReporter"]


class AuditReporter:
    """Summarize events and control coverage for audit consumers."""

    def run(
        self,
        *,
        events: Sequence[Mapping[str, Any]],
        controls: Sequence[str] | None = None,
        include_passed: bool = True,
        window: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Return an audit-oriented aggregation of the supplied events."""

        if not isinstance(events, Sequence):
            raise TypeError("events must be a sequence of mapping objects")

        normalized_controls = [
            item.strip()
            for item in (controls or [])
            if isinstance(item, str) and item.strip()
        ]

        severity_counter: Counter[str] = Counter()
        control_status: MutableMapping[str, Counter[str]] = defaultdict(Counter)
        notes: list[str] = []
        total_failures = 0
        total_passes = 0

        for event in events:
            if not isinstance(event, Mapping):
                continue
            severity = str(event.get("severity", "info")).strip().lower() or "info"
            severity_counter[severity] += 1

            status = str(event.get("status", "")).strip().lower()
            if status in {"failed", "fail", "non_compliant"}:
                total_failures += 1
            elif status in {"passed", "pass", "compliant"}:
                total_passes += 1

            event_controls = event.get("controls")
            if isinstance(event_controls, Sequence):
                for control in event_controls:
                    control_id = str(control).strip()
                    if not control_id:
                        continue
                    control_status[control_id][status or "unknown"] += 1

            rationale = event.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                notes.append(rationale.strip())

        filtered_control_summary: dict[str, Mapping[str, int]] = {}
        requested_controls = set(normalized_controls)
        for control_id, status_counts in control_status.items():
            if requested_controls and control_id not in requested_controls:
                continue
            if not include_passed:
                status_counts = Counter(
                    {
                        status: count
                        for status, count in status_counts.items()
                        if status not in {"passed", "pass", "compliant"}
                    }
                )
            filtered_control_summary[control_id] = dict(status_counts)

        window_payload = {}
        if isinstance(window, Mapping):
            start = window.get("start")
            end = window.get("end")
            if isinstance(start, str) and start.strip():
                window_payload["start"] = start.strip()
            if isinstance(end, str) and end.strip():
                window_payload["end"] = end.strip()

        summary = {
            "window": window_payload,
            "totals": {
                "events": len(events),
                "failures": total_failures,
                "passes": total_passes,
            },
            "by_severity": dict(severity_counter),
            "controls": filtered_control_summary,
            "notes": sorted(set(notes)),
        }
        return summary
