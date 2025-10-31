"""Normalize system inventories and metrics into compact payloads."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["SysSnapshot", "SnapshotHost"]


@dataclass(frozen=True)
class SnapshotHost:
    """Representation of a host included in the snapshot."""

    name: str
    metadata: Mapping[str, Any]


class SysSnapshot:
    """Build normalized views of host and metric state."""

    def run(
        self,
        *,
        hosts: Sequence[Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        observations: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        """Return a structured situational snapshot."""

        normalized_hosts = [_normalize_host(entry) for entry in (hosts or [])]
        normalized_metrics = _normalize_mapping(metrics or {})
        normalized_tags = sorted({tag.strip() for tag in (tags or []) if isinstance(tag, str) and tag.strip()})
        normalized_observations = [
            note.strip()
            for note in (observations or [])
            if isinstance(note, str) and note.strip()
        ]

        payload = {
            "hosts": [asdict(host) for host in normalized_hosts],
            "metrics": normalized_metrics,
            "tags": normalized_tags,
            "observations": normalized_observations,
            "summary": {
                "host_count": len(normalized_hosts),
                "metric_count": len(normalized_metrics),
                "tag_count": len(normalized_tags),
            },
        }
        return payload


def _normalize_host(entry: Any) -> SnapshotHost:
    if isinstance(entry, SnapshotHost):
        return entry
    if isinstance(entry, Mapping):
        name = str(entry.get("name") or entry.get("hostname") or entry.get("id") or "unknown").strip()
        metadata = _normalize_mapping(entry)
        metadata.setdefault("name", name)
        return SnapshotHost(name=name or "unknown", metadata=metadata)
    if isinstance(entry, str):
        name = entry.strip()
        return SnapshotHost(name=name or "unknown", metadata={"name": name or "unknown"})
    return SnapshotHost(name="unknown", metadata={"name": "unknown"})


def _normalize_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: MutableMapping[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key)] = value
    return dict(normalized)
