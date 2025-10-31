"""Hermes persona helpers for orchestrating multi-step ingestion."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from modules.Tools.Base_Tools.data_bridge import DataBridge

__all__ = [
    "compose_ingestion_playbook",
    "stage_pipeline",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def compose_ingestion_playbook(
    *,
    source: str,
    objectives: Sequence[str],
    checkpoints: Sequence[str] | None = None,
    stakeholders: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """Generate a structured ingestion playbook skeleton."""

    if not source:
        raise ValueError("source must be provided")
    if not objectives:
        raise ValueError("objectives must contain at least one entry")

    await asyncio.sleep(0)

    payload: MutableMapping[str, object] = {
        "source": source,
        "objectives": list(dict.fromkeys(objectives)),
        "checkpoints": list(dict.fromkeys(checkpoints or ())),
        "stakeholders": list(dict.fromkeys(stakeholders or ())),
        "generated_at": _now_iso(),
    }
    payload["sections"] = {
        "Source": {
            "description": f"Ingestion source '{source}' with {len(payload['objectives'])} objectives.",
        },
        "Connectors": {
            "steps": [
                "api_connector",
                "file_ingest",
                "stream_monitor",
                "schema_infer",
                "data_bridge",
            ],
            "notes": "Sequence connectors in the listed order unless governance requires overrides.",
        },
        "Observability": {
            "checkpoints": payload["checkpoints"],
        },
        "Risks": {
            "items": [],
        },
        "Next Actions": {
            "items": ["Confirm contracts", "Schedule dry-run", "Publish pipeline summary"],
        },
    }
    return payload


async def stage_pipeline(
    *,
    source: str,
    operations: Iterable[Mapping[str, object]],
    dry_run: bool = True,
    bridge: DataBridge | None = None,
) -> Mapping[str, object]:
    """Execute a pipeline plan via the shared DataBridge helper."""

    plan = list(operations)
    if not plan:
        raise ValueError("operations must include at least one step")

    orchestrator = bridge or DataBridge()
    result = await orchestrator.run(source=source, operations=plan, dry_run=dry_run)
    result["planned_at"] = _now_iso()
    return result
