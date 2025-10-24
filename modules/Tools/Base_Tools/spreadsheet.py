"""Structured spreadsheet helper supporting append and replace flows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpreadsheetRow:
    """Represents a normalized spreadsheet row."""

    data: Mapping[str, object]


def _normalize_rows(rows: Iterable[Mapping[str, object]]) -> list[SpreadsheetRow]:
    normalized: list[SpreadsheetRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        normalized.append(SpreadsheetRow(data=dict(row)))
    return normalized


class SpreadsheetTool:
    """Maintain a lightweight tabular store keyed by sheet identifier."""

    def __init__(self) -> None:
        self._sheets: MutableMapping[str, list[SpreadsheetRow]] = {}

    async def run(
        self,
        *,
        sheet_id: str,
        rows: Sequence[Mapping[str, object]],
        mode: str = "append",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, object]:
        if not isinstance(sheet_id, str) or not sheet_id.strip():
            raise ValueError("Sheet identifier must be provided.")
        if not rows:
            raise ValueError("At least one row is required.")

        await asyncio.sleep(0)

        normalized_id = sheet_id.strip()
        normalized_rows = _normalize_rows(rows)
        if not normalized_rows:
            raise ValueError("No valid rows supplied to spreadsheet tool.")

        existing = self._sheets.setdefault(normalized_id, [])
        operation = mode.strip().lower()

        if operation == "replace":
            logger.debug("Replacing contents of sheet %s", normalized_id)
            existing.clear()
        elif operation not in {"append", "update", "replace"}:
            raise ValueError("Unsupported spreadsheet mode. Use append, update, or replace.")

        if operation == "update":
            existing[:] = normalized_rows
        else:
            existing.extend(normalized_rows)

        return {
            "sheet_id": normalized_id,
            "rows": [asdict(row) for row in normalized_rows],
            "total_rows": len(existing),
            "metadata": dict(metadata or {}),
            "mode": operation,
        }


__all__ = ["SpreadsheetTool", "SpreadsheetRow"]

