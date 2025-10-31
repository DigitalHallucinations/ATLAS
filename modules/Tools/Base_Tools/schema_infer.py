"""Schema inference utility for semi-structured ingestion payloads."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from modules.logging.logger import setup_logger

__all__ = ["SchemaInference", "SchemaInferenceError"]


logger = setup_logger(__name__)


class SchemaInferenceError(RuntimeError):
    """Raised when sample records cannot be analysed."""


@dataclass(frozen=True)
class FieldSummary:
    name: str
    types: Sequence[str]
    nullable: bool
    examples: Sequence[Any]
    presence: float


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return "unknown"


class SchemaInference:
    """Infer basic schema metadata from representative records."""

    def __init__(self, *, max_examples: int = 3) -> None:
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        self._max_examples = max_examples

    async def run(self, *, records: Iterable[Mapping[str, Any]]) -> Mapping[str, object]:
        samples = list(records)
        if not samples:
            raise SchemaInferenceError("At least one record is required for inference")

        await asyncio.sleep(0)

        type_map: MutableMapping[str, set[str]] = defaultdict(set)
        example_map: MutableMapping[str, list[Any]] = defaultdict(list)
        presence_map: MutableMapping[str, int] = defaultdict(int)

        for record in samples:
            if not isinstance(record, Mapping):
                raise SchemaInferenceError("Records must be mappings")
            for key, value in record.items():
                type_map[key].add(_infer_type(value))
                if value is not None and len(example_map[key]) < self._max_examples:
                    example_map[key].append(value)
                if value is not None:
                    presence_map[key] += 1

        summaries = []
        total = len(samples)
        for field_name in sorted(type_map.keys()):
            types = sorted(type_map[field_name])
            presence_ratio = presence_map[field_name] / total if total else 0.0
            summary = FieldSummary(
                name=field_name,
                types=tuple(types),
                nullable="null" in types,
                examples=tuple(example_map[field_name]),
                presence=round(presence_ratio, 3),
            )
            summaries.append(summary.__dict__)

        logger.info("Inferred schema for %d fields", len(summaries))
        return {
            "record_count": total,
            "fields": summaries,
        }
