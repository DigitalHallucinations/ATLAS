"""Async streaming utilities for tool execution."""
from __future__ import annotations

import inspect
from collections.abc import AsyncIterator as AsyncIteratorABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from modules.logging.logger import setup_logger

from .cache import record_tool_activity, stringify_tool_value

logger = setup_logger(__name__)


async def collect_async_chunks(stream: AsyncIterator) -> str:
    """Consume an async iterator of chunks into a single string."""

    chunks = []

    async for chunk in stream:
        if chunk is None:
            continue
        if isinstance(chunk, dict):
            text = chunk.get("content") or chunk.get("text") or chunk.get("message")
            if text is None:
                text = str(chunk)
        else:
            text = str(chunk)
        chunks.append(text)

    return "".join(chunks)


async def gather_async_iterator(stream: AsyncIterator) -> List[Any]:
    """Consume an async iterator into a list, skipping ``None`` items."""

    items: List[Any] = []

    async for item in stream:
        if item is None:
            continue
        items.append(item)

    return items


def is_async_stream(candidate: Any) -> bool:
    return isinstance(candidate, AsyncIteratorABC) or inspect.isasyncgen(candidate)


@dataclass
class ToolStreamCapture:
    items: List[Any]
    text: str
    entry: Dict[str, Any]


async def stream_tool_iterator(
    stream: AsyncIterator,
    *,
    log_entry: Dict[str, Any],
    active_entry: Optional[Dict[str, Any]] = None,
    on_chunk: Optional[Callable[[Any], None]] = None,
) -> ToolStreamCapture:
    """Iterate ``stream`` producing incremental tool activity updates."""

    collected_items: List[Any] = []
    text_fragments: List[str] = []
    if active_entry is None:
        active_entry = record_tool_activity({**log_entry, "result": []})
    else:
        active_entry = record_tool_activity({**log_entry, "result": []}, replace=active_entry)

    async for item in stream:
        if item is None:
            continue

        collected_items.append(item)
        text_value = stringify_tool_value(item)
        if text_value:
            text_fragments.append(text_value)

        if on_chunk is not None:
            try:
                on_chunk(item)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Tool chunk callback failed")

        update_payload = {
            **log_entry,
            "result": list(collected_items),
            "result_text": "".join(text_fragments),
            "status": log_entry.get("status", "running"),
            "completed_at": datetime.utcnow().isoformat(timespec="milliseconds"),
        }
        active_entry = record_tool_activity(update_payload, replace=active_entry)

    return ToolStreamCapture(collected_items, "".join(text_fragments), active_entry)


__all__ = [
    "ToolStreamCapture",
    "collect_async_chunks",
    "gather_async_iterator",
    "is_async_stream",
    "stream_tool_iterator",
]
