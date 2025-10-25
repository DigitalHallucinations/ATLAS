"""Aggregate tool activity and negotiation trace diagnostics for inspection."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from ATLAS import ToolManager as ToolManagerModule

__all__ = ["trace_explain"]


def trace_explain(
    *,
    conversation_id: str | None = None,
    trace_id: str | None = None,
    start_at: str | float | int | None = None,
    end_at: str | float | int | None = None,
    offset: int | None = None,
    limit: int | None = None,
    context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return merged diagnostics from the tool activity log and negotiation traces.

    Args:
        conversation_id: Optional conversation identifier used to filter entries.
        trace_id: Optional negotiation trace identifier to isolate a single trace.
        start_at: Optional ISO 8601 timestamp or epoch seconds marking the inclusive
            lower bound for events.
        end_at: Optional ISO 8601 timestamp or epoch seconds marking the inclusive
            upper bound for events.
        offset: Offset within the merged event timeline (defaults to ``0``).
        limit: Maximum number of merged events to include (``None`` returns all).
        context: Optional mapping containing helper accessors (for example a
            ``chat_session`` instance exposing ``get_negotiation_history``).

    Returns:
        Mapping[str, Any]: Structured payload describing the merged diagnostic
        stream alongside applied filters and pagination metadata.
    """

    normalized_context = dict(context) if isinstance(context, Mapping) else {}
    session = _resolve_chat_session(normalized_context)
    context_conversation = _coerce_string(
        normalized_context.get("conversation_id")
        or getattr(session, "conversation_id", None)
    )

    applied_conversation = _coerce_string(conversation_id) or context_conversation
    applied_trace_id = _coerce_string(trace_id)

    start_ts = _coerce_timestamp(start_at)
    end_ts = _coerce_timestamp(end_at)

    page_offset = 0 if offset is None else _coerce_non_negative_int(offset, "offset")
    page_limit = None if limit is None else _coerce_non_negative_int(limit, "limit")

    tool_entries = _safe_get_tool_activity_log()
    negotiation_history = _safe_get_negotiation_history(session, normalized_context)

    activity_events = _build_activity_events(
        tool_entries,
        conversation_filter=applied_conversation,
        start_ts=start_ts,
        end_ts=end_ts,
        fallback_conversation=context_conversation,
    )

    negotiation_events = _build_negotiation_events(
        negotiation_history,
        conversation_filter=applied_conversation,
        trace_filter=applied_trace_id,
        start_ts=start_ts,
        end_ts=end_ts,
        fallback_conversation=context_conversation,
    )

    all_events = sorted(
        activity_events + negotiation_events,
        key=lambda entry: (entry["epoch"] is None, entry["epoch"] or 0.0),
    )

    total_events = len(all_events)
    slice_start = min(page_offset, total_events)
    if page_limit is None:
        slice_end = total_events
    else:
        slice_end = min(slice_start + page_limit, total_events)

    paged_events = [copy.deepcopy(entry) for entry in all_events[slice_start:slice_end]]

    filters: MutableMapping[str, Any] = {}
    if applied_conversation:
        filters["conversation_id"] = applied_conversation
    if applied_trace_id:
        filters["trace_id"] = applied_trace_id
    if start_ts is not None:
        filters["start_at"] = _format_timestamp(start_ts)
    if end_ts is not None:
        filters["end_at"] = _format_timestamp(end_ts)

    context_snapshot: MutableMapping[str, Any] = {}
    if context_conversation and not filters.get("conversation_id"):
        context_snapshot["conversation_id"] = context_conversation

    return {
        "filters": dict(filters),
        "context": dict(context_snapshot),
        "counts": {
            "tool_activity": len(activity_events),
            "negotiation_traces": len(negotiation_events),
        },
        "pagination": {
            "offset": page_offset,
            "limit": page_limit,
            "returned": len(paged_events),
            "total": total_events,
        },
        "events": paged_events,
        "tool_activity": [copy.deepcopy(entry["data"]) for entry in activity_events],
        "negotiation_traces": [
            copy.deepcopy(entry["data"]) for entry in negotiation_events
        ],
    }


def _resolve_chat_session(context: Mapping[str, Any]) -> Any:
    for key in ("chat_session", "session", "conversation"):
        candidate = context.get(key)
        if candidate is not None:
            return candidate
    return None


def _safe_get_tool_activity_log() -> Sequence[Mapping[str, Any]]:
    getter = getattr(ToolManagerModule, "get_tool_activity_log", None)
    if not callable(getter):
        return []
    try:
        result = getter()
    except Exception:
        return []
    if isinstance(result, Sequence):
        return result
    return []


def _safe_get_negotiation_history(
    session: Any, context: Mapping[str, Any]
) -> Sequence[Mapping[str, Any]]:
    history: Sequence[Mapping[str, Any]] | None = None
    if session is not None:
        getter = getattr(session, "get_negotiation_history", None)
        if callable(getter):
            try:
                candidate = getter()
            except Exception:
                candidate = None
            if isinstance(candidate, Sequence):
                history = candidate
    if history is None:
        candidate = context.get("negotiation_history")
        if isinstance(candidate, Sequence):
            history = candidate
    return history or []


def _build_activity_events(
    entries: Iterable[Mapping[str, Any]],
    *,
    conversation_filter: Optional[str],
    start_ts: Optional[float],
    end_ts: Optional[float],
    fallback_conversation: Optional[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_copy = copy.deepcopy(dict(entry))
        conversation = _extract_conversation_id(entry_copy)
        if conversation is None and fallback_conversation:
            conversation = fallback_conversation
        if conversation_filter and conversation != conversation_filter:
            continue
        epoch = _resolve_activity_epoch(entry_copy)
        if not _within_range(epoch, start_ts, end_ts):
            continue
        events.append(
            {
                "type": "tool_activity",
                "conversation_id": conversation,
                "epoch": epoch,
                "timestamp": _format_timestamp(epoch) if epoch is not None else None,
                "data": entry_copy,
            }
        )
    return events


def _build_negotiation_events(
    history: Iterable[Mapping[str, Any]],
    *,
    conversation_filter: Optional[str],
    trace_filter: Optional[str],
    start_ts: Optional[float],
    end_ts: Optional[float],
    fallback_conversation: Optional[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for trace in history:
        if not isinstance(trace, Mapping):
            continue
        if trace_filter and trace.get("id") != trace_filter:
            continue
        trace_copy = copy.deepcopy(dict(trace))
        conversation = _coerce_string(trace_copy.get("conversation_id"))
        if conversation is None and fallback_conversation:
            conversation = fallback_conversation
        if conversation_filter and conversation != conversation_filter:
            continue
        epoch = _resolve_trace_epoch(trace_copy)
        if not _within_range(epoch, start_ts, end_ts):
            continue
        events.append(
            {
                "type": "negotiation_trace",
                "conversation_id": conversation,
                "epoch": epoch,
                "timestamp": _format_timestamp(epoch) if epoch is not None else None,
                "data": trace_copy,
            }
        )
    return events


def _coerce_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_non_negative_int(value: Any, field: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"{field} must be zero or positive")
    return integer


def _coerce_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            normalized = text
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.timestamp()
    return None


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _within_range(
    epoch: Optional[float], start_ts: Optional[float], end_ts: Optional[float]
) -> bool:
    if epoch is None:
        return True
    if start_ts is not None and epoch < start_ts:
        return False
    if end_ts is not None and epoch > end_ts:
        return False
    return True


def _extract_conversation_id(entry: Mapping[str, Any]) -> Optional[str]:
    direct = entry.get("conversation_id") or entry.get("conversationId")
    candidate = _coerce_string(direct)
    if candidate:
        return candidate

    arguments = entry.get("arguments")
    if isinstance(arguments, Mapping):
        for key in ("conversation_id", "conversationId"):
            candidate = _coerce_string(arguments.get(key))
            if candidate:
                return candidate
        context = arguments.get("context")
        if isinstance(context, Mapping):
            for key in ("conversation_id", "conversationId"):
                candidate = _coerce_string(context.get(key))
                if candidate:
                    return candidate

    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("conversation_id", "conversationId"):
            candidate = _coerce_string(metadata.get(key))
            if candidate:
                return candidate

    tracing = entry.get("tracing")
    if isinstance(tracing, Mapping):
        candidate = _coerce_string(tracing.get("conversation_id"))
        if candidate:
            return candidate

    return None


def _resolve_activity_epoch(entry: Mapping[str, Any]) -> Optional[float]:
    for key in ("completed_at", "completedAt", "started_at", "startedAt"):
        epoch = _coerce_timestamp(entry.get(key))
        if epoch is not None:
            return epoch
    metrics = entry.get("metrics")
    if isinstance(metrics, Mapping):
        for key in ("completed_at", "started_at"):
            epoch = _coerce_timestamp(metrics.get(key))
            if epoch is not None:
                return epoch
    return None


def _resolve_trace_epoch(trace: Mapping[str, Any]) -> Optional[float]:
    for key in ("completed_at", "started_at"):
        epoch = _coerce_timestamp(trace.get(key))
        if epoch is not None:
            return epoch
    return None

