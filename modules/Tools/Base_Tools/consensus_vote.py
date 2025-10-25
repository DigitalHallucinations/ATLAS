"""Tool wrapper exposing the consensus vote orchestration protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from modules.orchestration.consensus import (
    NegotiationOutcome,
    NegotiationParticipant,
    Proposal,
    vote,
)

__all__ = ["consensus_vote"]

ParticipantDefinition = Any


async def consensus_vote(
    *,
    participants: Sequence[ParticipantDefinition] | Mapping[str, ParticipantDefinition],
    quorum_threshold: float,
    timeout: float,
    context: Mapping[str, Any] | None = None,
    participant_contexts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Execute the consensus vote protocol and return a serialisable snapshot.

    Parameters
    ----------
    participants:
        Iterable describing the negotiation participants. Entries may either be
        :class:`~modules.orchestration.consensus.NegotiationParticipant`
        instances, mappings with ``id``/``participant_id`` keys and a callable
        under ``callable``/``propose``/``fn``/``function``, or a tuple of
        ``(participant_id, callable)``.
    quorum_threshold:
        Required quorum expressed as a ratio between 0.0 and 1.0 (inclusive).
    timeout:
        Per-participant timeout in seconds. Non-positive values disable the
        timeout guard.
    context:
        Optional mapping supplied to each participant during proposal
        collection.
    participant_contexts:
        Optional mapping or iterable of context overrides applied on a
        per-participant basis. When an iterable is supplied each item should be a
        mapping with an ``id`` or ``participant_id`` key identifying the target
        participant.

    Returns
    -------
    Mapping[str, Any]
        Structured payload containing outcome flags, the selected proposal (if
        any), and the full negotiation trace serialised as dictionaries.
    """

    base_context = _normalize_context(context)
    participant_overrides = _normalize_participant_contexts(participant_contexts)
    participant_objects = _normalize_participants(participants, participant_overrides)

    if not participant_objects:
        raise ValueError("At least one participant definition is required.")

    outcome = await vote(
        participant_objects,
        context=base_context,
        quorum_threshold=_coerce_quorum(quorum_threshold),
        timeout=_coerce_timeout(timeout),
    )

    return _serialize_outcome(outcome)


def _normalize_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(context, Mapping):
        return dict(context)
    return {}


def _normalize_participant_contexts(
    contexts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    if isinstance(contexts, Mapping):
        for participant_id, payload in contexts.items():
            key = _coerce_identifier(participant_id)
            if not key or not isinstance(payload, Mapping):
                continue
            normalized[key] = dict(payload)
    elif isinstance(contexts, Sequence) and not isinstance(contexts, (str, bytes, bytearray)):
        for entry in contexts:
            if not isinstance(entry, Mapping):
                continue
            key = _coerce_identifier(entry.get("id") or entry.get("participant_id"))
            if not key:
                continue
            payload = entry.get("context") or entry
            if isinstance(payload, Mapping):
                normalized[key] = dict(payload)
    return normalized


def _normalize_participants(
    participants: Sequence[ParticipantDefinition] | Mapping[str, ParticipantDefinition],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[NegotiationParticipant]:
    normalized: list[NegotiationParticipant] = []

    if isinstance(participants, Mapping):
        items = participants.items()
    else:
        items = enumerate(participants)  # type: ignore[arg-type]

    for _, raw_entry in items:
        participant = _coerce_participant(raw_entry, overrides)
        if participant is not None:
            normalized.append(participant)

    return normalized


def _coerce_participant(
    entry: ParticipantDefinition,
    overrides: Mapping[str, Mapping[str, Any]],
) -> NegotiationParticipant | None:
    if isinstance(entry, NegotiationParticipant):
        return entry

    participant_id: str | None = None
    callable_obj: Callable[[Mapping[str, Any]], Awaitable[Any]] | Callable[[Mapping[str, Any]], Any] | None = None
    override_payload: Mapping[str, Any] | None = None

    if isinstance(entry, Mapping):
        participant_id = _coerce_identifier(
            entry.get("participant_id") or entry.get("id") or entry.get("name")
        )
        callable_obj = _extract_callable(entry)
        context_override = entry.get("context")
        if isinstance(context_override, Mapping):
            override_payload = context_override
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        seq = list(entry)
        if seq:
            participant_id = _coerce_identifier(seq[0])
        if len(seq) > 1 and callable(seq[1]):
            callable_obj = seq[1]  # type: ignore[assignment]
        if len(seq) > 2 and isinstance(seq[2], Mapping):
            override_payload = seq[2]

    if not participant_id or callable_obj is None:
        return None

    merged_override: Mapping[str, Any] | None = overrides.get(participant_id)
    if override_payload:
        if merged_override:
            merged = dict(merged_override)
            merged.update(dict(override_payload))
            merged_override = merged
        else:
            merged_override = dict(override_payload)

    async def _propose(runtime_context: Mapping[str, Any]) -> Proposal:
        call_context = dict(runtime_context) if isinstance(runtime_context, Mapping) else {}
        if merged_override:
            call_context.update(merged_override)
        result = callable_obj(call_context)
        if asyncio.iscoroutine(result):
            result = await result
        return _coerce_proposal(result, participant_id)

    return NegotiationParticipant(participant_id=participant_id, propose=_propose)


def _extract_callable(entry: Mapping[str, Any]):
    for key in ("callable", "propose", "fn", "function", "handler"):
        candidate = entry.get(key)
        if callable(candidate):
            return candidate
    return None


def _coerce_identifier(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _coerce_quorum(value: Any) -> float:
    try:
        quorum = float(value)
    except (TypeError, ValueError):
        quorum = 0.0
    return max(0.0, min(1.0, quorum))


def _coerce_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = 0.0
    return max(0.0, timeout)


def _coerce_proposal(result: Any, participant_id: str) -> Proposal:
    if isinstance(result, Proposal):
        return result
    if isinstance(result, Mapping):
        content = result.get("content")
        if content is None:
            content = result.get("text")
        if content is None:
            content = ""
        return Proposal(
            participant_id=result.get("participant_id") or participant_id,
            content=str(content),
            score=result.get("score"),
            rationale=result.get("rationale"),
            payload=result.get("payload"),
            metadata=dict(result.get("metadata") or {}),
        )
    return Proposal(participant_id=participant_id, content=str(result))


def _serialize_outcome(outcome: NegotiationOutcome) -> Mapping[str, Any]:
    selected = _serialize_proposal(outcome.selected_proposal)
    trace_payload = outcome.trace.to_dict() if outcome.trace else None
    return {
        "success": outcome.success,
        "status": outcome.status.value,
        "selected_proposal": selected,
        "trace": trace_payload,
    }


def _serialize_proposal(proposal: Proposal | None) -> Mapping[str, Any] | None:
    if proposal is None:
        return None
    return {
        "participant": proposal.participant_id,
        "content": proposal.content,
        "score": proposal.score,
        "rationale": proposal.rationale,
        "payload": proposal.payload,
        "metadata": dict(proposal.metadata or {}),
    }
