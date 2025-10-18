"""Coordination primitives for multi-agent negotiations.

This module provides a light-weight orchestration layer that can be used by
agents or higher-level planners to reach consensus on a final response before
surfacing it to the user.  Three basic protocols are implemented:

``vote``
    Each participant submits a proposal.  The proposal with the highest
    weighted vote wins when quorum is met.

``critique``
    Participants submit proposals and optionally critique their peers.  The
    aggregated score (proposal score plus critique deltas) determines the
    winner when quorum is met.

``contract_net``
    Participants submit bids.  The highest-value bid that meets quorum wins,
    mirroring the classic contract-net protocol used in MAS literature.

The public API intentionally favours simple dataclasses so higher level
systems can serialise the traces for UI or audit purposes without having to
depend on ORM layers or external services.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, List, Mapping, Optional, Sequence


class NegotiationStatus(str, Enum):
    """Enumeration describing the outcome of a negotiation round."""

    SUCCESS = "success"
    QUORUM_FAILED = "quorum_failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(slots=True)
class Proposal:
    """Normalised proposal returned by a negotiation participant."""

    participant_id: str
    content: str
    score: Optional[float] = None
    rationale: Optional[str] = None
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Critique:
    """Represents peer feedback about a proposal."""

    critic_id: str
    target_id: str
    delta: float
    rationale: Optional[str] = None


@dataclass(slots=True)
class ParticipantTrace:
    """Captured data for a single participant during a negotiation."""

    participant_id: str
    status: str
    score: Optional[float] = None
    rationale: Optional[str] = None
    content: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant": self.participant_id,
            "status": self.status,
            "score": self.score,
            "rationale": self.rationale,
            "content": self.content,
            "metadata": dict(self.metadata or {}),
            "error": self.error,
        }


@dataclass(slots=True)
class NegotiationTrace:
    """Structured trace describing an entire negotiation round."""

    protocol: str
    quorum_threshold: float
    timeout: float
    participants: List[ParticipantTrace]
    status: NegotiationStatus
    selected: Optional[Proposal] = None
    critiques: Sequence[Critique] = field(default_factory=tuple)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    notes: Optional[str] = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable representation of the trace."""

        selected_payload: Optional[dict[str, Any]]
        if self.selected is not None:
            selected_payload = {
                "participant": self.selected.participant_id,
                "content": self.selected.content,
                "score": self.selected.score,
                "rationale": self.selected.rationale,
                "metadata": dict(self.selected.metadata or {}),
            }
        else:
            selected_payload = None

        return {
            "id": self.trace_id,
            "protocol": self.protocol,
            "status": self.status.value,
            "quorum_threshold": self.quorum_threshold,
            "timeout": self.timeout,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "selected": selected_payload,
            "participants": [entry.to_dict() for entry in self.participants],
            "critiques": [
                {
                    "critic": critique.critic_id,
                    "target": critique.target_id,
                    "delta": critique.delta,
                    "rationale": critique.rationale,
                }
                for critique in self.critiques
            ],
            "notes": self.notes,
        }


@dataclass(slots=True)
class NegotiationOutcome:
    """Container returned by each protocol with the final decision."""

    status: NegotiationStatus
    trace: NegotiationTrace
    selected_proposal: Optional[Proposal] = None

    @property
    def success(self) -> bool:
        return self.status is NegotiationStatus.SUCCESS and self.selected_proposal is not None


class NegotiationError(RuntimeError):
    """Raised when the orchestration layer encounters a fatal issue."""


Context = Mapping[str, Any]
ProposalFn = Callable[[Context], Awaitable[Proposal]]
CritiqueFn = Callable[[Sequence[Proposal], Context], Awaitable[Sequence[Critique]]]


@dataclass(slots=True)
class NegotiationParticipant:
    """Simple wrapper bundling callbacks for a participant."""

    participant_id: str
    propose: ProposalFn
    critique: Optional[CritiqueFn] = None


async def vote(
    participants: Sequence[NegotiationParticipant],
    *,
    context: Context,
    quorum_threshold: float,
    timeout: float,
) -> NegotiationOutcome:
    """Run a weighted vote amongst ``participants``."""

    trace, proposals = await _collect_proposals(
        participants,
        context=context,
        timeout=timeout,
        protocol="vote",
        quorum_threshold=quorum_threshold,
    )

    if proposals is None:
        return NegotiationOutcome(trace=trace, status=trace.status, selected_proposal=None)

    ballots: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        score = _coerce_score(proposal.score, default=1.0)
        bucket = ballots.setdefault(
            proposal.content,
            {"score": 0.0, "proposal": proposal},
        )
        bucket["score"] += score
        # Prefer the proposal with the highest individual score for tie-breaks.
        if proposal.score is not None and (
            bucket["proposal"].score or float("-inf")
        ) < proposal.score:
            bucket["proposal"] = proposal

    if not ballots:
        trace.status = NegotiationStatus.ERROR
        trace.notes = "No ballots were recorded despite quorum being met."
        trace.completed_at = time.time()
        return NegotiationOutcome(trace=trace, status=trace.status, selected_proposal=None)

    winner_content, aggregate = max(
        ballots.items(),
        key=lambda item: (item[1]["score"], item[1]["proposal"].score or 0.0),
    )
    winner = aggregate["proposal"]
    trace.selected = Proposal(
        participant_id=winner.participant_id,
        content=winner_content,
        score=aggregate["score"],
        rationale=winner.rationale,
        payload=winner.payload,
        metadata=winner.metadata,
    )
    trace.status = NegotiationStatus.SUCCESS
    trace.completed_at = time.time()
    return NegotiationOutcome(
        status=NegotiationStatus.SUCCESS,
        trace=trace,
        selected_proposal=winner,
    )


async def critique(
    participants: Sequence[NegotiationParticipant],
    *,
    context: Context,
    quorum_threshold: float,
    timeout: float,
) -> NegotiationOutcome:
    """Run a critique-driven protocol where peers can adjust scores."""

    trace, proposals = await _collect_proposals(
        participants,
        context=context,
        timeout=timeout,
        protocol="critique",
        quorum_threshold=quorum_threshold,
    )

    if proposals is None:
        return NegotiationOutcome(trace=trace, status=trace.status, selected_proposal=None)

    adjustments, critique_records = await _collect_critiques(
        participants,
        proposals,
        context=context,
        timeout=timeout,
    )
    trace.critiques = critique_records

    score_table: dict[str, float] = {}
    for proposal in proposals:
        score_table[proposal.participant_id] = _coerce_score(proposal.score)

    for target, delta in adjustments.items():
        score_table[target] = score_table.get(target, 0.0) + delta

    winner = max(
        proposals,
        key=lambda proposal: score_table.get(proposal.participant_id, float("-inf")),
    )

    trace.selected = Proposal(
        participant_id=winner.participant_id,
        content=winner.content,
        score=score_table.get(winner.participant_id),
        rationale=winner.rationale,
        payload=winner.payload,
        metadata=winner.metadata,
    )
    trace.status = NegotiationStatus.SUCCESS
    trace.completed_at = time.time()
    return NegotiationOutcome(
        status=NegotiationStatus.SUCCESS,
        trace=trace,
        selected_proposal=winner,
    )


async def contract_net(
    participants: Sequence[NegotiationParticipant],
    *,
    context: Context,
    quorum_threshold: float,
    timeout: float,
) -> NegotiationOutcome:
    """Implement a basic contract-net style auction."""

    trace, proposals = await _collect_proposals(
        participants,
        context=context,
        timeout=timeout,
        protocol="contract_net",
        quorum_threshold=quorum_threshold,
    )

    if proposals is None:
        return NegotiationOutcome(trace=trace, status=trace.status, selected_proposal=None)

    winner = max(
        proposals,
        key=lambda proposal: (
            _coerce_score(proposal.score),
            proposal.metadata.get("utility", 0.0),
        ),
    )

    trace.selected = Proposal(
        participant_id=winner.participant_id,
        content=winner.content,
        score=_coerce_score(winner.score),
        rationale=winner.rationale,
        payload=winner.payload,
        metadata=winner.metadata,
    )
    trace.status = NegotiationStatus.SUCCESS
    trace.completed_at = time.time()
    return NegotiationOutcome(
        status=NegotiationStatus.SUCCESS,
        trace=trace,
        selected_proposal=winner,
    )


async def run_protocol(
    protocol: str,
    participants: Sequence[NegotiationParticipant],
    *,
    context: Context,
    quorum_threshold: float,
    timeout: float,
) -> NegotiationOutcome:
    """Dispatch ``protocol`` by name."""

    normalized = (protocol or "").strip().lower() or "vote"
    if normalized == "vote":
        return await vote(
            participants,
            context=context,
            quorum_threshold=quorum_threshold,
            timeout=timeout,
        )
    if normalized == "critique":
        return await critique(
            participants,
            context=context,
            quorum_threshold=quorum_threshold,
            timeout=timeout,
        )
    if normalized == "contract_net":
        return await contract_net(
            participants,
            context=context,
            quorum_threshold=quorum_threshold,
            timeout=timeout,
        )

    raise NegotiationError(f"Unsupported negotiation protocol: {protocol}")


async def _collect_proposals(
    participants: Sequence[NegotiationParticipant],
    *,
    context: Context,
    timeout: float,
    protocol: str,
    quorum_threshold: float,
) -> tuple[NegotiationTrace, Optional[List[Proposal]]]:
    quorum_threshold = max(0.0, min(1.0, quorum_threshold))
    timeout = max(timeout, 0.0)
    participant_traces: List[ParticipantTrace] = []
    proposals: List[Proposal] = []
    total = len(participants)

    trace = NegotiationTrace(
        protocol=protocol,
        quorum_threshold=quorum_threshold,
        timeout=timeout,
        participants=participant_traces,
        status=NegotiationStatus.ERROR,
    )

    responded = 0
    timed_out = False

    for participant in participants:
        status = "unknown"
        record = ParticipantTrace(participant_id=participant.participant_id, status=status)
        participant_traces.append(record)

        try:
            proposal = await asyncio.wait_for(
                participant.propose(context),
                timeout=timeout if timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            record.status = "timeout"
            timed_out = True
            continue
        except Exception as exc:  # pragma: no cover - defensive logging upstream
            record.status = "error"
            record.error = str(exc)
            continue

        responded += 1
        record.status = "responded"
        record.score = proposal.score
        record.rationale = proposal.rationale
        record.content = proposal.content
        record.metadata = proposal.metadata
        proposals.append(proposal)

    required = int(total * quorum_threshold)
    if total and quorum_threshold > 0 and (total * quorum_threshold) % 1:
        required += 1

    if required == 0:
        required = 1 if total else 0

    if responded < required:
        trace.status = NegotiationStatus.TIMEOUT if timed_out else NegotiationStatus.QUORUM_FAILED
        trace.notes = (
            f"Collected {responded} of {total} proposals; required {required} responses for quorum."
        )
        trace.completed_at = time.time()
        return trace, None

    trace.status = NegotiationStatus.SUCCESS
    return trace, proposals


async def _collect_critiques(
    participants: Sequence[NegotiationParticipant],
    proposals: Sequence[Proposal],
    *,
    context: Context,
    timeout: float,
) -> tuple[dict[str, float], Sequence[Critique]]:
    adjustments: dict[str, float] = {}
    critiques: List[Critique] = []

    if not proposals:
        return adjustments, critiques

    for participant in participants:
        if participant.critique is None:
            continue

        try:
            feedback = await asyncio.wait_for(
                participant.critique(proposals, context),
                timeout=timeout if timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            continue
        except Exception:  # pragma: no cover - defensive logging left to caller
            continue

        if not feedback:
            continue

        for item in feedback:
            if not isinstance(item, Critique):
                continue
            critiques.append(item)
            adjustments[item.target_id] = adjustments.get(item.target_id, 0.0) + item.delta

    return adjustments, critiques


def _coerce_score(value: Optional[float], *, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = [
    "NegotiationError",
    "NegotiationOutcome",
    "NegotiationParticipant",
    "NegotiationStatus",
    "NegotiationTrace",
    "Proposal",
    "Critique",
    "run_protocol",
    "vote",
    "critique",
    "contract_net",
]
