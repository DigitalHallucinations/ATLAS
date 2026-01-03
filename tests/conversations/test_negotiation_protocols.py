import asyncio

from modules.orchestration.consensus import (
    NegotiationParticipant,
    NegotiationStatus,
    Proposal,
    run_protocol,
)


def test_vote_protocol_selects_highest_score() -> None:
    async def proposer_a(_context):
        return Proposal(
            participant_id="alpha",
            content="Plan A",
            score=0.6,
            rationale="Primary option",
        )

    async def proposer_b(_context):
        return Proposal(
            participant_id="beta",
            content="Plan B",
            score=0.9,
            rationale="Better option",
        )

    participants = [
        NegotiationParticipant(participant_id="alpha", propose=proposer_a),
        NegotiationParticipant(participant_id="beta", propose=proposer_b),
    ]

    outcome = asyncio.run(
        run_protocol(
            "vote",
            participants,
            context={},
            quorum_threshold=0.5,
            timeout=0.5,
        )
    )

    assert outcome.success
    assert outcome.selected_proposal is not None
    assert outcome.selected_proposal.participant_id == "beta"
    assert outcome.trace.status is NegotiationStatus.SUCCESS


def test_vote_protocol_quorum_failure_triggers_status() -> None:
    async def slow_proposer(_context):
        await asyncio.sleep(0.2)
        return Proposal(
            participant_id="slow",
            content="Delayed",
            score=0.1,
        )

    async def fast_proposer(_context):
        return Proposal(
            participant_id="fast",
            content="Quick",
            score=0.8,
        )

    participants = [
        NegotiationParticipant(participant_id="fast", propose=fast_proposer),
        NegotiationParticipant(participant_id="slow", propose=slow_proposer),
    ]

    outcome = asyncio.run(
        run_protocol(
            "vote",
            participants,
            context={},
            quorum_threshold=1.0,
            timeout=0.05,
        )
    )

    assert not outcome.success
    assert outcome.trace.status in {NegotiationStatus.TIMEOUT, NegotiationStatus.QUORUM_FAILED}


def test_vote_protocol_timeout_status() -> None:
    async def sleepy(_context):
        await asyncio.sleep(0.2)
        return Proposal(participant_id="sleep", content="Z", score=0.1)

    participants = [NegotiationParticipant(participant_id="sleep", propose=sleepy)]

    outcome = asyncio.run(
        run_protocol(
            "vote",
            participants,
            context={},
            quorum_threshold=0.5,
            timeout=0.01,
        )
    )

    assert not outcome.success
    assert outcome.trace.status is NegotiationStatus.TIMEOUT
