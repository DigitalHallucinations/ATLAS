import asyncio
import sys
import types

aiohttp_module = sys.modules.get("aiohttp")
if aiohttp_module is None:  # pragma: no cover - test dependency shim
    aiohttp_module = types.ModuleType("aiohttp")
    sys.modules["aiohttp"] = aiohttp_module


if not hasattr(aiohttp_module, "ClientTimeout"):  # pragma: no cover - test dependency shim
    class _ClientTimeout:
        def __init__(self, *args, **kwargs) -> None:
            pass

    aiohttp_module.ClientTimeout = _ClientTimeout

from modules.Tools.Base_Tools.consensus_vote import consensus_vote
from modules.Tools.tool_maps.maps import function_map
from modules.orchestration.consensus import Proposal


def test_consensus_vote_selects_highest_score() -> None:
    recorded_weights: dict[str, float | None] = {}

    async def proposer_alpha(ctx):
        recorded_weights["alpha"] = ctx.get("weight")
        return Proposal(
            participant_id="alpha",
            content="Plan Alpha",
            score=ctx.get("weight", 0.0),
            rationale="Baseline option",
        )

    async def proposer_beta(ctx):
        recorded_weights["beta"] = ctx.get("weight")
        return Proposal(
            participant_id="beta",
            content="Plan Beta",
            score=1.0,
            rationale="Preferred option",
        )

    result = asyncio.run(
        consensus_vote(
            participants=[
                {"id": "alpha", "callable": proposer_alpha},
                {"id": "beta", "propose": proposer_beta},
            ],
            quorum_threshold=0.5,
            timeout=0.2,
            context={"topic": "demo"},
            participant_contexts={
                "alpha": {"weight": 0.2},
                "beta": {"weight": 0.9},
            },
        )
    )

    assert result["success"] is True
    assert result["selected_proposal"] is not None
    assert result["selected_proposal"]["participant"] == "beta"
    assert recorded_weights["alpha"] == 0.2
    assert recorded_weights["beta"] == 0.9


def test_consensus_vote_trace_includes_participants() -> None:
    async def proposer_a(_ctx):
        return {"content": "Option A", "score": 0.6}

    async def proposer_b(_ctx):
        return {"content": "Option B", "score": 0.4}

    result = asyncio.run(
        consensus_vote(
            participants=[
                ("participant_a", proposer_a),
                ("participant_b", proposer_b),
            ],
            quorum_threshold=1.0,
            timeout=0.5,
        )
    )

    trace = result["trace"]
    assert trace["protocol"] == "vote"
    participant_status = {
        entry["participant"]: entry["status"] for entry in trace["participants"]
    }
    assert participant_status == {
        "participant_a": "responded",
        "participant_b": "responded",
    }
    assert trace["selected"]["participant"] == result["selected_proposal"]["participant"]


def test_consensus_vote_accessible_via_function_map() -> None:
    assert "consensus.vote" in function_map
    tool = function_map["consensus.vote"]

    async def proposer(_ctx):
        return Proposal(
            participant_id="solo",
            content="Solo Plan",
            score=0.5,
        )

    async def runner():
        return await tool(
            participants=[{"id": "solo", "callable": proposer}],
            quorum_threshold=0.0,
            timeout=0.0,
        )

    result = asyncio.run(runner())

    assert result["success"] is True
    assert result["selected_proposal"]["participant"] == "solo"
    assert result["trace"]["status"] == "success"
