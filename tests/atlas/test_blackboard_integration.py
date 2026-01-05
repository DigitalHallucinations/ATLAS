import asyncio
import sys
from types import SimpleNamespace

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )


class _DummyValidator:
    def __init__(self, *_args, **_kwargs):
        self.schema = {}

    def iter_errors(self, *_args, **_kwargs):
        return []


if "jsonschema" not in sys.modules:
    sys.modules["jsonschema"] = SimpleNamespace(
        Draft202012Validator=_DummyValidator,
        exceptions=SimpleNamespace(SchemaError=Exception),
    )

import pytest

from core.messaging import AgentBus, get_agent_bus
from core.SkillManager import SkillExecutionContext
from modules.Server import RequestContext
from modules.Server.conversation_routes import ConversationAuthorizationError
from modules.Server.routes import AtlasServer
from modules.orchestration.blackboard import BlackboardStore, configure_blackboard


@pytest.fixture(autouse=True)
def _reset_blackboard() -> None:
    configure_blackboard(BlackboardStore())


def test_blackboard_persists_across_contexts(_reset_blackboard):
    context = SkillExecutionContext(conversation_id="conv-123", conversation_history=[])
    context.blackboard.publish_hypothesis("New lead", "Consider exploring dataset A")
    context.blackboard.publish_claim("Verified", "Dataset A aligns with expectations")

    followup_context = SkillExecutionContext(conversation_id="conv-123", conversation_history=[])
    summary = followup_context.blackboard.summary()

    assert summary["counts"]["hypothesis"] == 1
    assert summary["counts"]["claim"] == 1
    assert len(summary["entries"]) == 2


def test_blackboard_routes_crud_cycle(_reset_blackboard):
    server = AtlasServer()
    tenant_context = RequestContext(tenant_id="tenant-alpha")

    created = server.handle_request(
        "/blackboard/conversation/demo",
        method="POST",
        query={
            "category": "hypothesis",
            "title": "Shared insight",
            "content": "Let's examine the latest telemetry logs.",
            "author": "analyst",
        },
        context=tenant_context,
    )
    entry = created["entry"]
    entry_id = entry["id"]
    assert entry["metadata"]["tenant_id"] == "tenant-alpha"

    summary = server.handle_request(
        "/blackboard/conversation/demo",
        method="GET",
        query={"summary": True},
        context=tenant_context,
    )
    assert summary["counts"]["hypothesis"] == 1

    updated = server.handle_request(
        f"/blackboard/conversation/demo/{entry_id}",
        method="PATCH",
        query={"content": "Telemetry parsed and normalized."},
        context=tenant_context,
    )
    assert updated["entry"]["content"] == "Telemetry parsed and normalized."

    deleted = server.handle_request(
        f"/blackboard/conversation/demo/{entry_id}",
        method="DELETE",
        query={},
        context=tenant_context,
    )
    assert deleted["success"] is True

    remaining = server.handle_request(
        "/blackboard/conversation/demo",
        method="GET",
        query={},
        context=tenant_context,
    )
    assert remaining["count"] == 0


def test_blackboard_routes_require_context(_reset_blackboard) -> None:
    server = AtlasServer()

    with pytest.raises(ConversationAuthorizationError):
        server.handle_request(
            "/blackboard/conversation/demo",
            method="POST",
            query={
                "category": "hypothesis",
                "title": "Shared insight",
                "content": "Tenant context required.",
            },
        )


def test_blackboard_routes_enforce_tenant_isolation(_reset_blackboard) -> None:
    server = AtlasServer()
    tenant_alpha = RequestContext(tenant_id="tenant-alpha")
    tenant_beta = RequestContext(tenant_id="tenant-beta")

    created = server.handle_request(
        "/blackboard/conversation/demo",
        method="POST",
        query={
            "category": "hypothesis",
            "title": "Shared insight",
            "content": "Let's examine the latest telemetry logs.",
        },
        context=tenant_alpha,
    )
    entry_id = created["entry"]["id"]

    with pytest.raises(ConversationAuthorizationError):
        server.handle_request(
            "/blackboard/conversation/demo",
            method="GET",
            query={},
            context=tenant_beta,
        )

    with pytest.raises(ConversationAuthorizationError):
        server.handle_request(
            f"/blackboard/conversation/demo/{entry_id}",
            method="PATCH",
            query={"content": "Cross tenant update"},
            context=tenant_beta,
        )

    with pytest.raises(ConversationAuthorizationError):
        server.handle_request(
            f"/blackboard/conversation/demo/{entry_id}",
            method="DELETE",
            query={},
            context=tenant_beta,
        )


def test_blackboard_concurrent_publication(_reset_blackboard):
    async def _run() -> None:
        context = SkillExecutionContext(
            conversation_id="conv-concurrent", conversation_history=[]
        )
        client = context.blackboard
        loop = asyncio.get_running_loop()

        await asyncio.gather(
            loop.run_in_executor(None, client.publish_hypothesis, "Idea A", "Outline approach"),
            loop.run_in_executor(None, client.publish_claim, "Validated", "Results confirmed"),
            loop.run_in_executor(None, client.publish_artifact, "Attachment", "Stored in S3"),
        )

        entries = client.list_entries()
        assert len(entries) == 3

        followup = SkillExecutionContext(
            conversation_id="conv-concurrent", conversation_history=[]
        )
        assert len(followup.blackboard.list_entries()) == 3

    asyncio.run(_run())


def test_blackboard_stream_receives_updates(_reset_blackboard):
    async def _run() -> None:
        server = AtlasServer()
        scope_id = "stream-demo"
        stream = server.stream_blackboard_events(
            "conversation",
            scope_id,
            context=RequestContext(tenant_id="default"),
        )

        next_event = asyncio.create_task(stream.__anext__())

        context = SkillExecutionContext(conversation_id=scope_id, conversation_history=[])
        context.blackboard.publish_hypothesis("Signal", "Monitoring spikes")

        event = await asyncio.wait_for(next_event, timeout=1.0)
        assert event["action"] == "created"
        assert event["entry"]["scope_id"] == scope_id

        await stream.aclose()

    asyncio.run(_run())


def test_blackboard_normalizes_tags(_reset_blackboard):
    store = configure_blackboard()

    entry = store.create_entry(
        "scope-x",
        category="hypothesis",
        title="Observation",
        content="Investigate duplicates",
        tags=["Alpha", "Alpha", "  ", 101],  # type: ignore[list-item]
    )

    assert entry.tags == ("Alpha", "101")

    updated = store.update_entry(
        entry.entry_id,
        scope_id="scope-x",
        tags=["Beta", " beta ", "Beta"],
    )

    assert updated is not None
    assert updated.tags == ("Beta", "beta")
