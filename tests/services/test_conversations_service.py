from __future__ import annotations

from unittest.mock import Mock

from ATLAS.services.conversations import ConversationService


def _service(repository=None, logger=None, tenant_id="tenant") -> ConversationService:
    if repository is None:
        repository = Mock()
    if logger is None:
        logger = Mock()
    return ConversationService(
        repository=repository,
        logger=logger,
        tenant_id=tenant_id,
    )


def test_get_recent_conversations_coerces_limit_and_uses_repository() -> None:
    repository = Mock()
    repository.list_conversations_for_tenant.return_value = [
        {"id": "one"},
        {"id": "two"},
    ]

    service = _service(repository=repository)

    result = service.get_recent_conversations(limit="5")

    repository.list_conversations_for_tenant.assert_called_once_with(
        "tenant", limit=5, order="desc"
    )
    assert result == [{"id": "one"}, {"id": "two"}]


def test_get_conversation_messages_respects_limit_and_batch_size() -> None:
    repository = Mock()
    repository.stream_conversation_messages.return_value = [
        {"id": "m1"},
        {"id": "m2"},
        {"id": "m3"},
    ]

    service = _service(repository=repository)

    result = service.get_conversation_messages(
        "conv",
        limit=2,
        include_deleted=False,
        batch_size="3",
    )

    repository.stream_conversation_messages.assert_called_once_with(
        "conv",
        tenant_id="tenant",
        batch_size=3,
        direction="forward",
        include_deleted=False,
    )
    assert result == [{"id": "m1"}, {"id": "m2"}]


def test_reset_conversation_messages_propagates_error_message() -> None:
    repository = Mock()
    repository.reset_conversation.side_effect = RuntimeError("boom")
    logger = Mock()

    service = _service(repository=repository, logger=logger)

    result = service.reset_conversation_messages("conv")

    repository.reset_conversation.assert_called_once_with("conv", tenant_id="tenant")
    logger.error.assert_called_once()
    assert result["success"] is False
    assert "boom" in result["error"]


def test_listener_invocation_and_cleanup() -> None:
    logger = Mock()
    service = _service(logger=logger)

    events: list[dict[str, str]] = []

    def recorder(payload: dict[str, str]) -> None:
        events.append(payload)

    def failing_listener(_payload: dict[str, str]) -> None:
        raise RuntimeError("listener failure")

    service.add_listener(recorder)
    service.add_listener(failing_listener)

    service.notify_updated("identifier", reason="deleted")

    assert events == [{"conversation_id": "identifier", "reason": "deleted"}]
    logger.debug.assert_called_once()

    service.remove_listener(recorder)
    service.remove_listener(failing_listener)
    service.notify_updated("identifier")

    assert len(events) == 1


def test_retention_runner_unavailable_returns_error() -> None:
    service = _service()

    available, reason, context = service.assess_retention_availability(runner=None)
    assert available is False
    assert "endpoint" in (reason or "").lower()
    assert context["tenant_id"] == "tenant"

    result = service.run_conversation_retention(runner=None)
    assert result["success"] is False
    assert "available" in result["error"].lower()


def test_retention_requires_admin_role_before_invoking_runner() -> None:
    runner = Mock()
    service = _service()

    result = service.run_conversation_retention(runner=runner, context={"roles": ["user"]})

    runner.assert_not_called()
    assert result["success"] is False
    assert "admin" in result["error"].lower()


def test_retention_runs_when_authorised_and_returns_stats() -> None:
    runner = Mock(return_value={"messages": {"deleted": 2}})
    service = _service()

    result = service.run_conversation_retention(
        runner=runner,
        context={"roles": ["admin"], "tenant_id": "custom", "user_id": "alice"},
    )

    runner.assert_called_once()
    call = runner.call_args
    assert call.kwargs["context"]["tenant_id"] == "custom"
    assert call.kwargs["context"]["roles"] == ("admin",)
    assert call.kwargs["context"]["user_id"] == "alice"
    assert result["success"] is True
    assert result["stats"] == {"messages": {"deleted": 2}}
