"""Tests for exposing conversation helpers through :mod:`AtlasServer`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modules.Server import AtlasServer, RequestContext
from modules.Server.conversation_routes import (
    ConversationAuthorizationError,
    ConversationNotFoundError,
    ConversationValidationError,
)
from modules.conversation_store import ConversationStoreRepository


@pytest.fixture()
def repository() -> MagicMock:
    repo = MagicMock(spec=ConversationStoreRepository)
    repo.list_conversations_for_tenant.return_value = [
        {
            "id": "conv-1",
            "tenant_id": "tenant-123",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": {},
            "title": None,
            "session_id": None,
        }
    ]
    repo.get_conversation.return_value = {
        "id": "conv-1",
        "tenant_id": "tenant-123",
        "created_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "title": None,
        "session_id": None,
    }
    return repo


@pytest.fixture()
def server(repository: MagicMock) -> AtlasServer:
    return AtlasServer(conversation_repository=repository)


@pytest.fixture()
def request_context() -> RequestContext:
    return RequestContext(tenant_id="tenant-123")


def test_list_conversations_passes_parameters(
    server: AtlasServer, repository: MagicMock, request_context: RequestContext
) -> None:
    payload = server.list_conversations(
        {"limit": "5", "offset": "2", "order": "asc"},
        context=request_context,
    )

    repository.list_conversations_for_tenant.assert_called_once_with(
        "tenant-123", limit=5, offset=2, order="asc"
    )
    assert payload["items"]
    assert payload["count"] == len(payload["items"])
    assert payload["order"] == "asc"
    assert payload["limit"] == 5
    assert payload["offset"] == 2


def test_list_conversations_rejects_invalid_parameters(
    server: AtlasServer, repository: MagicMock, request_context: RequestContext
) -> None:
    with pytest.raises(ConversationValidationError):
        server.list_conversations({"order": "sideways"}, context=request_context)

    repository.list_conversations_for_tenant.assert_not_called()


def test_reset_conversation_requires_existing_record(
    server: AtlasServer, repository: MagicMock, request_context: RequestContext
) -> None:
    repository.get_conversation.return_value = None

    with pytest.raises(ConversationNotFoundError):
        server.reset_conversation("conv-1", context=request_context)

    repository.reset_conversation.assert_not_called()


def test_reset_and_delete_conversation_delegate_to_repository(
    server: AtlasServer, repository: MagicMock, request_context: RequestContext
) -> None:
    reset_response = server.reset_conversation("conv-1", context=request_context)
    repository.reset_conversation.assert_called_once_with(
        "conv-1", tenant_id="tenant-123"
    )
    assert reset_response["status"] == "reset"
    assert reset_response["conversation"]["id"] == "conv-1"

    delete_response = server.delete_conversation("conv-1", context=request_context)
    repository.hard_delete_conversation.assert_called_once_with(
        "conv-1", tenant_id="tenant-123"
    )
    assert delete_response["status"] == "deleted"
    assert delete_response["conversation"]["id"] == "conv-1"


def test_handle_request_dispatches_conversation_routes(
    server: AtlasServer, repository: MagicMock, request_context: RequestContext
) -> None:
    repository.list_conversations_for_tenant.reset_mock()
    repository.reset_conversation.reset_mock()
    repository.hard_delete_conversation.reset_mock()

    listing = server.handle_request(
        "/conversations",
        method="GET",
        query={"order": "desc"},
        context=request_context,
    )
    assert listing["items"]
    repository.list_conversations_for_tenant.assert_called_once_with(
        "tenant-123", limit=None, offset=0, order="desc"
    )

    server.handle_request(
        "/conversations/conv-1/reset",
        method="POST",
        query={},
        context=request_context,
    )
    repository.reset_conversation.assert_called_once_with(
        "conv-1", tenant_id="tenant-123"
    )

    server.handle_request(
        "/conversations/conv-1",
        method="DELETE",
        query={},
        context=request_context,
    )
    repository.hard_delete_conversation.assert_called_once_with(
        "conv-1", tenant_id="tenant-123"
    )


def test_handle_request_requires_context_for_conversations(
    server: AtlasServer,
) -> None:
    with pytest.raises(ConversationAuthorizationError):
        server.handle_request("/conversations", method="GET")
