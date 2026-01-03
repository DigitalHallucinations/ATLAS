from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Tuple

if "ATLAS.ToolManager" not in sys.modules:
    tool_manager_stub = types.ModuleType("ATLAS.ToolManager")
    tool_manager_stub.load_function_map_from_current_persona = lambda *_args, **_kwargs: {}
    tool_manager_stub.load_functions_from_json = lambda *_args, **_kwargs: {}
    tool_manager_stub.compute_tool_policy_snapshot = lambda *_args, **_kwargs: {}
    tool_manager_stub._resolve_function_callable = (
        lambda *_args, **_kwargs: lambda *_inner_args, **_inner_kwargs: None
    )
    tool_manager_stub.get_tool_activity_log = lambda *_args, **_kwargs: []
    tool_manager_stub.ToolPolicyDecision = type("ToolPolicyDecision", (), {})
    tool_manager_stub.SandboxedToolRunner = type("SandboxedToolRunner", (), {})
    tool_manager_stub.use_tool = lambda *_args, **_kwargs: None
    tool_manager_stub.call_model_with_new_prompt = lambda *_args, **_kwargs: None
    sys.modules["ATLAS.ToolManager"] = tool_manager_stub

from ATLAS.ATLAS import ATLAS as AtlasApplication


class DummyLogger:
    def __init__(self) -> None:
        self.records: List[Tuple[str, str]] = []

    def warning(self, message: str, *args: Any, **_kwargs: Any) -> None:  # pragma: no cover - helper
        text = message % args if args else message
        self.records.append(("warning", text))

    def error(self, message: str, *args: Any, **_kwargs: Any) -> None:  # pragma: no cover - helper
        text = message % args if args else message
        self.records.append(("error", text))


class SearchServerStub:
    def __init__(self) -> None:
        self.search_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def search_tasks(self, payload: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        snapshot = (dict(payload), dict(context or {}))
        self.search_calls.append(snapshot)
        return {"items": [{"id": "task-1"}], "count": 1}


class ListServerStub(SearchServerStub):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def list_tasks(self, params: Dict[str, Any], *, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        snapshot = (dict(params), dict(context or {}))
        self.list_calls.append(snapshot)
        return {
            "items": [{"id": "task-2"}],
            "page": {"next_cursor": None, "count": 1, "page_size": 1},
        }


def test_atlas_search_tasks_uses_search_route() -> None:
    atlas = AtlasApplication.__new__(AtlasApplication)
    atlas.logger = DummyLogger()
    server = SearchServerStub()
    atlas.server = server
    atlas.tenant_id = "tenant-1"

    result = atlas.search_tasks(
        text="Draft",
        metadata={"team": "blue"},
        status="ready",
        owner_id="owner-123",
        limit=10,
        offset=5,
    )

    assert result == {"items": [{"id": "task-1"}], "count": 1}
    assert server.search_calls
    payload, context = server.search_calls[0]
    assert payload["text"] == "Draft"
    assert payload["metadata"] == {"team": "blue"}
    assert payload["status"] == "ready"
    assert payload["owner_id"] == "owner-123"
    assert payload["limit"] == 10
    assert payload["offset"] == 5
    assert context["tenant_id"] == "tenant-1"


def test_atlas_search_tasks_uses_list_route_when_no_filters() -> None:
    atlas = AtlasApplication.__new__(AtlasApplication)
    atlas.logger = DummyLogger()
    server = ListServerStub()
    atlas.server = server
    atlas.tenant_id = "tenant-2"

    result = atlas.search_tasks(status="draft", conversation_id="conv-1", limit=25, cursor="cursor-1")

    assert "page" in result
    assert result["items"] == [{"id": "task-2"}]
    assert not server.search_calls
    assert server.list_calls
    params, context = server.list_calls[0]
    assert params["status"] == "draft"
    assert params["conversation_id"] == "conv-1"
    assert params["page_size"] == 25
    assert params["cursor"] == "cursor-1"
    assert context["tenant_id"] == "tenant-2"
