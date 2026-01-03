import asyncio
import importlib
import os
import sys
import types
from datetime import timezone as _tz
from pathlib import Path
from typing import Mapping
from unittest.mock import Mock

from modules.Tools.Base_Tools.hitl_approval import HITLApprovalTool
from modules.task_store import TaskStatus

hitl_module = importlib.import_module("modules.Tools.Base_Tools.hitl_approval")


def test_hitl_request_creates_review_task(monkeypatch):
    created_record = {
        "id": "task-123",
        "status": TaskStatus.REVIEW.value,
        "title": "HITL Approval",
        "description": "Need approval",
        "metadata": {},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }

    service = Mock()
    service.create_task.return_value = created_record

    events: list[tuple[str, Mapping[str, object]]] = []
    monkeypatch.setattr(
        hitl_module,
        "publish_bus_event",
        lambda name, payload: events.append((name, dict(payload))),
    )

    tool = HITLApprovalTool(task_service=service)
    result = asyncio.run(
        tool.run(
            operation="request",
            tenant_id="tenant-1",
            conversation_id="conv-1",
            reason="Need approval",
            context={"document": "spec"},
            assignees=["alice", "bob", "alice"],
            escalation_policy={"sla_hours": 2},
        )
    )

    service.create_task.assert_called_once()
    _, kwargs = service.create_task.call_args
    assert kwargs["status"] == TaskStatus.REVIEW
    metadata = kwargs["metadata"]
    assert metadata["hitl"]["reason"] == "Need approval"
    assert metadata["hitl"]["assignees"] == ["alice", "bob"]
    assert metadata["hitl"]["escalation"] == {"sla_hours": 2}
    assert events[-1][0] == "hitl.approval.requested"
    assert result["operation"] == "request"
    assert result["task_id"] == "task-123"


def test_hitl_status_polls_task(monkeypatch):
    record = {
        "id": "task-456",
        "status": TaskStatus.REVIEW.value,
        "metadata": {"hitl": {"reason": "Need approval"}},
        "title": "HITL Approval",
        "description": "Need approval",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z",
    }

    service = Mock()
    service.get_task.return_value = record

    events: list[tuple[str, Mapping[str, object]]] = []
    monkeypatch.setattr(
        hitl_module,
        "publish_bus_event",
        lambda name, payload: events.append((name, dict(payload))),
    )

    tool = HITLApprovalTool(task_service=service)
    result = asyncio.run(
        tool.run(
            operation="status",
            tenant_id="tenant-1",
            task_id="task-456",
        )
    )

    service.get_task.assert_called_once_with("task-456", tenant_id="tenant-1")
    assert events[-1][0] == "hitl.approval.status"
    assert result["status"] == TaskStatus.REVIEW.value
    assert result["operation"] == "status"


def test_hitl_resolve_updates_task(monkeypatch):
    snapshot = {
        "id": "task-789",
        "status": TaskStatus.REVIEW.value,
        "metadata": {"hitl": {"reason": "Need approval"}},
        "title": "HITL Approval",
        "description": "Need approval",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T01:00:00Z",
    }
    transitioned = {
        "id": "task-789",
        "status": TaskStatus.DONE.value,
        "metadata": {"hitl": {"reason": "Need approval"}},
        "title": "HITL Approval",
        "description": "Need approval",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T02:00:00Z",
    }

    service = Mock()
    service.get_task.return_value = snapshot
    service.update_task.return_value = snapshot
    service.transition_task.return_value = transitioned

    events: list[tuple[str, Mapping[str, object]]] = []
    monkeypatch.setattr(
        hitl_module,
        "publish_bus_event",
        lambda name, payload: events.append((name, dict(payload))),
    )

    tool = HITLApprovalTool(task_service=service)
    result = asyncio.run(
        tool.run(
            operation="resolve",
            tenant_id="tenant-1",
            task_id="task-789",
            resolution={"decision": "approved"},
            target_status="done",
        )
    )

    service.get_task.assert_called_with("task-789", tenant_id="tenant-1")
    service.update_task.assert_called_once()
    update_metadata = service.update_task.call_args.kwargs["changes"]["metadata"]
    resolution_block = update_metadata["hitl"]["resolution"]
    assert resolution_block["decision"] == "approved"
    assert "resolved_at" in resolution_block
    service.transition_task.assert_called_once()
    assert service.transition_task.call_args.kwargs["target_status"] == TaskStatus.DONE
    assert events[-1][0] == "hitl.approval.resolved"
    assert result["status"] == TaskStatus.DONE.value
    assert result["operation"] == "resolve"


def test_hitl_tool_visible_via_tool_manager(monkeypatch):
    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")

        def _safe_load(_stream):
            return {}

        yaml_stub.safe_load = _safe_load
        sys.modules["yaml"] = yaml_stub

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")

        def _noop(*_args, **_kwargs):
            return None

        dotenv_stub.load_dotenv = _noop
        dotenv_stub.set_key = _noop
        dotenv_stub.find_dotenv = _noop
        sys.modules["dotenv"] = dotenv_stub

    if "pytz" not in sys.modules:
        pytz_stub = types.ModuleType("pytz")

        def _timezone(name: str):
            if name.upper() == "UTC":
                return _tz.utc
            raise ValueError(name)

        pytz_stub.UTC = _tz.utc
        pytz_stub.timezone = _timezone
        sys.modules["pytz"] = pytz_stub

    if "aiohttp" not in sys.modules:
        aiohttp_stub = types.ModuleType("aiohttp")
        sys.modules["aiohttp"] = aiohttp_stub

    aiohttp_mod = sys.modules["aiohttp"]

    class _ClientTimeout:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    class _ClientSession:  # pragma: no cover - simple stub
        def __init__(self, *_, **__):
            pass

    class _ClientError(Exception):
        pass

    class _ClientResponse:  # pragma: no cover - simple stub
        pass

    monkeypatch.setattr(aiohttp_mod, "ClientTimeout", _ClientTimeout, raising=False)
    monkeypatch.setattr(aiohttp_mod, "ClientSession", _ClientSession, raising=False)
    monkeypatch.setattr(aiohttp_mod, "ClientError", _ClientError, raising=False)
    monkeypatch.setattr(aiohttp_mod, "ClientResponse", _ClientResponse, raising=False)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )
    tool_manager._default_function_map_cache = None

    function_map = tool_manager.load_default_function_map(refresh=True)
    assert isinstance(function_map, Mapping)
    assert "hitl.approval" in function_map
