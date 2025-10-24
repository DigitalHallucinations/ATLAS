"""Unit tests around :mod:`modules.Server.task_routes` pagination behaviour."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

from modules.Server.conversation_routes import RequestContext
from modules.Server.task_routes import TaskRoutes, _decode_cursor, _encode_cursor


def _isoformat(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class _TaskServiceStub:
    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def list_tasks(self, **kwargs):  # type: ignore[override]
        self.calls.append(dict(kwargs))
        return list(self.results)


def test_list_tasks_delegates_pagination_arguments():
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tasks = [
        {
            "id": str(uuid.uuid4()),
            "created_at": _isoformat(base_time - timedelta(minutes=index)),
        }
        for index in range(3)
    ]

    service = _TaskServiceStub(tasks)
    routes = TaskRoutes(service, page_size_limit=10)
    context = RequestContext(tenant_id="tenant-1")

    cursor = _encode_cursor(_isoformat(base_time + timedelta(minutes=5)), str(uuid.uuid4()))

    response = routes.list_tasks({"page_size": 2, "cursor": cursor}, context=context)

    expected_cursor = _decode_cursor(cursor)
    assert service.calls == [
        {
            "tenant_id": context.tenant_id,
            "status": None,
            "owner_id": None,
            "conversation_id": None,
            "limit": 3,
            "cursor": expected_cursor,
        }
    ]

    assert response["items"] == tasks[:2]
    assert response["page"]["count"] == 2
    assert response["page"]["page_size"] == 2
    assert response["page"]["next_cursor"] == _encode_cursor(
        tasks[1]["created_at"], tasks[1]["id"]
    )
