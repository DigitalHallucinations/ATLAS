"""Ensure the task queue integrates cleanly during application startup."""

from __future__ import annotations

import importlib
import sys
import types


def test_get_default_service_uses_lazy_config_manager(monkeypatch):
    """`get_default_task_queue_service()` should initialize without circular imports."""

    jobstore_url = "postgresql://localhost/test-jobstore"

    stub_config_module = types.ModuleType("ATLAS.config")

    class StubConfigManager:
        def __init__(self) -> None:
            self._config = {"task_queue": {"jobstore_url": jobstore_url}}

        def get_config(self, key: str, default=None):
            return self._config.get(key, default)

        def get_job_store_url(self) -> str:
            return jobstore_url

    stub_config_module.ConfigManager = StubConfigManager

    monkeypatch.setitem(sys.modules, "ATLAS.config", stub_config_module)
    try:
        import ATLAS as atlas_package
    except ModuleNotFoundError:  # pragma: no cover - package should exist in test environment
        atlas_package = types.ModuleType("ATLAS")
        monkeypatch.setitem(sys.modules, "ATLAS", atlas_package)

    monkeypatch.setattr(atlas_package, "config", stub_config_module, raising=False)

    sys.modules.pop("modules.Tools.Base_Tools.task_queue", None)
    task_queue = importlib.import_module("modules.Tools.Base_Tools.task_queue")

    class _Executor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

    class _JobStore:
        def __init__(self, url: str):
            self.url = url

    class _Scheduler:
        def __init__(self, *_, **__):
            self.running = False

        def add_listener(self, *_args, **_kwargs):
            return None

        def start(self, paused: bool = False):
            self.running = not paused

        def shutdown(self, wait: bool = False):
            self.running = False

    monkeypatch.setattr(task_queue, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(task_queue, "SQLAlchemyJobStore", _JobStore)
    monkeypatch.setattr(task_queue, "BackgroundScheduler", _Scheduler)
    monkeypatch.setattr(task_queue.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(task_queue, "_DEFAULT_SERVICE", None)
    monkeypatch.setattr(task_queue, "_SERVICE_REGISTRY", {})

    service = task_queue.get_default_task_queue_service()
    try:
        assert service is task_queue.get_default_task_queue_service()
    finally:
        service.shutdown(wait=False)
        task_queue._DEFAULT_SERVICE = None
        task_queue._SERVICE_REGISTRY.clear()
