import asyncio
import importlib
import json
import os
import socket
import sys
import types
import time
from collections.abc import AsyncIterator as AsyncIteratorABC, Mapping
from pathlib import Path
import shutil
import textwrap

import pytest


@pytest.fixture(autouse=True)
def _stub_tool_event_bus(monkeypatch):
    from modules.Tools import tool_event_system

    def _fake_publish(event_name, payload, *, emit_legacy=True, **kwargs):
        if emit_legacy:
            tool_event_system.event_system.publish(event_name, payload)
        return "test-correlation"

    monkeypatch.setattr(
        tool_event_system,
        "publish_bus_event",
        _fake_publish,
        raising=False,
    )


def _normalize_tool_response_payload(response):
    def _clone(value):
        if isinstance(value, dict):
            return {key: _clone(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_clone(item) for item in value]
        if isinstance(value, tuple):
            return [_clone(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    if isinstance(response, dict):
        return _clone(response)

    if isinstance(response, list):
        normalized = []
        for item in response:
            if isinstance(item, (dict, list, tuple)):
                normalized.append(_normalize_tool_response_payload(item))
            else:
                normalized.append(
                    {"type": "output_text", "text": "" if item is None else str(item)}
                )
        return normalized

    if isinstance(response, tuple):
        return _normalize_tool_response_payload(list(response))

    return {"type": "output_text", "text": "" if response is None else str(response)}


def _resolve_callable(entry):
    if isinstance(entry, dict) and "callable" in entry:
        return entry["callable"]
    return entry


class _DummyConfigManager:
    def __init__(self, *, log_full_payloads: bool = False, summary_length: int = 256):
        self._config = {
            "tool_logging": {
                "log_full_payloads": log_full_payloads,
                "payload_summary_length": summary_length,
            }
        }

    def get_config(self, key, default=None):
        return self._config.get(key, default)


def _clear_provider_env(monkeypatch):
    for key in [
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROK_API_KEY",
        "XI_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def _ensure_jsonschema(monkeypatch):
    if "jsonschema" in sys.modules:
        return

    jsonschema_stub = types.ModuleType("jsonschema")

    class _DummyValidationError(Exception):
        pass

    class _DummyValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def validate(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return None

        def iter_errors(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return iter(())

    jsonschema_stub.ValidationError = _DummyValidationError
    jsonschema_stub.Draft7Validator = _DummyValidator
    monkeypatch.setitem(sys.modules, "jsonschema", jsonschema_stub)


def _ensure_aiohttp(monkeypatch):
    existing = sys.modules.get("aiohttp")

    class _DummyResponse:
        status = 200

        async def json(self):  # pragma: no cover - helper stub
            return {}

        async def __aenter__(self):  # pragma: no cover - helper stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - helper stub
            return False

    class _DummyClientSession:
        async def __aenter__(self):  # pragma: no cover - helper stub
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - helper stub
            return False

        async def get(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return _DummyResponse()

    class _DummyClientTimeout:
        def __init__(self, *args, **kwargs):  # pragma: no cover - helper stub
            self.total = kwargs.get("total")

    if existing is not None:
        if not hasattr(existing, "ClientSession"):
            existing.ClientSession = _DummyClientSession
        if not hasattr(existing, "ClientTimeout"):
            existing.ClientTimeout = _DummyClientTimeout
        return

    aiohttp_stub = types.ModuleType("aiohttp")

    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = _DummyClientTimeout
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)


def _ensure_yaml(monkeypatch):
    _ensure_jsonschema(monkeypatch)
    _ensure_aiohttp(monkeypatch)
    if "yaml" in sys.modules:
        return

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda stream: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)



def _ensure_dotenv(monkeypatch):
    if "dotenv" in sys.modules:
        return

    dotenv_stub = types.ModuleType("dotenv")

    def _load_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
        return True

    def _set_key(*_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def _find_dotenv(*_args, **_kwargs):  # pragma: no cover - stub helper
        return ""

    dotenv_stub.load_dotenv = _load_dotenv
    dotenv_stub.set_key = _set_key
    dotenv_stub.find_dotenv = _find_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

def _ensure_pytz(monkeypatch):
    if "pytz" in sys.modules:
        return

    import datetime

    pytz_stub = types.SimpleNamespace(
        timezone=lambda *_args, **_kwargs: datetime.timezone.utc,
        utc=datetime.timezone.utc,
    )
    monkeypatch.setitem(sys.modules, "pytz", pytz_stub)


def _ensure_sqlalchemy(monkeypatch):
    if "sqlalchemy" in sys.modules:
        return

    sqlalchemy_stub = types.ModuleType("sqlalchemy")

    class _SQLCallable:
        def __init__(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            pass

        def __call__(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

    sqlalchemy_stub.create_engine = lambda *_args, **_kwargs: None
    sqlalchemy_stub.Column = _SQLCallable
    sqlalchemy_stub.DateTime = _SQLCallable
    sqlalchemy_stub.Enum = _SQLCallable
    sqlalchemy_stub.ForeignKey = _SQLCallable
    sqlalchemy_stub.Index = _SQLCallable
    sqlalchemy_stub.Integer = _SQLCallable
    sqlalchemy_stub.Boolean = _SQLCallable
    sqlalchemy_stub.Float = _SQLCallable
    sqlalchemy_stub.String = _SQLCallable
    sqlalchemy_stub.Text = _SQLCallable
    sqlalchemy_stub.UniqueConstraint = _SQLCallable
    sqlalchemy_stub.inspect = lambda *_args, **_kwargs: types.SimpleNamespace(
        has_table=lambda *_a, **_k: False
    )
    sqlalchemy_stub.and_ = lambda *_args, **_kwargs: None
    sqlalchemy_stub.or_ = lambda *_args, **_kwargs: None
    sqlalchemy_stub.select = lambda *_args, **_kwargs: None
    sqlalchemy_stub.text = lambda *_args, **_kwargs: None
    sqlalchemy_stub.delete = lambda *_args, **_kwargs: None

    class _FuncProxy:
        def __getattr__(self, _name):  # pragma: no cover - helper stub
            return _SQLCallable()

    sqlalchemy_stub.func = _FuncProxy()
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_stub)

    dialects_stub = types.ModuleType("sqlalchemy.dialects")
    postgres_stub = types.ModuleType("sqlalchemy.dialects.postgresql")
    postgres_stub.ARRAY = _SQLCallable
    postgres_stub.JSONB = _SQLCallable
    postgres_stub.UUID = _SQLCallable
    postgres_stub.TSVECTOR = _SQLCallable
    monkeypatch.setitem(sys.modules, "sqlalchemy.dialects", dialects_stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy.dialects.postgresql", postgres_stub)

    exc_stub = types.ModuleType("sqlalchemy.exc")
    exc_stub.IntegrityError = type("IntegrityError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "sqlalchemy.exc", exc_stub)

    if "modules.task_store" not in sys.modules:
        task_store_stub = types.ModuleType("modules.task_store")

        class _DummyRepository:
            def __init__(self, *_args, **_kwargs):  # pragma: no cover - helper stub
                pass

            def create_schema(self) -> None:  # pragma: no cover - helper stub
                return None

        class _DummyService:
            def __init__(self, *_args, **_kwargs):  # pragma: no cover - helper stub
                pass

        def _ensure_schema(*_args, **_kwargs):  # pragma: no cover - helper stub
            return None

        task_store_stub.TaskStoreRepository = _DummyRepository
        task_store_stub.TaskService = _DummyService
        task_store_stub.ensure_task_schema = _ensure_schema
        monkeypatch.setitem(sys.modules, "modules.task_store", task_store_stub)

    engine_stub = types.ModuleType("sqlalchemy.engine")
    engine_stub.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", engine_stub)

    orm_stub = types.ModuleType("sqlalchemy.orm")

    class _Sessionmaker:
        def __call__(self, *_args, **_kwargs):  # pragma: no cover - helper stub
            return None

    orm_stub.sessionmaker = _Sessionmaker
    orm_stub.Session = object
    orm_stub.joinedload = lambda *_args, **_kwargs: None
    orm_stub.relationship = lambda *_args, **_kwargs: None
    orm_stub.declarative_base = lambda *_args, **_kwargs: type("Base", (), {})
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm", orm_stub)


def test_default_function_map_vector_store_entries_dispatch(monkeypatch):
    import asyncio
    import importlib

    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    _ensure_sqlalchemy(monkeypatch)

    from modules.Tools.Base_Tools import vector_store as vector_store_module
    import modules.Tools.tool_maps.maps as maps_module

    call_log = []

    async def fake_upsert(*, namespace, vectors, **kwargs):
        call_log.append(("upsert", namespace, kwargs.get("config_manager"), vectors))
        return {"namespace": namespace, "ids": ["v-1"], "upserted_count": len(vectors)}

    async def fake_query(
        *,
        namespace,
        query,
        top_k=5,
        filter=None,
        include_values=False,
        **kwargs,
    ):
        call_log.append(("query", namespace, kwargs.get("config_manager"), top_k, include_values))
        return {
            "namespace": namespace,
            "top_k": int(top_k),
            "matches": [],
        }

    async def fake_delete(*, namespace, **kwargs):
        call_log.append(("delete", namespace, kwargs.get("config_manager")))
        return {"namespace": namespace, "deleted": True, "removed_ids": []}

    monkeypatch.setattr(vector_store_module, "upsert_vectors", fake_upsert)
    monkeypatch.setattr(vector_store_module, "query_vectors", fake_query)
    monkeypatch.setattr(vector_store_module, "delete_namespace", fake_delete)

    reloaded_maps = importlib.reload(maps_module)

    async def _exercise():
        function_map = reloaded_maps.function_map

        upsert_result = await function_map["upsert_vectors"](
            namespace="example",
            vectors=[{"id": "vec-1", "values": [0.1, 0.2]}],
        )
        assert upsert_result["namespace"] == "example"
        assert call_log[0][0] == "upsert"
        assert call_log[0][1] == "example"
        assert call_log[0][2] is reloaded_maps._config_manager

        query_result = await function_map["query_vectors"](
            namespace="example",
            query=[0.1, 0.2],
            top_k=2,
            include_values=True,
        )
        assert query_result == {"namespace": "example", "top_k": 2, "matches": []}
        assert call_log[1][0] == "query"
        assert call_log[1][1] == "example"
        assert call_log[1][2] is reloaded_maps._config_manager
        assert call_log[1][3] == 2
        assert call_log[1][4] is True

        delete_result = await function_map["delete_namespace"](namespace="example")
        assert delete_result == {"namespace": "example", "deleted": True, "removed_ids": []}
        assert call_log[2][0] == "delete"
        assert call_log[2][1] == "example"
        assert call_log[2][2] is reloaded_maps._config_manager

    try:
        asyncio.run(_exercise())
    finally:
        monkeypatch.undo()
        importlib.reload(maps_module)


def test_default_function_map_task_queue_entries_dispatch(monkeypatch):
    import importlib

    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    _ensure_sqlalchemy(monkeypatch)

    from modules.Tools.Base_Tools import task_queue as task_queue_module
    import modules.Tools.tool_maps.maps as maps_module

    call_log = []

    def _make_stub(name, result):
        def _stub(*args, **kwargs):
            call_log.append((name, args, kwargs))
            return result

        return _stub

    monkeypatch.setattr(
        task_queue_module,
        "enqueue_task",
        _make_stub("enqueue", {"status": "queued", "job_id": "enqueue-1"}),
    )
    monkeypatch.setattr(
        task_queue_module,
        "schedule_cron_task",
        _make_stub("schedule", {"status": "scheduled", "job_id": "schedule-1"}),
    )
    monkeypatch.setattr(
        task_queue_module,
        "cancel_task",
        _make_stub("cancel", {"status": "cancelled", "job_id": "cancel-1"}),
    )
    monkeypatch.setattr(
        task_queue_module,
        "get_task_status",
        _make_stub("status", {"status": "succeeded", "job_id": "status-1"}),
    )

    try:
        reloaded_maps = importlib.reload(maps_module)
        function_map = reloaded_maps.function_map

        enqueue_payload = {"value": 1}
        enqueue_result = function_map["task_queue_enqueue"](
            name="example", payload=enqueue_payload, delay_seconds=5
        )
        assert enqueue_result == {"status": "queued", "job_id": "enqueue-1"}
        enqueue_entry = call_log[0]
        assert enqueue_entry[0] == "enqueue"
        assert enqueue_entry[1] == ()
        enqueue_kwargs = enqueue_entry[2]
        assert enqueue_kwargs["name"] == "example"
        assert enqueue_kwargs["payload"] is enqueue_payload
        assert enqueue_kwargs["delay_seconds"] == 5
        assert enqueue_kwargs["config_manager"] is reloaded_maps._config_manager

        cron_payload = {"value": 2}
        schedule_result = function_map["task_queue_schedule"](
            name="cron", cron_schedule="* * * * *", payload=cron_payload
        )
        assert schedule_result == {"status": "scheduled", "job_id": "schedule-1"}
        schedule_entry = call_log[1]
        assert schedule_entry[0] == "schedule"
        assert schedule_entry[1] == ()
        schedule_kwargs = schedule_entry[2]
        assert schedule_kwargs["name"] == "cron"
        assert schedule_kwargs["cron_schedule"] == "* * * * *"
        assert schedule_kwargs["payload"] is cron_payload
        assert schedule_kwargs["config_manager"] is reloaded_maps._config_manager

        cancel_result = function_map["task_queue_cancel"]("job-123")
        assert cancel_result == {"status": "cancelled", "job_id": "cancel-1"}
        cancel_entry = call_log[2]
        assert cancel_entry[0] == "cancel"
        assert cancel_entry[1] == ("job-123",)
        cancel_kwargs = cancel_entry[2]
        assert cancel_kwargs["config_manager"] is reloaded_maps._config_manager

        status_result = function_map["task_queue_status"]("job-123")
        assert status_result == {"status": "succeeded", "job_id": "status-1"}
        status_entry = call_log[3]
        assert status_entry[0] == "status"
        assert status_entry[1] == ("job-123",)
        status_kwargs = status_entry[2]
        assert status_kwargs["config_manager"] is reloaded_maps._config_manager
    finally:
        monkeypatch.undo()
        importlib.reload(maps_module)


def test_persona_toolbox_manifest_includes_required_metadata(monkeypatch):
    """Persona toolbox manifests should include the extended metadata fields."""

    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    _ensure_sqlalchemy(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "ResumeGenius"
    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_payload_cache.pop(persona_name, None)

    config_manager = tool_manager.ConfigManager()

    persona_payload = {
        "name": persona_name,
        "allowed_tools": [
            "google_search",
            "get_current_info",
            "policy_reference",
            "ats_scoring_service",
        ],
    }

    functions = tool_manager.load_functions_from_json(
        persona_payload, refresh=True, config_manager=config_manager
    )

    assert isinstance(functions, list)
    assert functions, "expected persona toolbox manifest to load"

    google_entry = next(entry for entry in functions if entry["name"] == "google_search")
    assert google_entry["idempotency_key"] is False
    assert google_entry["capabilities"] == ["web_search", "knowledge_lookup"]
    assert google_entry["providers"] == [
        {
            "name": "google_cse",
            "priority": 0,
            "health_check_interval": 300,
            "config": {
                "api_key_env": "GOOGLE_API_KEY",
                "api_key_config": "GOOGLE_API_KEY",
                "cse_id_env": "GOOGLE_CSE_ID",
                "cse_id_config": "GOOGLE_CSE_ID",
            },
        },
        {
            "name": "serpapi",
            "priority": 10,
            "health_check_interval": 300,
        },
    ]

    info_entry = next(entry for entry in functions if entry["name"] == "get_current_info")
    assert info_entry["idempotency_key"] is False
    assert info_entry["capabilities"] == ["time_information", "date_information"]
    assert info_entry["cost_per_call"] == 0.0

    shared_map = tool_manager.load_default_function_map(
        refresh=True, config_manager=config_manager
    )
    assert "policy_reference" in shared_map
    policy_entry = shared_map["policy_reference"]
    assert isinstance(policy_entry, dict)
    metadata = policy_entry.get("metadata")
    assert metadata["idempotency_key"]["required"] is True

    assert "terminal_command" in shared_map
    terminal_entry = shared_map["terminal_command"]
    assert isinstance(terminal_entry, dict)
    terminal_metadata = terminal_entry.get("metadata")
    assert terminal_metadata["safety_level"] == "high"
    assert terminal_metadata["requires_consent"] is True
    assert terminal_metadata["requires_flags"] == {
        "read": ["type.personal_assistant.terminal_read_enabled"],
        "write": [
            "type.personal_assistant.terminal_read_enabled",
            "type.personal_assistant.terminal_write_enabled",
        ],
    }

    assert "geocode_location" not in shared_map

    weather_payload = {
        "name": "WeatherGenius",
        "allowed_tools": [
            "google_search",
            "get_current_info",
            "policy_reference",
            "get_current_weather",
            "get_historical_weather",
            "get_daily_weather_summary",
            "weather_alert_feed",
            "geocode_location",
        ],
    }

    weather_functions = tool_manager.load_functions_from_json(
        weather_payload, refresh=True, config_manager=config_manager
    )

    geocode_entry = next(
        entry for entry in weather_functions if entry["name"] == "geocode_location"
    )
    assert geocode_entry["capabilities"] == ["geolocation", "mapping"]
    assert geocode_entry["auth"]["required"] is True

    assert "get_current_location" in shared_map
    current_location_entry = shared_map["get_current_location"]
    assert isinstance(current_location_entry, dict)
    current_location_metadata = current_location_entry.get("metadata")
    assert current_location_metadata["capabilities"] == [
        "geolocation",
        "context_awareness",
    ]
    assert current_location_metadata["auth"]["required"] is False

    assert "debian12_calendar" in shared_map
    debian_calendar_entry = shared_map["debian12_calendar"]
    assert isinstance(debian_calendar_entry, dict)
    debian_calendar_metadata = debian_calendar_entry.get("metadata")
    assert debian_calendar_metadata["capabilities"] == [
        "calendar_read",
        "calendar_write",
    ]
    assert debian_calendar_metadata["providers"][0]["name"] == "debian12_local"


def test_resume_genius_manifest_includes_ats_scoring(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    _ensure_sqlalchemy(monkeypatch)

    if "modules.task_store" not in sys.modules:
        task_store_stub = types.ModuleType("modules.task_store")

        class _DummyRepository:
            def __init__(self, *_args, **_kwargs):
                pass

            def create_schema(self) -> None:
                return None

        class _DummyService:
            def __init__(self, *_args, **_kwargs):
                pass

        task_store_stub.TaskStoreRepository = _DummyRepository
        task_store_stub.TaskService = _DummyService
        monkeypatch.setitem(sys.modules, "modules.task_store", task_store_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "ResumeGenius"
    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_payload_cache.pop(persona_name, None)

    config_manager = tool_manager.ConfigManager()
    persona_payload = {
        "name": persona_name,
        "allowed_tools": [
            "google_search",
            "get_current_info",
            "policy_reference",
            "ats_scoring_service",
        ],
    }
    functions = tool_manager.load_functions_from_json(
        persona_payload, refresh=True, config_manager=config_manager
    )

    names = [entry["name"] for entry in functions]
    assert "ats_scoring_service" in names, names
    ats_entry = next(entry for entry in functions if entry["name"] == "ats_scoring_service")
    assert ats_entry["capabilities"] == ["resume_analysis", "ats_compliance"]
    assert ats_entry["auth"]["required"] is True
    assert ats_entry["default_timeout"] == 20

    properties = ats_entry["parameters"]["properties"]
    assert {"resume_text", "job_description"}.issubset(properties)
    assert "ATS" in ats_entry["description"]


def test_shared_terminal_command_manifest_entry(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    persona_name = "TerminalTester"
    toolbox_root = Path("modules") / "Personas" / persona_name / "Toolbox"
    toolbox_root.mkdir(parents=True, exist_ok=True)
    functions_path = toolbox_root / "functions.json"
    functions_path.write_text("[]", encoding="utf-8")

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    try:
        tool_manager._function_payload_cache.pop(persona_name, None)
        tool_manager._function_map_cache.pop(persona_name, None)

        functions = tool_manager.load_functions_from_json(
            {"name": persona_name, "allowed_tools": ["terminal_command"]},
            refresh=True,
        )

        assert isinstance(functions, list)
        assert len(functions) == 1
        terminal_entry = functions[0]
        assert terminal_entry["name"] == "terminal_command"
        assert terminal_entry["safety_level"] == "high"
        assert terminal_entry["requires_consent"] is True
        assert terminal_entry["requires_flags"] == {
            "read": ["type.personal_assistant.terminal_read_enabled"],
            "write": [
                "type.personal_assistant.terminal_read_enabled",
                "type.personal_assistant.terminal_write_enabled",
            ],
        }
        assert terminal_entry["parameters"]["required"] == ["command"]
    finally:
        tool_manager._function_payload_cache.pop(persona_name, None)
        tool_manager._function_map_cache.pop(persona_name, None)
        shutil.rmtree(toolbox_root.parent, ignore_errors=True)


def test_evaluate_tool_policy_blocks_calendar_write_without_flag(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    metadata = {
        "requires_flags": {
            "create": [
                "type.personal_assistant.access_to_calendar",
                "type.personal_assistant.calendar_write_enabled",
            ]
        }
    }
    persona = {
        "name": "Atlas",
        "type": {
            "personal_assistant": {
                "access_to_calendar": "True",
                "calendar_write_enabled": "False",
            }
        },
    }

    decision = tool_manager._evaluate_tool_policy(
        function_name="debian12_calendar",
        metadata=metadata,
        current_persona=persona,
        conversation_manager=None,
        conversation_id=None,
        tool_arguments={"operation": "create"},
    )

    assert decision.allowed is False
    assert "calendar_write_enabled" in (decision.reason or "")
    assert decision.denied_operations["create"] == (
        "type.personal_assistant.calendar_write_enabled",
    )


def test_evaluate_tool_policy_allows_calendar_write_with_flag(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    metadata = {
        "requires_flags": {
            "create": [
                "type.personal_assistant.access_to_calendar",
                "type.personal_assistant.calendar_write_enabled",
            ]
        }
    }
    persona = {
        "name": "Atlas",
        "type": {
            "personal_assistant": {
                "access_to_calendar": "True",
                "calendar_write_enabled": "True",
            }
        },
    }

    decision = tool_manager._evaluate_tool_policy(
        function_name="debian12_calendar",
        metadata=metadata,
        current_persona=persona,
        conversation_manager=None,
        conversation_id=None,
        tool_arguments={"operation": "create"},
    )

    assert decision.allowed is True
    assert dict(decision.denied_operations) == {}
    assert not decision.reason


def test_evaluate_tool_policy_blocks_terminal_without_read_flag(monkeypatch):
    _ensure_jsonschema(monkeypatch)
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    metadata = {
        "requires_flags": {
            "read": ["type.personal_assistant.terminal_read_enabled"],
            "write": [
                "type.personal_assistant.terminal_read_enabled",
                "type.personal_assistant.terminal_write_enabled",
            ],
        }
    }
    persona = {
        "name": "Atlas",
        "type": {
            "personal_assistant": {
                "terminal_read_enabled": "False",
                "terminal_write_enabled": "False",
            }
        },
    }

    decision = tool_manager._evaluate_tool_policy(
        function_name="terminal_command",
        metadata=metadata,
        current_persona=persona,
        conversation_manager=None,
        conversation_id=None,
    )

    assert decision.allowed is False
    assert "terminal_read_enabled" in (decision.reason or "")
    assert decision.denied_operations["read"] == (
        "type.personal_assistant.terminal_read_enabled",
    )


def test_evaluate_tool_policy_allows_terminal_read_only(monkeypatch):
    _ensure_jsonschema(monkeypatch)
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    metadata = {
        "requires_flags": {
            "read": ["type.personal_assistant.terminal_read_enabled"],
            "write": [
                "type.personal_assistant.terminal_read_enabled",
                "type.personal_assistant.terminal_write_enabled",
            ],
        }
    }
    persona = {
        "name": "Atlas",
        "type": {
            "personal_assistant": {
                "terminal_read_enabled": "True",
                "terminal_write_enabled": "False",
            }
        },
    }

    decision = tool_manager._evaluate_tool_policy(
        function_name="terminal_command",
        metadata=metadata,
        current_persona=persona,
        conversation_manager=None,
        conversation_id=None,
    )

    assert decision.allowed is True
    assert decision.denied_operations["write"] == (
        "type.personal_assistant.terminal_write_enabled",
    )
    assert "terminal_write_enabled" in (decision.reason or "")


def test_compute_snapshot_disables_terminal_without_read(monkeypatch):
    _ensure_jsonschema(monkeypatch)
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    geocode_stub = types.ModuleType("modules.Tools.location_services.geocode")
    geocode_stub.geocode_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.geocode", geocode_stub
    )
    ip_api_stub = types.ModuleType("modules.Tools.location_services.ip_api")
    ip_api_stub.get_current_location = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "modules.Tools.location_services.ip_api", ip_api_stub
    )

    class _DummyResponse:
        status = 200

        async def json(self):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return _DummyResponse()

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientSession = _DummyClientSession
    aiohttp_stub.ClientTimeout = lambda *args, **kwargs: types.SimpleNamespace(
        total=kwargs.get("total")
    )
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_stub)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    function_map = {
        "terminal_command": {
            "metadata": {
                "requires_flags": {
                    "read": ["type.personal_assistant.terminal_read_enabled"],
                    "write": [
                        "type.personal_assistant.terminal_read_enabled",
                        "type.personal_assistant.terminal_write_enabled",
                    ],
                }
            }
        }
    }
    persona = {
        "name": "Atlas",
        "type": {
            "personal_assistant": {
                "terminal_read_enabled": "False",
                "terminal_write_enabled": "False",
            }
        },
    }

    snapshot = tool_manager.compute_tool_policy_snapshot(
        function_map,
        current_persona=persona,
    )

    decision = snapshot["terminal_command"]
    assert decision.allowed is False
    assert decision.denied_operations["read"] == (
        "type.personal_assistant.terminal_read_enabled",
    )
    assert "terminal_read_enabled" in (decision.reason or "")

def test_tool_manager_import_without_credentials(monkeypatch):
    """Importing ToolManager should not require configuration credentials."""

    _clear_provider_env(monkeypatch)
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    sys.modules.pop("ATLAS.ToolManager", None)

    module = importlib.import_module("ATLAS.ToolManager")

    assert hasattr(module, "use_tool"), "ToolManager module should expose use_tool"


def test_use_tool_prefers_supplied_config_manager(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    class DummyConfigManager:
        def __init__(self):
            self.provider_manager = DummyProviderManager()

    conversation_history = DummyConversationHistory()

    dummy_config = DummyConfigManager()

    async def echo_tool(value, context=None):
        return value

    message = {
        "function_call": {
            "id": "call-echo",
            "name": "echo_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    function_map = {"echo_tool": echo_tool}
    tool_manager._tool_activity_log.clear()

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=dummy_config.provider_manager,
            config_manager=dummy_config,
        )

        assert response == "model-output"
        assert (
            dummy_config.provider_manager.generate_calls
        ), "Supplied config manager should be used"
        payload = dummy_config.provider_manager.generate_calls[0]
        assert payload["conversation_manager"] is conversation_history
        assert payload["conversation_id"] == "conversation"
        assert payload["user"] == "user"
        assert payload["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "ping"},
                "tool_call_id": "call-echo",
            },
        ]
        recorded_message = conversation_history.messages[-1]
        assert recorded_message["content"] == [
            {"type": "output_text", "text": "model-output"}
        ]
        assert conversation_history.responses[-1]["tool_call_id"] == "call-echo"
        log_entries = tool_manager.get_tool_activity_log()
        assert log_entries[-1]["tool_call_id"] == "call-echo"

    asyncio.run(run_test())


def test_use_tool_blocks_high_risk_tool_without_consent(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()

    class ConsentConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []
            self.requests = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)
            return entry

        def get_history(self, user, conversation_id):
            return list(self.history)

        def has_tool_consent(self, conversation_id, tool_name):
            return False

        def request_tool_consent(self, conversation_id, tool_name, metadata=None):
            self.requests.append(
                {
                    "conversation_id": conversation_id,
                    "tool_name": tool_name,
                    "metadata": dict(metadata or {}),
                }
            )
            return False

    class DummyProviderManager:
        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **_kwargs):
            raise AssertionError("model should not run when policy blocks the tool")

    conversation_history = ConsentConversationHistory()
    provider = DummyProviderManager()

    executions = []

    async def restricted_tool():
        executions.append(True)
        return "blocked"

    message = {
        "function_call": {
            "id": "call-restricted",
            "name": "restricted_tool",
            "arguments": json.dumps({}),
        }
    }

    function_map = {
        "restricted_tool": {
            "callable": restricted_tool,
            "metadata": {"safety_level": "high", "requires_consent": True},
        }
    }

    async def run_test():
        with pytest.raises(tool_manager.ToolExecutionError) as exc_info:
            await tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message=message,
                conversation_history=conversation_history,
                function_map=function_map,
                functions=None,
                current_persona={"name": "CodeGenius"},
                temperature_var=0.0,
                top_p_var=1.0,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider,
                config_manager=None,
            )

        error = exc_info.value
        assert error.error_type == "tool_policy_violation"
        assert "requires approval" in str(error)
        assert executions == []
        assert conversation_history.requests, "Consent should be requested"
        recorded_entry = conversation_history.messages[-1]
        metadata = recorded_entry.get("metadata", {})
        assert metadata.get("error_type") == "tool_policy_violation"

    asyncio.run(run_test())


def test_use_tool_allows_high_risk_tool_with_consent_and_sandbox(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()

    class ApprovingConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []
            self.requests = []
            self._consent = False

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)
            return entry

        def get_history(self, user, conversation_id):
            return list(self.history)

        def has_tool_consent(self, conversation_id, tool_name):
            return self._consent

        def request_tool_consent(self, conversation_id, tool_name, metadata=None):
            self.requests.append(
                {
                    "conversation_id": conversation_id,
                    "tool_name": tool_name,
                    "metadata": dict(metadata or {}),
                }
            )
            self._consent = True
            return True

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    class ToolSafetyConfig:
        def __init__(self):
            self.provider_manager = DummyProviderManager()
            self._config = {
                "tool_safety": {"network_allowlist": ["localhost"]},
                "tool_defaults": {"timeout_seconds": 30},
            }

        def get_config(self, key, default=None):
            return self._config.get(key, default)

    config_manager = ToolSafetyConfig()
    provider = config_manager.provider_manager
    conversation_history = ApprovingConversationHistory()

    connection_calls = []

    class DummyConnection:
        def __init__(self, address):
            self.address = address

    def fake_create_connection(address, *args, **kwargs):
        connection_calls.append(address)
        return DummyConnection(address)

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(tool_manager.socket, "create_connection", fake_create_connection)

    async def sandboxed_tool(host, context=None):
        flag = os.environ.get("ATLAS_SANDBOX_ACTIVE")
        blocked_message = None
        try:
            socket.create_connection(("blocked.example.com", 443))
        except Exception as exc:
            blocked_message = str(exc)
        allowed = socket.create_connection((host, 80))
        return {
            "flag": flag,
            "blocked": blocked_message,
            "allowed": allowed.address,
        }

    message = {
        "function_call": {
            "id": "call-sandbox",
            "name": "sandboxed_tool",
            "arguments": json.dumps({"host": "localhost"}),
        }
    }

    function_map = {
        "sandboxed_tool": {
            "callable": sandboxed_tool,
            "metadata": {"safety_level": "high", "requires_consent": True},
        }
    }

    async def run_test():
        result = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona={"name": "CodeGenius"},
            temperature_var=0.2,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=config_manager,
        )

        assert result == "model-output"
        assert conversation_history.requests, "Consent request should be recorded"
        tool_entry = conversation_history.responses[-1]
        payload = tool_entry["content"]
        assert payload["flag"] == "1"
        assert "blocked.example.com" in payload["blocked"]
        assert payload["allowed"] == ["localhost", 80]
        assert connection_calls == [("localhost", 80)]
        assert provider.generate_calls, "Model response should be generated"

    asyncio.run(run_test())


def test_use_tool_records_structured_follow_up(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return {
                "content": [
                    {"type": "output_text", "text": "Here is data:"},
                    {"type": "output_json", "json": {"value": 1}},
                ],
                "audio": {"voice": "tester"},
            }

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()

    async def echo_tool(value, context=None):
        return value

    message = {
        "function_call": {
            "id": "call-echo",
            "name": "echo_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"echo_tool": echo_tool},
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
        )

        assert response == "Here is data:"
        assert provider.generate_calls
        stored_message = conversation_history.messages[-1]
        assert stored_message["content"] == [
            {"type": "output_text", "text": "Here is data:"},
            {"type": "output_json", "json": {"value": 1}},
        ]
        assert stored_message.get("audio") == {"voice": "tester"}

    asyncio.run(run_test())


def test_use_tool_retries_idempotent_tool(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider_manager = DummyProviderManager()

    attempts = []
    observed_keys = []

    async def flaky_tool(value, context=None):
        attempts.append(value)
        assert context is not None
        observed_keys.append(context.get("idempotency_key"))
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return {"value": value}

    message = {
        "function_call": {
            "id": "call-idempotent",
            "name": "flaky_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    function_map = {
        "flaky_tool": {
            "callable": flaky_tool,
            "metadata": {"side_effects": "write", "idempotency_key": True},
        }
    }

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(tool_manager.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(tool_manager.random, "uniform", lambda *_: 0.0)

    async def run_test():
        result = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider_manager,
            config_manager=None,
        )

        assert result == "model-output"

    asyncio.run(run_test())

    assert attempts == ["ping", "ping"]
    assert len(set(observed_keys)) == 1
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 0.5
    assert provider_manager.calls


def test_use_tool_does_not_retry_non_idempotent_tool(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.messages = []

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    conversation_history = DummyConversationHistory()

    class DummyProviderManager:
        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **_kwargs):  # pragma: no cover - unused
            return "unused"

    provider_manager = DummyProviderManager()

    attempts = []

    async def failing_tool(value, context=None):
        attempts.append(context.get("idempotency_key") if context else None)
        raise RuntimeError("persistent failure")

    message = {
        "function_call": {
            "id": "call-non-idempotent",
            "name": "failing_tool",
            "arguments": json.dumps({"value": "ping"}),
        }
    }

    function_map = {
        "failing_tool": {
            "callable": failing_tool,
            "metadata": {"side_effects": "write", "idempotency_key": False},
        }
    }

    async def fail_sleep(_):  # pragma: no cover - defensive
        raise AssertionError("sleep should not be invoked for non-idempotent tools")

    monkeypatch.setattr(tool_manager.asyncio, "sleep", fail_sleep)

    with pytest.raises(tool_manager.ToolExecutionError):
        asyncio.run(
            tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message=message,
                conversation_history=conversation_history,
                function_map=function_map,
                functions=None,
                current_persona=None,
                temperature_var=0.5,
                top_p_var=0.9,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider_manager,
                config_manager=None,
            )
        )

    assert len(attempts) == 1
def test_use_tool_records_structured_error(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.messages = []

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            return entry

        def add_response(self, *args, **kwargs):  # pragma: no cover - safety guard
            raise AssertionError("add_response should not be called during error handling")

    conversation_history = DummyConversationHistory()
    provider_manager = types.SimpleNamespace(get_current_model=lambda: "dummy-model")

    async def run_test():
        with pytest.raises(tool_manager.ToolExecutionError) as excinfo:
            await tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message={
                    "function_call": {
                        "id": "call-error",
                        "name": "echo_tool",
                        "arguments": "{not-json}",
                    }
                },
                conversation_history=conversation_history,
                function_map={"echo_tool": lambda value, context=None: value},
                functions=None,
                current_persona=None,
                temperature_var=0.0,
                top_p_var=1.0,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider_manager,
                config_manager=None,
            )

        error = excinfo.value
        assert isinstance(error, tool_manager.ToolExecutionError)
        assert error.tool_call_id == "call-error"
        assert error.function_name == "echo_tool"
        assert error.error_type == "invalid_arguments"
        assert conversation_history.messages, "Error should be recorded in history"
        recorded_entry = conversation_history.messages[-1]
        assert error.entry == recorded_entry
        assert recorded_entry["role"] == "tool"
        assert recorded_entry["tool_call_id"] == "call-error"
        assert recorded_entry["content"][0]["type"] == "output_text"
        assert "Invalid JSON in function arguments" in recorded_entry["content"][0]["text"]
        metadata = recorded_entry["metadata"]
        assert metadata["status"] == "error"
        assert metadata["name"] == "echo_tool"
        assert metadata["error_type"] == "invalid_arguments"

    asyncio.run(run_test())


def test_use_tool_handles_multiple_tool_calls(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()
    tool_manager._tool_activity_log.clear()

    call_sequence = []

    async def async_tool(value):
        call_sequence.append(("async_tool", value))
        await asyncio.sleep(0)
        return f"async:{value}"

    def sync_tool(value):
        call_sequence.append(("sync_tool", value))
        return f"sync:{value}"

    message = {
        "tool_calls": [
            {
                "id": "call-async",
                "function": {"name": "async_tool", "arguments": json.dumps({"value": "one"})},
            },
            {
                "id": "call-sync",
                "function": {"name": "sync_tool", "arguments": {"value": "two"}},
            },
        ]
    }

    function_map = {"async_tool": async_tool, "sync_tool": sync_tool}

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
        )

        assert response == "model-output"
        assert call_sequence == [("async_tool", "one"), ("sync_tool", "two")]
        assert len(conversation_history.responses) == 2
        assert conversation_history.responses[0]["content"]["text"] == "async:one"
        assert conversation_history.responses[0]["tool_call_id"] == "call-async"
        assert conversation_history.responses[1]["content"]["text"] == "sync:two"
        assert conversation_history.responses[1]["tool_call_id"] == "call-sync"

        tool_entries = [
            entry for entry in conversation_history.history if entry.get("role") == "tool"
        ]
        assert [entry.get("tool_call_id") for entry in tool_entries[-2:]] == [
            "call-async",
            "call-sync",
        ]

        assert provider.generate_calls, "Model should be called after executing tools"
        assert provider.generate_calls[0]["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "async:one"},
                "tool_call_id": "call-async",
            },
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "sync:two"},
                "tool_call_id": "call-sync",
            },
        ]

        log_entries = tool_manager.get_tool_activity_log()
        assert len(log_entries) >= 2
        assert log_entries[-2]["tool_name"] == "async_tool"
        assert log_entries[-2]["tool_call_id"] == "call-async"
        assert log_entries[-1]["tool_name"] == "sync_tool"
        assert log_entries[-1]["tool_call_id"] == "call-sync"

    asyncio.run(run_test())


def test_use_tool_enforces_timeout(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()
    tool_manager._conversation_tool_runtime_ms.clear()

    class DummyConversationHistory:
        def __init__(self):
            self.messages = []

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            return entry

        def add_response(self, *args, **kwargs):  # pragma: no cover - safety guard
            raise AssertionError("add_response should not be called when tool times out")

    conversation_history = DummyConversationHistory()
    provider_manager = types.SimpleNamespace(get_current_model=lambda: "dummy-model")

    class DummyConfigManager:
        def get_config(self, key, default=None):
            if key == "tool_defaults":
                return {"timeout_seconds": 0.05}
            if key == "conversation":
                return {"max_tool_duration_ms": 1000}
            return default

    async def slow_tool(delay):
        await asyncio.sleep(delay)
        return "finished"

    message = {
        "function_call": {
            "id": "call-timeout",
            "name": "slow_tool",
            "arguments": json.dumps({"delay": 0.2}),
        }
    }

    async def run_test():
        with pytest.raises(tool_manager.ToolExecutionError) as excinfo:
            await tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message=message,
                conversation_history=conversation_history,
                function_map={"slow_tool": slow_tool},
                functions=None,
                current_persona=None,
                temperature_var=0.0,
                top_p_var=1.0,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider_manager,
                config_manager=DummyConfigManager(),
            )

        error = excinfo.value
        assert error.error_type == "timeout"
        assert error.tool_call_id == "call-timeout"
        assert "timeout" in str(error).lower()
        assert conversation_history.messages, "Timeout should be recorded in history"
        recorded_entry = conversation_history.messages[-1]
        metadata = recorded_entry.get("metadata", {})
        assert metadata.get("error_type") == "timeout"
        assert recorded_entry["tool_call_id"] == "call-timeout"

        runtime = tool_manager._conversation_tool_runtime_ms.get("conversation", 0)
        assert runtime > 0

        activity_entry = tool_manager.get_tool_activity_log()[-1]
        assert activity_entry["tool_name"] == "slow_tool"
        assert activity_entry.get("error_type") == "timeout"

    asyncio.run(run_test())


def test_use_tool_respects_conversation_runtime_budget(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()
    tool_manager._conversation_tool_runtime_ms.clear()

    class DummyConversationHistory:
        def __init__(self):
            self.history = []
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)
            return entry

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    provider = DummyProviderManager()

    class DummyConfigManager:
        def get_config(self, key, default=None):
            if key == "tool_defaults":
                return {"timeout_seconds": 1.0}
            if key == "conversation":
                return {"max_tool_duration_ms": 20}
            return default

    async def slow_tool(delay):
        await asyncio.sleep(delay)
        return f"done:{delay}"

    message = {
        "function_call": {
            "id": "call-slow",
            "name": "slow_tool",
            "arguments": json.dumps({"delay": 0.05}),
        }
    }

    conversation_history = DummyConversationHistory()

    async def run_test():
        result = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"slow_tool": slow_tool},
            functions=None,
            current_persona=None,
            temperature_var=0.0,
            top_p_var=1.0,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=DummyConfigManager(),
        )

        assert result == "model-output"
        assert conversation_history.responses, "First tool call should record a response"

        with pytest.raises(tool_manager.ToolExecutionError) as excinfo:
            await tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message=message,
                conversation_history=conversation_history,
                function_map={"slow_tool": slow_tool},
                functions=None,
                current_persona=None,
                temperature_var=0.0,
                top_p_var=1.0,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider,
                config_manager=DummyConfigManager(),
            )

        error = excinfo.value
        assert error.error_type == "tool_runtime_budget_exceeded"
        assert error.tool_call_id == "call-slow"
        assert len(conversation_history.responses) == 1
        failure_entry = conversation_history.messages[-1]
        assert failure_entry["metadata"]["error_type"] == "tool_runtime_budget_exceeded"

        runtime = tool_manager._conversation_tool_runtime_ms.get("conversation", 0)
        assert runtime >= 50, "Tracked runtime should accumulate across calls"

    asyncio.run(run_test())

def test_use_tool_replays_generation_settings(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Hi"},
            ]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {"role": "tool", "content": _normalize_tool_response_payload(response)}
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class RecordingProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "default-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "assistant-reply"

    conversation_history = DummyConversationHistory()
    provider = RecordingProviderManager()

    async def echo_tool(value: str, context=None):
        return f"echo:{value}"

    async def run_test():
        await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message={
                "function_call": {
                    "id": "call-1",
                    "name": "echo_tool",
                    "arguments": json.dumps({"value": "data"}),
                }
            },
            conversation_history=conversation_history,
            function_map={"echo_tool": echo_tool},
            functions=[{"name": "echo_tool"}],
            current_persona={"name": "Helper"},
            temperature_var=0.2,
            top_p_var=0.5,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
            generation_settings={
                "model": "persona-model",
                "tool_choice": {"type": "function", "function": {"name": "echo_tool"}},
                "parallel_tool_calls": False,
                "json_mode": True,
            },
        )

    asyncio.run(run_test())

    assert provider.generate_calls, "Follow-up generation should be invoked"
    call_kwargs = provider.generate_calls[0]
    assert call_kwargs.get("model") == "persona-model"
    assert call_kwargs.get("tool_choice", {}).get("function", {}).get("name") == "echo_tool"
    assert call_kwargs.get("parallel_tool_calls") is False
    assert call_kwargs.get("json_mode") is True


def test_use_tool_runs_sync_tool_in_thread(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()
    tool_manager._tool_activity_log.clear()

    def slow_tool(value):
        time.sleep(0.2)
        return f"slow:{value}"

    message = {
        "function_call": {
            "id": "call-slow",
            "name": "slow_tool",
            "arguments": json.dumps({"value": "one"}),
        }
    }

    function_map = {"slow_tool": slow_tool}

    async def run_test():
        task = asyncio.create_task(
            tool_manager.use_tool(
                user="user",
                conversation_id="conversation",
                message=message,
                conversation_history=conversation_history,
                function_map=function_map,
                functions=None,
                current_persona=None,
                temperature_var=0.5,
                top_p_var=0.9,
                frequency_penalty_var=0.0,
                presence_penalty_var=0.0,
                conversation_manager=conversation_history,
                provider_manager=provider,
                config_manager=None,
            )
        )

        await asyncio.sleep(0.05)
        assert not task.done(), "use_tool should not block the event loop when running sync tools"

        response = await task
        assert response == "model-output"
        assert conversation_history.responses[-1]["content"]["text"] == "slow:one"
        assert conversation_history.responses[-1]["tool_call_id"] == "call-slow"
        log_entries = tool_manager.get_tool_activity_log()
        assert log_entries[-1]["tool_call_id"] == "call-slow"
        assert provider.generate_calls, "Provider should be invoked after tool execution"
        assert provider.generate_calls[0]["messages"] == [
            {"role": "user", "content": "Hi"},
            {
                "role": "tool",
                "content": {"type": "output_text", "text": "slow:one"},
                "tool_call_id": "call-slow",
            },
        ]

    asyncio.run(run_test())


def test_call_model_with_new_prompt_collects_stream(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class StreamingProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)

            async def _generator():
                for chunk in ["Hello", " ", "world"]:
                    yield chunk

            return _generator()

    provider = StreamingProviderManager()

    result = asyncio.run(
        tool_manager.call_model_with_new_prompt(
            messages=[{"role": "user", "content": "Hi"}],
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            functions=None,
            provider_manager=provider,
            conversation_manager=None,
            conversation_id="conversation",
            user="user",
            prompt="Please continue",
        )
    )

    assert result == "Hello world"
    assert provider.generate_calls
    assert provider.generate_calls[0].get("stream") is False


def test_use_tool_streams_final_response_when_enabled(monkeypatch):


    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class StreamingProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)

            async def _generator():
                for chunk in ["stream", "-", "result"]:
                    yield chunk

            return _generator()

    provider = StreamingProviderManager()

    async def tool_fn(value):
        return f"tool:{value}"

    conversation_history = DummyConversationHistory()

    message = {
        "function_call": {
            "id": "call-stream",
            "name": "tool_fn",
            "arguments": json.dumps({"value": "input"}),
        }
    }

    tool_manager._tool_activity_log.clear()

    persona = types.SimpleNamespace(name="Persona-Alpha")

    async def run_test():
        response = await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"tool_fn": tool_fn},
            functions=None,
            current_persona=persona,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
            stream=True,
        )

        assert isinstance(response, AsyncIteratorABC)
        chunks = []
        async for piece in response:
            chunks.append(piece)
        return "".join(chunks), list(chunks)

    collected, chunk_list = asyncio.run(run_test())

    assert collected == "stream-result"
    assert chunk_list == ["stream", "-", "result"]
    assert provider.generate_calls
    assert provider.generate_calls[0].get("stream") is True
    assert conversation_history.responses
    assert conversation_history.messages, "Assistant follow-up should be stored after streaming"
    recorded_message = conversation_history.messages[-1]
    assert recorded_message["content"] == [
        {"type": "output_text", "text": "stream"},
        {"type": "output_text", "text": "-"},
        {"type": "output_text", "text": "result"},
    ]
    assert recorded_message.get("metadata", {}).get("tool_call_ids") == [
        "call-stream"
    ]
    assert recorded_message.get("metadata", {}).get("persona") == "Persona-Alpha"



def test_use_tool_streams_tool_activity(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()

    async def generator_tool(value):
        yield {"type": "output_text", "text": value}
        await asyncio.sleep(0)
        yield {"type": "output_text", "text": value.upper()}

    message = {
        "function_call": {
            "id": "call-generator-stream",
            "name": "generator_tool",
            "arguments": json.dumps({"value": "chunk"}),
        }
    }

    published_entries: list[dict[str, object]] = []

    def fake_publish(event_name, payload):
        if event_name == "tool_activity":
            published_entries.append(payload)

    monkeypatch.setattr(tool_manager.event_system, "publish", fake_publish)
    monkeypatch.setattr(
        tool_manager,
        "_default_config_manager",
        _DummyConfigManager(log_full_payloads=True),
        raising=False,
    )
    tool_manager._tool_activity_log.clear()

    async def run_test():
        return await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"generator_tool": generator_tool},
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
            stream=True,
        )

    response = asyncio.run(run_test())

    assert response == "model-output"
    assert provider.generate_calls
    assert conversation_history.responses
    assert conversation_history.messages

    tool_entry = conversation_history.responses[0]
    expected_chunks = [
        {"type": "output_text", "text": "chunk"},
        {"type": "output_text", "text": "CHUNK"},
    ]
    assert tool_entry["content"] == _normalize_tool_response_payload(expected_chunks)

    assert published_entries, "tool activity events should be published"
    tracked = [
        entry
        for entry in published_entries
        if entry.get("tool_call_id") == "call-generator-stream"
    ]
    assert tracked, "expected matching tool activity events"
    recorded_results = [entry.get("result") for entry in tracked]
    assert recorded_results[0] == []
    assert recorded_results[-1] == expected_chunks
    assert any(len(result) == 1 for result in recorded_results if isinstance(result, list))


def test_tool_activity_log_redacts_sensitive_values(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()

    monkeypatch.setattr(
        tool_manager,
        "_default_config_manager",
        _DummyConfigManager(log_full_payloads=False, summary_length=64),
        raising=False,
    )

    entry = {
        "tool_name": "secrets",
        "tool_call_id": "call-1",
        "status": "success",
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T00:00:01",
        "arguments": {"api_key": "sk-live-1234567890ABCDEF"},
        "result": {"token": "AKIA1234567890ABCD"},
        "stdout": "api_key=sk-live-ABCDEF0123456789",
        "stderr": "token=AKIA1234567890ABCD",
        "duration_ms": 1000,
    }

    published: list[dict[str, object]] = []

    def capture_publish(event_name, payload):
        if event_name == "tool_activity":
            published.append(payload)

    monkeypatch.setattr(tool_manager.event_system, "publish", capture_publish)

    tool_manager._record_tool_activity(entry)

    log_entry = tool_manager.get_tool_activity_log()[-1]
    assert log_entry["payload_included"] is False
    for field in ("arguments", "result", "stdout", "stderr"):
        preview = log_entry["payload_preview"][field]
        assert "<redacted>" in preview
    assert log_entry["metrics"]["status"] == "success"
    assert log_entry["metrics"]["latency_ms"] == 1000

    assert published, "expected a published tool activity event"
    published_entry = published[-1]
    assert published_entry["payload_included"] is False
    assert "<redacted>" in published_entry["arguments"]


def test_tool_activity_log_honors_full_payload_logging(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)
    tool_manager._tool_activity_log.clear()

    monkeypatch.setattr(
        tool_manager,
        "_default_config_manager",
        _DummyConfigManager(log_full_payloads=True, summary_length=32),
        raising=False,
    )

    entry = {
        "tool_name": "inspector",
        "status": "success",
        "arguments": {"value": "ok"},
        "result": ["a", "b"],
        "stdout": "plain output",
        "stderr": "",
    }

    tool_manager._record_tool_activity(entry)
    log_entry = tool_manager.get_tool_activity_log()[-1]
    assert log_entry["payload_included"] is True
    assert log_entry["payload"]["result"] == ["a", "b"]
    assert log_entry["result"] == ["a", "b"]
    assert log_entry["metrics"]["status"] == "success"



def test_use_tool_consumes_async_generator_tool(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "Hi"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "model-output"

    conversation_history = DummyConversationHistory()
    provider = DummyProviderManager()

    async def generator_tool(value):
        print("tool-start")
        yield {"type": "output_text", "text": value}
        print("tool-middle")
        await asyncio.sleep(0)
        print("tool-warning", file=sys.stderr)
        yield {"type": "output_text", "text": value.upper()}
        print("tool-end")

    message = {
        "function_call": {
            "id": "call-generator",
            "name": "generator_tool",
            "arguments": json.dumps({"value": "chunk"}),
        }
    }

    tool_manager._tool_activity_log.clear()

    async def run_test():
        return await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map={"generator_tool": generator_tool},
            functions=None,
            current_persona=None,
            temperature_var=0.5,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider,
            config_manager=None,
        )

    response = asyncio.run(run_test())

    assert response == "model-output"
    assert conversation_history.responses
    assert conversation_history.messages

    tool_entry = conversation_history.responses[0]
    expected_chunks = [
        {"type": "output_text", "text": "chunk"},
        {"type": "output_text", "text": "CHUNK"},
    ]
    assert tool_entry["content"] == _normalize_tool_response_payload(expected_chunks)
    assert conversation_history.messages[0]["content"] == [
        {"type": "output_text", "text": "model-output"}
    ]

    assert provider.generate_calls

    log_entry = tool_manager._tool_activity_log[-1]
    assert log_entry["result"] == expected_chunks
    assert "tool-start" in log_entry["stdout"]
    assert "tool-middle" in log_entry["stdout"]
    assert "tool-end" in log_entry["stdout"]
    assert "tool-warning" in log_entry["stderr"]


def test_load_function_map_caches_by_persona(monkeypatch):
    persona_name = "CachePersona"
    persona_dir = Path("modules/Personas") / persona_name / "Toolbox"
    maps_path = persona_dir / "maps.py"
    module_name = f"persona_{persona_name}_maps"

    persona_dir.mkdir(parents=True, exist_ok=True)
    maps_path.write_text(
        textwrap.dedent(
            """
            EXECUTION_COUNTER = globals().get("EXECUTION_COUNTER", 0) + 1

            def sample_tool():
                return "ok"

            function_map = {"sample_tool": sample_tool}
            """
        )
    )

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )
    monkeypatch.setitem(tool_manager.__dict__, "_function_map_cache", {})
    sys.modules.pop(module_name, None)

    spec_calls = {"count": 0}
    original_spec = importlib.util.spec_from_file_location

    def counting_spec(*args, **kwargs):
        spec_calls["count"] += 1
        return original_spec(*args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", counting_spec)

    persona_payload = {"name": persona_name}

    try:
        first_map = tool_manager.load_function_map_from_current_persona(persona_payload)
        second_map = tool_manager.load_function_map_from_current_persona(persona_payload)

        assert set(first_map.keys()) == {"sample_tool"}
        assert set(second_map.keys()) == {"sample_tool"}
        first_entry = first_map["sample_tool"]
        second_entry = second_map["sample_tool"]
        assert _resolve_callable(first_entry) is _resolve_callable(second_entry)
        assert spec_calls["count"] == 1

        module = sys.modules[module_name]
        assert getattr(module, "EXECUTION_COUNTER", 0) == 1

        maps_path.write_text(
            textwrap.dedent(
                """
                EXECUTION_COUNTER = globals().get("EXECUTION_COUNTER", 0) + 1

                def updated_tool():
                    return "updated"

                function_map = {"updated_tool": updated_tool}
                """
            )
        )

        new_time = os.path.getmtime(maps_path) + 1
        os.utime(maps_path, (new_time, new_time))

        third_map = tool_manager.load_function_map_from_current_persona(persona_payload)

        assert spec_calls["count"] == 2
        assert "updated_tool" in third_map
        assert "sample_tool" not in third_map

    finally:
        sys.modules.pop(module_name, None)
        shutil.rmtree(Path("modules/Personas") / persona_name, ignore_errors=True)


def test_load_function_map_falls_back_to_default(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "MissingPersonaFallback"
    persona_dir = Path("modules/Personas") / persona_name
    shutil.rmtree(persona_dir, ignore_errors=True)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_map_cache.clear()
    tool_manager._default_function_map_cache = None
    sys.modules.pop(f"persona_{persona_name}_maps", None)
    sys.modules.pop("modules.Tools.tool_maps.maps", None)

    try:
        function_map = tool_manager.load_function_map_from_current_persona({"name": persona_name})

        assert isinstance(function_map, dict)
        assert "google_search" in function_map
        assert "policy_reference" in function_map
        assert persona_name not in tool_manager._function_map_cache
        assert tool_manager._default_function_map_cache is not None
    finally:
        sys.modules.pop(f"persona_{persona_name}_maps", None)
        tool_manager._default_function_map_cache = None
        shutil.rmtree(persona_dir, ignore_errors=True)


def test_persona_function_map_includes_metadata(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    persona_payload = {"name": "ATLAS"}

    function_map = tool_manager.load_function_map_from_current_persona(
        persona_payload, refresh=True
    )

    assert "google_search" in function_map
    assert "policy_reference" in function_map
    google_entry = function_map["google_search"]
    assert isinstance(google_entry, dict)
    assert callable(_resolve_callable(google_entry))

    metadata = google_entry.get("metadata")
    assert isinstance(metadata, Mapping)
    assert metadata.get("version") == "1.0.0"
    assert metadata.get("side_effects") == "none"
    assert metadata.get("allow_parallel") is True
    assert abs(metadata.get("cost_per_call") - 0.005) < 1e-9
    assert metadata.get("cost_unit") == "USD"
    auth_block = metadata.get("auth")
    assert isinstance(auth_block, dict)
    assert auth_block.get("env") == "GOOGLE_API_KEY"

    time_entry = function_map["get_current_info"]
    assert isinstance(time_entry, dict)
    time_meta = time_entry.get("metadata")
    assert isinstance(time_meta, Mapping)
    assert time_meta.get("default_timeout") == 5
    assert time_meta.get("side_effects") == "none"
    assert time_meta.get("cost_per_call") == 0.0
    assert time_meta.get("cost_unit") == "USD"


def test_medic_persona_exposes_pubmed_fetch(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    persona_payload = {"name": "MEDIC"}

    function_map = tool_manager.load_function_map_from_current_persona(
        persona_payload, refresh=True
    )

    assert "fetch_pubmed_details" in function_map
    fetch_entry = function_map["fetch_pubmed_details"]
    assert isinstance(fetch_entry, dict)
    fetch_callable = _resolve_callable(fetch_entry)
    assert callable(fetch_callable)


def test_docgenius_persona_includes_medical_search_suite(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)

    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    persona_payload = {"name": "DocGenius"}

    function_map = tool_manager.load_function_map_from_current_persona(
        persona_payload, refresh=True
    )

    for tool_name in ("search_pubmed", "search_pmc", "fetch_pubmed_details"):
        assert tool_name in function_map
        entry = function_map[tool_name]
        assert isinstance(entry, dict)
        resolved = _resolve_callable(entry)
        assert callable(resolved)


def test_use_tool_resolves_google_search_with_default_map(monkeypatch):
    _ensure_yaml(monkeypatch)
    _ensure_dotenv(monkeypatch)
    _ensure_pytz(monkeypatch)
    tool_manager = importlib.import_module("ATLAS.ToolManager")
    tool_manager = importlib.reload(tool_manager)

    persona_name = "PersonaWithoutToolbox"
    persona_dir = Path("modules/Personas") / persona_name
    shutil.rmtree(persona_dir, ignore_errors=True)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    tool_manager._function_map_cache.clear()
    tool_manager._default_function_map_cache = None
    sys.modules.pop(f"persona_{persona_name}_maps", None)
    sys.modules.pop("modules.Tools.tool_maps.maps", None)

    persona_payload = {"name": persona_name}

    class DummyConversationHistory:
        def __init__(self):
            self.history = [{"role": "user", "content": "hello"}]
            self.responses = []
            self.messages = []

        def add_response(
            self,
            user,
            conversation_id,
            response,
            timestamp,
            *,
            tool_call_id=None,
            metadata=None,
        ):
            entry = {
                "role": "tool",
                "content": _normalize_tool_response_payload(response),
            }
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            if metadata:
                entry["metadata"] = dict(metadata)
            self.responses.append(entry)
            self.history.append(entry)

        def add_message(
            self,
            user,
            conversation_id,
            role,
            content,
            timestamp,
            *,
            metadata=None,
            **kwargs,
        ):
            entry = {"role": role, "content": content}
            if metadata:
                entry["metadata"] = dict(metadata)
            if kwargs:
                entry.update(kwargs)
            self.messages.append(entry)
            self.history.append(entry)

        def get_history(self, user, conversation_id):
            return list(self.history)

    class DummyProviderManager:
        def __init__(self):
            self.generate_calls = []

        def get_current_model(self):
            return "dummy-model"

        async def generate_response(self, **kwargs):
            self.generate_calls.append(kwargs)
            return "final-response"

    shared_map = tool_manager.load_function_map_from_current_persona(persona_payload, refresh=True)

    assert isinstance(shared_map, dict)
    assert "google_search" in shared_map
    assert "policy_reference" in shared_map

    captured_args = {}

    async def fake_google_search(query, k=None, context=None):
        captured_args["query"] = query
        captured_args["k"] = k
        return {"query": query, "k": k}

    function_map = dict(shared_map)
    function_map["google_search"] = fake_google_search

    conversation_history = DummyConversationHistory()
    provider_manager = DummyProviderManager()

    message = {
        "tool_calls": [
            {
                "id": "call-google",
                "type": "tool",
                "function": {
                    "name": "google_search",
                    "arguments": json.dumps({"query": "atlas project", "k": 3}),
                },
            }
        ]
    }

    tool_manager._tool_activity_log.clear()

    async def run_test():
        return await tool_manager.use_tool(
            user="user",
            conversation_id="conversation",
            message=message,
            conversation_history=conversation_history,
            function_map=function_map,
            functions=None,
            current_persona=persona_payload,
            temperature_var=0.1,
            top_p_var=0.9,
            frequency_penalty_var=0.0,
            presence_penalty_var=0.0,
            conversation_manager=conversation_history,
            provider_manager=provider_manager,
            config_manager=None,
        )

    try:
        result = asyncio.run(run_test())

        assert result == "final-response"
        assert captured_args == {"query": "atlas project", "k": 3}
        assert conversation_history.responses
        response_entry = conversation_history.responses[-1]
        assert response_entry["tool_call_id"] == "call-google"
        assert response_entry["content"] == {"query": "atlas project", "k": 3}
        assert provider_manager.generate_calls
        final_entry = conversation_history.messages[-1]
        assert final_entry["content"] == [
            {"type": "output_text", "text": "final-response"}
        ]
        assert persona_name not in tool_manager._function_map_cache
        assert tool_manager._default_function_map_cache is not None
    finally:
        sys.modules.pop(f"persona_{persona_name}_maps", None)
        tool_manager._default_function_map_cache = None
        shutil.rmtree(persona_dir, ignore_errors=True)

