import dataclasses
import sys
import types
from pathlib import Path

import pytest

import importlib

# Stub heavy tool dependencies to avoid pulling optional runtime packages during tests.
base_tools_pkg = types.ModuleType("modules.Tools.Base_Tools")
base_tools_pkg.__path__ = []
vector_store_stub = types.ModuleType("modules.Tools.Base_Tools.vector_store")

class _VectorStoreService:  # pragma: no cover - test shim
    async def upsert_vectors(self, *args, **kwargs):
        raise NotImplementedError

    async def query_vectors(self, *args, **kwargs):
        raise NotImplementedError

    async def delete_namespace(self, *args, **kwargs):
        raise NotImplementedError


class _VectorRecord:  # pragma: no cover - test shim
    def __init__(self, id=None, values=(), metadata=None):
        self.id = id
        self.values = values
        self.metadata = metadata or {}


class _QueryMatch:  # pragma: no cover - test shim
    def __init__(self, id=None, score=0.0, values=None, metadata=None):
        self.id = id
        self.score = score
        self.values = values
        self.metadata = metadata or {}


vector_store_stub.VectorStoreService = _VectorStoreService
vector_store_stub.VectorRecord = _VectorRecord
vector_store_stub.QueryMatch = _QueryMatch

tools_pkg = types.ModuleType("modules.Tools")
tools_pkg.__path__ = []
sys.modules["modules.Tools"] = tools_pkg
sys.modules["modules.Tools.Base_Tools"] = base_tools_pkg
sys.modules["modules.Tools.Base_Tools.vector_store"] = vector_store_stub
setattr(base_tools_pkg, "vector_store", vector_store_stub)
tool_event_system = types.ModuleType("modules.Tools.tool_event_system")
tool_event_system.publish_bus_event = lambda *args, **kwargs: None
sys.modules["modules.Tools.tool_event_system"] = tool_event_system
task_queue_stub = types.ModuleType("modules.Tools.Base_Tools.task_queue")
task_queue_stub.TaskQueueService = type("TaskQueueService", (), {})
task_queue_stub.get_default_task_queue_service = lambda *args, **kwargs: None
sys.modules["modules.Tools.Base_Tools.task_queue"] = task_queue_stub
setattr(base_tools_pkg, "task_queue", task_queue_stub)

task_store_stub = types.ModuleType("modules.task_store")
task_store_stub.__path__ = []
task_store_models_stub = types.ModuleType("modules.task_store.models")
task_store_stub.models = task_store_models_stub
task_store_stub.TaskService = type("TaskService", (), {})
task_store_stub.TaskStoreRepository = type("TaskStoreRepository", (), {})
sys.modules["modules.task_store"] = task_store_stub
sys.modules["modules.task_store.models"] = task_store_models_stub
task_store_repository_stub = types.ModuleType("modules.task_store.repository")
task_store_repository_stub.TaskStoreRepository = task_store_stub.TaskStoreRepository
sys.modules["modules.task_store.repository"] = task_store_repository_stub

job_store_stub = types.ModuleType("modules.job_store")
job_store_stub.__path__ = []
job_store_stub.JobService = type("JobService", (), {})
job_store_stub.MongoJobStoreRepository = type("MongoJobStoreRepository", (), {})
job_store_stub.ensure_job_schema = lambda *args, **kwargs: None
sys.modules["modules.job_store"] = job_store_stub
sys.modules["modules.job_store.models"] = types.ModuleType("modules.job_store.models")
job_store_repository_stub = types.ModuleType("modules.job_store.repository")
job_store_repository_stub.JobStoreRepository = type("JobStoreRepository", (), {})
sys.modules["modules.job_store.repository"] = job_store_repository_stub

atlas_config = importlib.import_module("ATLAS.config")
if not hasattr(atlas_config, "PerformanceMode") or not hasattr(atlas_config, "StorageArchitecture"):
    pytest.skip("ATLAS.config is unavailable; skipping setup controller tests.", allow_module_level=True)

PerformanceMode = atlas_config.PerformanceMode
StorageArchitecture = atlas_config.StorageArchitecture
from ATLAS.setup import (
    AdminProfile,
    JobSchedulingState,
    KvStoreState,
    RetryPolicyState,
    VectorStoreState,
    SetupWizardController,
)
from modules.conversation_store.bootstrap import BootstrapError


class _StubKVStoreSection:
    def __init__(self):
        self._settings = {
            "default_adapter": "postgres",
            "adapters": {"postgres": {"reuse_conversation_store": True, "url": None}},
        }

    def get_settings(self):
        return dict(self._settings)


class _StubPersistenceSection:
    def __init__(self):
        self.kv_store = _StubKVStoreSection()


class _StubConfigManager:
    """Minimal stub providing the methods SetupWizardController expects."""

    UNSET = object()
    write_calls = 0
    export_paths: list[str] = []
    import_paths: list[str] = []
    current_host = "localhost"
    next_import_host = "imported"

    def __init__(self):
        self.env_config = {}
        self.yaml_config: dict[str, object] = {}
        self.config: dict[str, object] = {}
        self.persistence = _StubPersistenceSection()

    def get_conversation_database_config(self):
        host = self.__class__.current_host
        return {"url": f"postgresql://{host}/atlas"}

    def get_conversation_backend(self):
        return "postgresql"

    def get_job_scheduling_settings(self):
        return dict(getattr(self, "job_scheduling", {}))

    def get_messaging_settings(self):
        return {}

    def get_kv_store_settings(self):
        return self.persistence.kv_store.get_settings()

    def _get_provider_env_keys(self):
        return {}

    def get_config(self, _key):
        return None

    def get_default_provider(self):
        return None

    def get_default_model(self):
        return None

    def get_tts_enabled(self):
        return False

    def get_conversation_retention_policies(self):
        return {}

    def get_vector_store_settings(self):
        return {"default_adapter": "in_memory", "adapters": {"in_memory": {}}}

    def get_storage_architecture(self):
        return StorageArchitecture()

    def get_storage_architecture_settings(self):
        return StorageArchitecture().to_dict()

    def set_storage_architecture(self, architecture):
        self.storage_architecture = architecture
        return architecture.to_dict()

    def _write_yaml_config(self):
        self.__class__.write_calls += 1

    def set_job_scheduling_settings(
        self,
        *,
        enabled=None,
        job_store_url=None,
        max_workers=None,
        retry_policy=None,
        timezone=None,
        queue_size=None,
    ):
        self.job_scheduling = {}
        if enabled is not None:
            self.job_scheduling["enabled"] = bool(enabled)
        if job_store_url is not self.UNSET:
            if job_store_url:
                self.job_scheduling["job_store_url"] = job_store_url
        if max_workers is not self.UNSET and max_workers is not None:
            self.job_scheduling["max_workers"] = max_workers
        if timezone is not self.UNSET and timezone:
            self.job_scheduling["timezone"] = timezone
        if queue_size is not self.UNSET and queue_size is not None:
            self.job_scheduling["queue_size"] = queue_size
        if retry_policy is not None:
            self.job_scheduling["retry_policy"] = retry_policy
        return dict(self.job_scheduling)

    def get_audit_template(self):
        return None

    def get_data_residency_settings(self):
        return {}

    def get_company_identity(self):
        return {}

    def export_yaml_config(self, path):
        resolved = str(Path(path))
        self.__class__.export_paths.append(resolved)
        return resolved

    def import_yaml_config(self, path):
        resolved = str(Path(path))
        self.__class__.import_paths.append(resolved)
        self.__class__.current_host = self.__class__.next_import_host
        return {"DATABASE": {"HOST": "imported"}}


def test_build_summary_includes_kv_store_payload():
    controller = SetupWizardController(config_manager=_StubConfigManager())
    controller.state.kv_store = KvStoreState(
        reuse_conversation_store=False,
        url="postgresql://kv",
    )

    summary = controller.build_summary()

    assert "kv_store" in summary
    assert summary["kv_store"] == dataclasses.asdict(controller.state.kv_store)
    assert summary["setup_type"] == dataclasses.asdict(controller.state.setup_type)


def test_build_summary_includes_storage_architecture_and_mode():
    controller = SetupWizardController(config_manager=_StubConfigManager())
    controller.state.database.backend = "mongodb"
    controller.state.vector_store = VectorStoreState(adapter="qdrant")
    controller.state.kv_store = KvStoreState(
        reuse_conversation_store=False,
        url="redis://cache", 
    )
    controller.state.storage_architecture = StorageArchitecture(
        performance_mode=PerformanceMode.PERFORMANCE,
        conversation_backend="mongodb",
        kv_reuse_conversation_store=False,
        vector_store_adapter="qdrant",
    )

    summary = controller.build_summary()

    architecture = summary.get("storage_architecture")
    assert architecture["performance_mode"] == PerformanceMode.PERFORMANCE.value
    assert architecture["main_db"] == "mongodb"
    assert architecture["document_db"] == "mongodb"
    assert architecture["vector_db"] == "qdrant"
    assert architecture["kv_store"]["reuse_conversation_store"] is False
    assert architecture["kv_store"]["url"] == "redis://cache"


def test_user_profile_staging_and_summary_mask_secrets():
    controller = SetupWizardController(config_manager=_StubConfigManager())
    profile = AdminProfile(
        username="admin",
        email="admin@example.com",
        password="Secret123!",
        display_name="Admin",
        full_name="Administrator Example",
        domain="example.com",
        date_of_birth="1990-01-01",
        sudo_username="root",
        sudo_password="sudo-pass",
        privileged_db_username="postgres",
        privileged_db_password="db-secret",
    )

    controller.set_user_profile(profile)

    staged = controller.get_privileged_credentials()
    assert staged == ("postgres", "db-secret")

    summary = controller.build_summary()
    user_summary = summary["user"]

    assert user_summary["username"] == "admin"
    assert user_summary["email"] == "admin@example.com"
    assert user_summary["full_name"] == "Administrator Example"
    assert user_summary["domain"] == "example.com"
    assert user_summary["date_of_birth"] == "1990-01-01"
    assert user_summary["has_password"] is True
    assert user_summary["privileged_credentials"]["sudo_username"] == "root"
    assert user_summary["privileged_credentials"]["has_sudo_password"] is True
    assert user_summary["database_privileged_username"] == "postgres"
    assert user_summary["has_database_privileged_password"] is True
    assert "password" not in user_summary


def test_apply_setup_type_personal_only_applies_once():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("personal")

    assert controller.state.setup_type.mode == "personal"
    assert controller.state.setup_type.applied is True
    assert controller.state.message_bus.backend == "in_memory"
    assert controller.state.message_bus.redis_url is None
    assert controller.state.job_scheduling.enabled is False
    assert controller.state.kv_store.reuse_conversation_store is True
    assert controller.state.kv_store.url is None
    assert controller.state.optional.retention_days is None
    assert controller.state.optional.retention_history_limit is None
    assert controller.state.optional.http_auto_start is True

    controller.state.optional = dataclasses.replace(
        controller.state.optional,
        http_auto_start=False,
    )

    controller.apply_setup_type("PERSONAL")

    assert controller.state.optional.http_auto_start is False


def test_apply_setup_type_switches_between_presets():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("personal")
    controller.apply_setup_type("enterprise")

    assert controller.state.setup_type.mode == "enterprise"
    assert controller.state.setup_type.applied is True
    assert controller.state.message_bus.backend == "redis"
    assert controller.state.message_bus.redis_url == "redis://localhost:6379/0"
    assert controller.state.message_bus.stream_prefix == "atlas"
    assert controller.state.job_scheduling.enabled is True
    assert (
        controller.state.job_scheduling.job_store_url
        == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
    )
    assert controller.state.job_scheduling.queue_size == 500
    assert controller.state.job_scheduling.timezone == "UTC"
    assert controller.state.job_scheduling.max_workers == 8
    assert controller.state.kv_store.reuse_conversation_store is False
    assert (
        controller.state.kv_store.url
        == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache"
    )
    assert controller.state.optional.retention_days == 30
    assert controller.state.optional.retention_history_limit == 500
    assert controller.state.optional.http_auto_start is False

    custom_state = dataclasses.replace(controller.state.job_scheduling, queue_size=5)
    controller.state.job_scheduling = custom_state
    controller.apply_setup_type("enterprise")
    assert controller.state.job_scheduling.queue_size == 5


def test_apply_setup_type_student_defaults():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("student")

    assert controller.state.setup_type.mode == "student"
    assert controller.state.setup_type.applied is True
    assert controller.state.message_bus.backend == "in_memory"
    assert controller.state.message_bus.redis_url is None
    assert controller.state.job_scheduling.enabled is False
    assert controller.state.kv_store.reuse_conversation_store is True
    assert controller.state.optional.retention_days == 7
    assert controller.state.optional.retention_history_limit == 100
    assert controller.state.optional.http_auto_start is True


def test_apply_setup_type_student_with_local_only():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("student", local_only=True)

    assert controller.state.setup_type.mode == "student"
    assert controller.state.setup_type.local_only is True
    assert controller.state.database.backend == "sqlite"


def test_apply_setup_type_enthusiast_defaults():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("enthusiast")

    assert controller.state.setup_type.mode == "enthusiast"
    assert controller.state.setup_type.applied is True
    assert controller.state.database.backend == "postgresql"
    assert controller.state.message_bus.backend == "redis"
    assert controller.state.message_bus.redis_url == "redis://localhost:6379/0"
    assert controller.state.message_bus.stream_prefix == "atlas-power"
    assert controller.state.job_scheduling.enabled is True
    assert controller.state.job_scheduling.job_store_url == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs"
    assert controller.state.kv_store.reuse_conversation_store is False
    assert controller.state.kv_store.url == "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache"
    assert controller.state.optional.retention_days == 90
    assert controller.state.optional.retention_history_limit == 1000
    assert controller.state.optional.http_auto_start is True
    assert controller.state.storage_architecture.performance_mode == PerformanceMode.PERFORMANCE


def test_apply_setup_type_with_developer_mode():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    # Apply personal with developer mode enabled
    controller.apply_setup_type("personal", developer_mode=True)

    assert controller.state.setup_type.mode == "personal"
    assert controller.state.setup_type.developer_mode is True
    # Developer mode overlay should enable Redis and job scheduling
    assert controller.state.message_bus.backend == "redis"
    assert controller.state.message_bus.redis_url == "redis://localhost:6379/0"
    assert controller.state.job_scheduling.enabled is True


def test_set_developer_mode_toggle():
    controller = SetupWizardController(config_manager=_StubConfigManager())

    controller.apply_setup_type("student")
    assert controller.state.message_bus.backend == "in_memory"
    assert controller.state.setup_type.developer_mode is False

    # Enable developer mode
    controller.set_developer_mode(True)

    assert controller.state.setup_type.developer_mode is True
    assert controller.state.message_bus.backend == "redis"
    assert controller.state.job_scheduling.enabled is True


def test_run_preflight_sets_recommendation(monkeypatch):
    controller = SetupWizardController(config_manager=_StubConfigManager())

    def _fake_metrics():
        return {
            "cpu_cores": 12.0,
            "memory_total": 64 * 1024**3,
            "disk_free": 500 * 1024**3,
            "gpu_count": 1.0,
            "network_speed": 1000.0,
        }

    monkeypatch.setattr(controller, "_collect_preflight_metrics", _fake_metrics)

    profile = controller.run_preflight()

    assert profile.tier == "accelerated"
    assert controller.state.setup_recommended_mode == "performance"

    summary = controller.build_summary()
    assert summary["hardware_profile"]["tier"] == "accelerated"
    assert summary["setup_recommended_mode"] == "performance"


def test_export_config_writes_and_refreshes_state(tmp_path):
    _StubConfigManager.write_calls = 0
    _StubConfigManager.export_paths.clear()
    _StubConfigManager.current_host = "initial"
    manager = _StubConfigManager()
    controller = SetupWizardController(config_manager=manager)

    _StubConfigManager.current_host = "refreshed-host"
    destination = tmp_path / "export.yaml"

    result = controller.export_config(destination)

    assert result == str(destination)
    assert _StubConfigManager.write_calls >= 1
    assert _StubConfigManager.export_paths[-1] == str(destination)
    assert controller.config_manager is not manager
    assert controller.state.database.host == "refreshed-host"


def test_import_config_refreshes_state(tmp_path):
    _StubConfigManager.import_paths.clear()
    _StubConfigManager.current_host = "before"
    _StubConfigManager.next_import_host = "restored"
    controller = SetupWizardController(config_manager=_StubConfigManager())

    source = tmp_path / "config.yaml"
    result = controller.import_config(source)

    assert result == str(source)
    assert _StubConfigManager.import_paths[-1] == str(source)


def test_apply_job_scheduling_bootstraps_and_applies_schema(monkeypatch):
    engine_calls: dict[str, object] = {}

    class _DummyEngine:
        disposed = False

        def dispose(self):  # pragma: no cover - defensive
            self.disposed = True

    dummy_engine = _DummyEngine()

    def _fake_engine(url, *, future):
        engine_calls["url"] = url
        engine_calls["future"] = future
        return dummy_engine

    def _fake_schema(engine):
        engine_calls["schema_engine"] = engine

    def _fake_bootstrap(dsn: str, **kwargs):
        engine_calls["bootstrap_dsn"] = dsn
        engine_calls["bootstrap_kwargs"] = kwargs
        return "postgresql+psycopg://bootstrapped:secret@localhost:5432/job_store"

    monkeypatch.setattr("ATLAS.setup.controller.create_engine", _fake_engine)
    monkeypatch.setattr("ATLAS.setup.controller.ensure_job_schema", _fake_schema)

    controller = SetupWizardController(
        config_manager=_StubConfigManager(),
        bootstrap=_fake_bootstrap,
        request_privileged_password=lambda: "prompted",
    )
    controller.set_privileged_credentials(("admin", "pw"))

    state = JobSchedulingState(
        enabled=True,
        job_store_url="postgresql+psycopg://atlas:atlas@localhost:5432/jobs",
        max_workers=2,
        retry_policy=RetryPolicyState(),
        timezone="UTC",
        queue_size=5,
    )

    settings = controller.apply_job_scheduling(state)

    assert engine_calls["bootstrap_dsn"] == state.job_store_url
    assert engine_calls["bootstrap_kwargs"].get("privileged_credentials") == ("admin", "pw")
    assert engine_calls["bootstrap_kwargs"].get("request_privileged_password") is not None
    assert engine_calls["url"] == "postgresql+psycopg://bootstrapped:secret@localhost:5432/job_store"
    assert engine_calls["schema_engine"] is dummy_engine
    assert dummy_engine.disposed is True
    assert controller.state.job_scheduling.job_store_url == engine_calls["url"]
    assert settings.get("job_store_url") == engine_calls["url"]


def test_apply_job_scheduling_raises_on_schema_error(monkeypatch):
    def _fake_bootstrap(dsn: str, **_kwargs):
        return dsn

    class _Disposable:
        def dispose(self):
            pass

    def _fake_engine(url, *, future):  # pragma: no cover - defensive stub
        return _Disposable()

    def _fail_schema(_engine):
        raise RuntimeError("boom")

    monkeypatch.setattr("ATLAS.setup.controller.create_engine", _fake_engine)
    monkeypatch.setattr("ATLAS.setup.controller.ensure_job_schema", _fail_schema)

    controller = SetupWizardController(
        config_manager=_StubConfigManager(),
        bootstrap=_fake_bootstrap,
    )

    state = JobSchedulingState(
        enabled=True,
        job_store_url="postgresql+psycopg://atlas:atlas@localhost:5432/jobs",
        retry_policy=RetryPolicyState(),
    )

    with pytest.raises(BootstrapError):
        controller.apply_job_scheduling(state)

    assert getattr(controller.config_manager, "job_scheduling", {}) == {}
    assert controller.state.database.host == "restored"


def test_import_config_propagates_yaml_errors():
    class _FailingManager(_StubConfigManager):
        def import_yaml_config(self, path):  # type: ignore[override]
            raise ValueError("bad yaml")

    controller = SetupWizardController(
        config_manager=_FailingManager(),
        config_manager_factory=_FailingManager,
    )

    with pytest.raises(ValueError):
        controller.import_config("/tmp/config.yaml")
