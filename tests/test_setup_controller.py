import dataclasses
from pathlib import Path

import pytest

from ATLAS.setup import AdminProfile, KvStoreState, SetupWizardController


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

    def get_conversation_database_config(self):
        host = self.__class__.current_host
        return {"url": f"postgresql://{host}/atlas"}

    def get_job_scheduling_settings(self):
        return {}

    def get_messaging_settings(self):
        return {}

    def get_kv_store_settings(self):
        return {}

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

    def _write_yaml_config(self):
        self.__class__.write_calls += 1

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
    assert controller.state.job_scheduling.queue_size == 100
    assert controller.state.job_scheduling.timezone == "UTC"
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
