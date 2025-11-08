import dataclasses

from ATLAS.setup.controller import AdminProfile, KvStoreState, SetupWizardController


class _StubConfigManager:
    """Minimal stub providing the methods SetupWizardController expects."""

    UNSET = object()

    def __init__(self):
        self.env_config = {}

    def get_conversation_database_config(self):
        return {}

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


def test_build_summary_includes_kv_store_payload():
    controller = SetupWizardController(config_manager=_StubConfigManager())
    controller.state.kv_store = KvStoreState(
        reuse_conversation_store=False,
        url="postgresql://kv",
    )

    summary = controller.build_summary()

    assert "kv_store" in summary
    assert summary["kv_store"] == dataclasses.asdict(controller.state.kv_store)


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
