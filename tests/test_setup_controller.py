import dataclasses

from ATLAS.setup.controller import KvStoreState, SetupWizardController


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
