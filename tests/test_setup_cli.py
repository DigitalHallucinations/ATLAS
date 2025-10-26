import dataclasses
import types
from pathlib import Path

from ATLAS.setup import OptionalState, UserState
from ATLAS.setup.cli import SetupUtility
from ATLAS.setup.controller import (
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    ProviderState,
    SpeechState,
)


class DummyController:
    def __init__(self) -> None:
        self.state = types.SimpleNamespace(
            database=DatabaseState(),
            kv_store=KvStoreState(),
            job_scheduling=JobSchedulingState(),
            message_bus=MessageBusState(),
            providers=ProviderState(),
            speech=SpeechState(),
            user=UserState(),
            optional=OptionalState(),
        )
        self.applied_database_states: list[DatabaseState] = []
        self.registered_users: list[UserState] = []

    def apply_database_settings(self, state: DatabaseState) -> str:
        self.applied_database_states.append(state)
        ensured = (
            f"postgresql+psycopg://{state.user}:{state.password}@{state.host}:{state.port}/{state.database}"
        )
        self.state.database = dataclasses.replace(state, dsn=ensured)
        return ensured

    def apply_kv_store_settings(self, state: KvStoreState):  # pragma: no cover - not needed for these tests
        self.state.kv_store = dataclasses.replace(state)
        return {}

    def apply_job_scheduling(self, state: JobSchedulingState):  # pragma: no cover - not needed
        self.state.job_scheduling = dataclasses.replace(state)
        return {}

    def apply_message_bus(self, state: MessageBusState):  # pragma: no cover - not needed
        self.state.message_bus = dataclasses.replace(state)
        return {}

    def apply_provider_settings(self, state: ProviderState):  # pragma: no cover - not needed
        self.state.providers = dataclasses.replace(state)
        return self.state.providers

    def apply_speech_settings(self, state: SpeechState):  # pragma: no cover - not needed
        self.state.speech = dataclasses.replace(state)
        return self.state.speech

    def apply_optional_settings(self, state: OptionalState):  # pragma: no cover - not needed
        self.state.optional = dataclasses.replace(state)
        return self.state.optional

    def register_user(self, state: UserState):
        self.registered_users.append(state)
        self.state.user = dataclasses.replace(state)
        return {
            "username": state.username,
            "email": state.email,
            "display_name": state.display_name,
        }

    def build_summary(self):  # pragma: no cover - not needed
        return {}


def test_configure_database_persists_dsn():
    controller = DummyController()
    responses = iter(["db.internal", "6543", "atlas_prod", "atlas_user"])
    passwords = iter(["s3cret"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: next(passwords),
        print_func=lambda message: None,
    )

    dsn = utility.configure_database()

    applied = controller.applied_database_states[-1]
    assert applied.host == "db.internal"
    assert applied.port == 6543
    assert controller.state.database.dsn == dsn
    assert dsn.endswith("@db.internal:6543/atlas_prod")


def test_configure_user_registers_admin():
    controller = DummyController()
    responses = iter(["admin", "admin@example.com", "Administrator"])
    passwords = iter(["P@ssw0rd!", "P@ssw0rd!"])

    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: next(responses),
        getpass_func=lambda prompt: next(passwords),
        print_func=lambda message: None,
    )

    result = utility.configure_user()

    assert controller.registered_users
    registered = controller.registered_users[-1]
    assert registered.username == "admin"
    assert registered.email == "admin@example.com"
    assert registered.display_name == "Administrator"
    assert result["username"] == "admin"


def test_finalize_writes_marker(monkeypatch, tmp_path: Path):
    controller = DummyController()
    utility = SetupUtility(
        controller=controller,
        input_func=lambda prompt: "",
        getpass_func=lambda prompt: "",
        print_func=lambda message: None,
    )

    marker_path = tmp_path / "marker.json"
    captured = {}

    def fake_write(payload):
        captured.update(payload)
        return marker_path

    monkeypatch.setattr("ATLAS.setup.cli.write_setup_marker", fake_write)

    summary = {"database": {"dsn": "postgresql://example"}}
    result = utility.finalize(summary)

    assert captured["setup_complete"] is True
    assert captured["summary"] == summary
    assert result == marker_path
