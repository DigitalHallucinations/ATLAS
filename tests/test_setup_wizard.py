import types

import pytest

gi = pytest.importorskip("gi")

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Setup.setup_wizard import SetupWizardWindow
from ATLAS.setup import (
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    OptionalState,
    PrivilegedCredentialState,
    ProviderState,
    SpeechState,
    UserState,
)
from modules.conversation_store.bootstrap import BootstrapError


class FakeController:
    def __init__(self):
        self.state = types.SimpleNamespace(
            database=DatabaseState(),
            job_scheduling=JobSchedulingState(),
            message_bus=MessageBusState(),
            kv_store=KvStoreState(),
            providers=ProviderState(),
            speech=SpeechState(),
            optional=OptionalState(),
            user=UserState(),
        )
        self.calls = []
        self.profile_calls = []
        self.register_calls = []
        self.summary = {"status": "ok"}
        self._staged_privileged_credentials = None

    def apply_database_settings(self, state):
        self.calls.append(("database", state))
        self.state.database = state
        return "dsn"

    def apply_job_scheduling(self, state):
        self.calls.append(("job_scheduling", state))
        self.state.job_scheduling = state
        return {"enabled": state.enabled}

    def apply_message_bus(self, state):
        self.calls.append(("message_bus", state))
        self.state.message_bus = state
        return {"backend": state.backend}

    def apply_kv_store_settings(self, state):
        self.calls.append(("kv_store", state))
        self.state.kv_store = state
        return {"reuse": state.reuse_conversation_store}

    def apply_provider_settings(self, state):
        self.calls.append(("providers", state))
        self.state.providers = state
        return state

    def apply_speech_settings(self, state):
        self.calls.append(("speech", state))
        self.state.speech = state
        return state

    def apply_optional_settings(self, state):
        self.calls.append(("optional", state))
        self.state.optional = state
        return state

    def set_user_profile(self, profile):
        self.profile_calls.append(profile)
        self.state.user = UserState(
            username=profile.username or "",
            email=profile.email or "",
            password=profile.password or "",
            display_name=profile.display_name or "",
            full_name=profile.full_name or "",
            domain=profile.domain or "",
            date_of_birth=profile.date_of_birth or "",
            privileged_credentials=PrivilegedCredentialState(
                sudo_username=profile.sudo_username or "",
                sudo_password=profile.sudo_password or "",
            ),
        )
        if profile.privileged_db_username or profile.privileged_db_password:
            self.set_privileged_credentials(
                (profile.privileged_db_username or None, profile.privileged_db_password or None)
            )
        return self.state.user

    def set_privileged_credentials(self, credentials):
        self._staged_privileged_credentials = credentials

    def get_privileged_credentials(self):
        return self._staged_privileged_credentials

    def register_user(self, state=None):
        staged = state or self.state.user
        self.register_calls.append(staged)
        return {
            "username": staged.username,
            "display_name": staged.display_name,
            "full_name": staged.full_name,
        }

    def build_summary(self):
        return self.summary


def _complete_database_step(window):
    window._database_entries["host"].set_text("db.example.com")
    window._database_entries["port"].set_text("5433")
    window._database_entries["database"].set_text("atlas_test")
    window._database_entries["user"].set_text("atlas_user")
    window._database_entries["password"].set_text("secret")
    window._on_next_clicked(None)


def _complete_job_step(window):
    window._job_widgets["enabled"].set_active(False)
    window._on_next_clicked(None)


def _complete_message_bus_step(window):
    window._on_next_clicked(None)


def _complete_kv_step(window):
    window._on_next_clicked(None)


def _complete_providers_step(window):
    buffer = window._provider_buffer
    if buffer is not None:
        buffer.set_text("")
    window._on_next_clicked(None)


def _complete_speech_step(window):
    window._on_next_clicked(None)


def _populate_user_entries(window, **overrides):
    defaults = {
        "full_name": "Atlas Admin",
        "username": "admin",
        "email": "admin@example.com",
        "domain": "Example.COM",
        "date_of_birth": "1990-01-01",
        "password": "changeme",
        "confirm_password": "changeme",
        "sudo_username": "atlas-admin",
        "sudo_password": "SudoPass!",
        "confirm_sudo_password": "SudoPass!",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        window._user_entries[key].set_text(value)


def _complete_user_step(window, **overrides):
    _populate_user_entries(window, **overrides)
    window._on_next_clicked(None)


def test_stack_switcher_updates_current_index():
    application = Gtk.Application()
    controller = FakeController()
    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    assert window._current_index == 0
    assert controller.calls == []

    target_step = window._steps[3]
    window._stack.set_visible_child(target_step.widget)

    assert window._current_index == 3

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == ["message_bus"]
    assert window._current_index == 4


class _CallbackRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def test_setup_wizard_happy_path(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()
    marker_calls = []

    def _fake_write_setup_marker(summary):
        marker_calls.append(summary)
        return None

    monkeypatch.setattr(
        "GTKUI.Setup.setup_wizard.write_setup_marker",
        _fake_write_setup_marker,
    )

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _complete_user_step(window)

    assert controller.profile_calls
    assert controller.register_calls == []
    assert controller.calls == []
    assert window._current_index == 1
    assert controller.state.user.username == "admin"
    assert controller.state.user.domain == "example.com"
    staged_profile = controller.profile_calls[-1]
    assert staged_profile.username == "admin"
    assert staged_profile.email == "admin@example.com"
    assert staged_profile.full_name == "Atlas Admin"
    assert staged_profile.domain == "example.com"
    assert staged_profile.sudo_username == "atlas-admin"
    assert staged_profile.sudo_password == "SudoPass!"
    assert window._database_entries["user"].get_text() == "admin"
    tenant_entry = window._optional_widgets["tenant_id"]
    assert isinstance(tenant_entry, Gtk.Entry)
    assert tenant_entry.get_text() == "example.com"

    window._database_entries["host"].set_text("db.example.com")
    window._database_entries["port"].set_text("5433")
    window._database_entries["database"].set_text("atlas_test")
    window._database_entries["user"].set_text("atlas_user")
    window._database_entries["password"].set_text("secret")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == ["database"]
    assert window._current_index == 2
    assert not on_error.calls

    window._job_widgets["enabled"].set_active(True)
    window._job_widgets["job_store_url"].set_text("sqlite:///jobs.db")
    window._job_widgets["max_workers"].set_text("5")
    window._job_widgets["timezone"].set_text("UTC")
    window._job_widgets["queue_size"].set_text("50")
    window._job_widgets["retry_max_attempts"].set_text("4")
    window._job_widgets["retry_backoff_seconds"].set_text("10.5")
    window._job_widgets["retry_jitter_seconds"].set_text("1.5")
    window._job_widgets["retry_backoff_multiplier"].set_text("2.0")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == ["database", "job_scheduling"]
    assert window._current_index == 3

    window._message_widgets["backend"].set_active_id("redis")
    window._message_widgets["redis_url"].set_text("redis://localhost:6379/0")
    window._message_widgets["stream_prefix"].set_text("atlas")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
    ]
    assert window._current_index == 4

    window._kv_widgets["reuse"].set_active(False)
    window._kv_widgets["url"].set_text("postgresql://kv")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
        "kv_store",
    ]
    assert window._current_index == 5

    window._provider_entries["default_provider"].set_text("openai")
    window._provider_entries["default_model"].set_text("gpt-4o-mini")
    buffer = window._provider_buffer
    assert buffer is not None
    buffer.set_text("openai=sk-test")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
        "kv_store",
        "providers",
    ]
    assert window._current_index == 6

    window._speech_widgets["tts_enabled"].set_active(True)
    window._speech_widgets["stt_enabled"].set_active(True)
    window._speech_widgets["default_tts"].set_text("elevenlabs")
    window._speech_widgets["default_stt"].set_text("openai-whisper")
    window._speech_widgets["elevenlabs_key"].set_text("elevenlabs-key")
    window._speech_widgets["openai_key"].set_text("openai-key")
    window._speech_widgets["google_credentials"].set_text("/path/to/google.json")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
        "kv_store",
        "providers",
        "speech",
    ]
    assert window._current_index == 7

    window._optional_widgets["tenant_id"].set_text("tenant-123")
    window._optional_widgets["retention_days"].set_text("30")
    window._optional_widgets["retention_history_limit"].set_text("200")
    window._optional_widgets["scheduler_timezone"].set_text("UTC")
    window._optional_widgets["scheduler_queue_size"].set_text("10")
    window._optional_widgets["http_auto_start"].set_active(True)

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
        "kv_store",
        "providers",
        "speech",
        "optional",
    ]
    assert window._current_index == 7

    assert on_success.calls
    assert marker_calls == [controller.summary]
    assert len(controller.register_calls) == 1
    assert (
        window._status_label.get_text()
        == "Optional settings saved. Administrator account created. You can now sign in with the new administrator account."
    )

    window.close()


def test_setup_wizard_validation_error_surfaces_in_ui():
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _complete_user_step(window)

    window._database_entries["port"].set_text("invalid")

    window._on_next_clicked(None)

    assert window._current_index == 1
    assert on_success.calls == []
    assert on_error.calls
    assert "Port must be a valid integer" in window._status_label.get_text()
    if hasattr(window._status_label, "get_css_classes"):
        assert "error-text" in window._status_label.get_css_classes()

    window.close()


def test_setup_wizard_job_scheduling_validation():
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _complete_user_step(window)
    _complete_database_step(window)
    assert window._current_index == 2

    window._job_widgets["enabled"].set_active(True)
    window._job_widgets["retry_max_attempts"].set_text("not-a-number")

    window._on_next_clicked(None)

    assert window._current_index == 2
    assert on_success.calls == []
    assert on_error.calls
    assert "Max attempts must be an integer" in window._status_label.get_text()

    window.close()


def test_setup_wizard_requests_privileged_credentials(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _complete_user_step(window)
    window._database_entries["host"].set_text("db.example.com")
    window._database_entries["port"].set_text("5432")
    window._database_entries["database"].set_text("atlas")
    window._database_entries["user"].set_text("atlas")
    window._database_entries["password"].set_text("secret")

    calls = []

    def _apply_with_privileged(state, *, privileged_credentials=None):
        calls.append(privileged_credentials)
        controller.calls.append(("database", state, privileged_credentials))
        if privileged_credentials != ("postgres", "supersecret"):
            raise BootstrapError("superuser access required")
        controller.state.database = state
        return "dsn"

    controller.apply_database_settings = _apply_with_privileged

    prompts = []

    def _fake_prompt(self, *, existing=None, error=None):
        prompts.append((existing, str(error)))
        return ("postgres", "supersecret")

    window._prompt_privileged_credentials = types.MethodType(_fake_prompt, window)

    window._on_next_clicked(None)

    assert calls == [None, ("postgres", "supersecret")]
    assert window._privileged_credentials == ("postgres", "supersecret")
    assert window._current_index == 2
    assert prompts == [(None, "superuser access required")]
    assert not on_error.calls

    window.close()


def test_setup_wizard_optional_settings_validation():
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _complete_user_step(window)
    _complete_database_step(window)
    _complete_job_step(window)
    _complete_message_bus_step(window)
    _complete_kv_step(window)
    _complete_providers_step(window)
    _complete_speech_step(window)

    assert window._current_index == 7

    window._optional_widgets["retention_days"].set_text("invalid")

    window._on_next_clicked(None)

    assert window._current_index == 7
    assert on_success.calls == []
    assert on_error.calls
    assert "Conversation retention days must be an integer" in window._status_label.get_text()

    window.close()


def test_user_step_validation_and_domain_normalization():
    application = Gtk.Application()
    controller = FakeController()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
        controller=controller,
    )

    _populate_user_entries(window, confirm_password="different")
    window._on_next_clicked(None)
    assert window._current_index == 0
    assert "Passwords do not match" in window._status_label.get_text()

    _populate_user_entries(window, confirm_password="changeme", date_of_birth="01-01-1990")
    window._on_next_clicked(None)
    assert window._current_index == 0
    assert "Date of birth must use YYYY-MM-DD format" in window._status_label.get_text()

    _populate_user_entries(
        window,
        date_of_birth="1990-01-01",
        confirm_sudo_password="Mismatch",
    )
    window._on_next_clicked(None)
    assert window._current_index == 0
    assert "Sudo passwords do not match" in window._status_label.get_text()

    _complete_user_step(window, domain=" @MyOrg.COM ")
    assert window._current_index == 1
    assert controller.state.user.domain == "myorg.com"
    tenant_widget = window._optional_widgets["tenant_id"]
    assert isinstance(tenant_widget, Gtk.Entry)
    assert tenant_widget.get_text() == "myorg.com"

    window.close()
