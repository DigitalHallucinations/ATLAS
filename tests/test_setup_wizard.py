import types

import pytest

gi = pytest.importorskip("gi")

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk

from GTKUI.Setup.setup_wizard import SetupWizardWindow
from ATLAS.setup import (
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    VectorStoreState,
    OptionalState,
    PrivilegedCredentialState,
    ProviderState,
    SetupTypeState,
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
            vector_store=VectorStoreState(),
            kv_store=KvStoreState(),
            providers=ProviderState(),
            speech=SpeechState(),
            optional=OptionalState(),
            user=UserState(),
            setup_type=SetupTypeState(),
        )
        self.calls = []
        self.profile_calls = []
        self.register_calls = []
        self.summary = {"status": "ok"}
        self._staged_privileged_credentials = None
        self.setup_type_calls = []
        self.next_import_host = "imported-host"

    def apply_setup_type(self, mode):
        normalized = (mode or "").strip().lower()
        self.setup_type_calls.append(normalized)
        if normalized not in {"personal", "enterprise"}:
            self.state.setup_type = SetupTypeState(mode=normalized or "custom", applied=False)
            return self.state.setup_type

        current = self.state.setup_type
        if current.mode == normalized and current.applied:
            return current

        if normalized == "personal":
            self.state.message_bus = MessageBusState(backend="in_memory")
            self.state.job_scheduling = JobSchedulingState(enabled=False)
            self.state.kv_store = KvStoreState(reuse_conversation_store=True, url=None)
            self.state.optional = OptionalState(
                tenant_id=self.state.optional.tenant_id,
                retention_days=None,
                retention_history_limit=None,
                scheduler_timezone=self.state.optional.scheduler_timezone,
                scheduler_queue_size=self.state.optional.scheduler_queue_size,
                http_auto_start=True,
            )
        else:
            self.state.message_bus = MessageBusState(
                backend="redis", redis_url=self.state.message_bus.redis_url or "redis://localhost:6379/0", stream_prefix=self.state.message_bus.stream_prefix or "atlas"
            )
            self.state.job_scheduling = JobSchedulingState(
                enabled=True,
                job_store_url=self.state.job_scheduling.job_store_url
                or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_jobs",
                max_workers=self.state.job_scheduling.max_workers or 4,
                retry_policy=self.state.job_scheduling.retry_policy,
                timezone=self.state.job_scheduling.timezone or "UTC",
                queue_size=self.state.job_scheduling.queue_size or 100,
            )
            self.state.kv_store = KvStoreState(
                reuse_conversation_store=False,
                url=self.state.kv_store.url or "postgresql+psycopg://atlas:atlas@localhost:5432/atlas_cache",
            )
            self.state.optional = OptionalState(
                tenant_id=self.state.optional.tenant_id,
                retention_days=30,
                retention_history_limit=500,
                scheduler_timezone=self.state.optional.scheduler_timezone or "UTC",
                scheduler_queue_size=self.state.optional.scheduler_queue_size or 100,
                http_auto_start=False,
            )

        self.state.setup_type = SetupTypeState(mode=normalized, applied=True)
        return self.state.setup_type

    def apply_database_settings(self, state):
        self.calls.append(("database", state))
        self.state.database = state
        return "dsn"

    def apply_vector_store_settings(self, state):
        self.calls.append(("vector_store", state))
        self.state.vector_store = state
        return {"default_adapter": state.adapter}

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

    def export_config(self, path):
        self.calls.append(("export_config", path))
        return path

    def import_config(self, path):
        self.calls.append(("import_config", path))
        self.state.database = DatabaseState(host=self.next_import_host)
        return path


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


def test_provider_keys_masked_by_default_and_toggle():
    application = Gtk.Application()
    controller = FakeController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    buffer = window._provider_buffer
    assert buffer is not None
    buffer.set_text("openai=sk-test\nmistral = token-123")

    mask_buffer = window._provider_mask_buffer
    assert mask_buffer is not None
    masked_text = mask_buffer.get_text(
        mask_buffer.get_start_iter(), mask_buffer.get_end_iter(), True
    )
    assert "sk-test" not in masked_text
    assert "token-123" not in masked_text
    assert "openai" in masked_text
    assert "mistral" in masked_text

    stack = window._provider_stack
    assert stack is not None
    assert stack.get_visible_child_name() == "masked"

    toggle = window._provider_show_toggle
    assert isinstance(toggle, Gtk.CheckButton)
    toggle.set_active(True)
    assert stack.get_visible_child_name() == "visible"
    toggle.set_active(False)
    assert stack.get_visible_child_name() == "masked"

    window.close()


def test_export_config_menu_action_triggers_controller(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    chosen = "/tmp/export-config.yaml"
    monkeypatch.setattr(window, "_choose_export_path", lambda: chosen)

    window._on_export_config_action(None, None)

    assert ("export_config", chosen) in controller.calls
    assert window._toast_history and window._toast_history[-1][0] == "success"

    window.close()


def test_import_config_menu_action_rebuilds_steps(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()
    controller.next_import_host = "imported-db"

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    monkeypatch.setattr(window, "_choose_import_path", lambda: "/tmp/import-config.yaml")

    window._completed_steps.update({0, 1})
    window._on_import_config_action(None, None)

    assert ("import_config", "/tmp/import-config.yaml") in controller.calls
    assert window._database_entries["host"].get_text() == "imported-db"
    assert window._current_index == 0
    assert not window._completed_steps
    assert window._toast_history and window._toast_history[-1][0] == "success"

    window.close()


def test_import_config_menu_action_handles_error(monkeypatch):
    application = Gtk.Application()

    class ErrorController(FakeController):
        def import_config(self, path):
            raise ValueError("invalid YAML content")

    controller = ErrorController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    monkeypatch.setattr(window, "_choose_import_path", lambda: "/tmp/import-config.yaml")

    window._on_import_config_action(None, None)

    assert any("invalid YAML" in message for kind, message in window._toast_history if kind == "error")
    assert "invalid YAML" in window._status_label.get_text()

    window.close()


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


def _advance_intro(window):
    window._on_next_clicked(None)


def _complete_setup_type_step(window, mode="personal"):
    if window._current_index == 0:
        _advance_intro(window)
    button = window._setup_type_buttons.get(mode)
    assert isinstance(button, Gtk.CheckButton)
    button.set_active(True)
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

    target_step = window._steps[5]
    window._stack.set_visible_child(target_step.widget)

    assert window._current_index == 5

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == ["message_bus"]
    assert window._current_index == 6
def test_setup_type_headers_highlight_active_mode():
    application = Gtk.Application()
    controller = FakeController()
    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    personal_label = window._setup_type_headers.get("personal")
    enterprise_label = window._setup_type_headers.get("enterprise")

    assert isinstance(personal_label, Gtk.Label)
    assert isinstance(enterprise_label, Gtk.Label)

    if not hasattr(personal_label, "get_css_classes"):
        window.close()
        pytest.skip("GTK labels do not expose CSS classes")

    def _is_highlighted(label: Gtk.Label) -> bool:
        return any(cls in {"accent", "primary"} for cls in label.get_css_classes())

    assert _is_highlighted(personal_label)
    assert not _is_highlighted(enterprise_label)

    enterprise_button = window._setup_type_buttons.get("enterprise")
    assert isinstance(enterprise_button, Gtk.CheckButton)
    enterprise_button.set_active(True)

    assert _is_highlighted(enterprise_label)
    assert not _is_highlighted(personal_label)

    window.close()


class _CallbackRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def test_setup_wizard_debug_button_opens_log_window(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    debug_button = window._log_toggle_button
    assert isinstance(debug_button, Gtk.Button)

    tooltip_getter = getattr(debug_button, "get_tooltip_text", None)
    if callable(tooltip_getter):
        assert tooltip_getter() == "Show setup logs"

    ensure_calls: list[SetupWizardWindow] = []

    def _fake_ensure(self: SetupWizardWindow) -> types.SimpleNamespace:
        ensure_calls.append(self)
        stub_window = types.SimpleNamespace(present=lambda: None)
        self._set_log_button_active(True)
        self._log_window = stub_window
        return stub_window

    monkeypatch.setattr(SetupWizardWindow, "_ensure_log_window", _fake_ensure)

    emit = getattr(debug_button, "emit", None)
    if callable(emit):
        emit("clicked")
    else:
        window._on_log_button_clicked(debug_button)

    assert ensure_calls == [window]

    window.close()


def test_setup_wizard_alt_arrow_shortcuts(monkeypatch):
    application = Gtk.Application()
    controller = FakeController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    calls: list[str] = []

    def _record_back(*_args: object) -> None:
        calls.append("back")

    def _record_next(*_args: object) -> None:
        calls.append("next")

    monkeypatch.setattr(window, "_on_back_clicked", _record_back)
    monkeypatch.setattr(window, "_on_next_clicked", _record_next)

    alt_mask = getattr(Gdk.ModifierType, "ALT_MASK", getattr(Gdk.ModifierType, "MOD1_MASK", 0))

    assert window._on_window_key_pressed(None, Gdk.KEY_Left, 0, alt_mask) is True
    assert window._on_window_key_pressed(None, Gdk.KEY_Right, 0, alt_mask) is True
    assert window._on_window_key_pressed(None, Gdk.KEY_Right, 0, 0) is False
    assert calls == ["back", "next"]

    window.close()


def test_setup_wizard_inline_validation_feedback():
    application = Gtk.Application()
    controller = FakeController()

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    username_entry = window._user_entries["username"]
    username_entry.set_text("")
    window._run_validation(username_entry)
    tooltip_getter = getattr(username_entry, "get_tooltip_text", None)
    if callable(tooltip_getter):
        assert tooltip_getter() == "Username is required"

    retry_entry = window._job_widgets["retry_max_attempts"]
    assert isinstance(retry_entry, Gtk.Entry)
    retry_entry.set_text("invalid")
    window._run_validation(retry_entry)
    retry_tooltip_getter = getattr(retry_entry, "get_tooltip_text", None)
    if callable(retry_tooltip_getter):
        assert retry_tooltip_getter() == "Max attempts must be a whole number"

    retry_entry.set_text("5")
    window._run_validation(retry_entry)
    if callable(retry_tooltip_getter):
        assert retry_tooltip_getter() in {None, ""}

    window.close()


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

    _complete_setup_type_step(window)

    assert controller.setup_type_calls
    assert window._current_index == 2

    _complete_user_step(window)

    assert controller.profile_calls
    assert controller.register_calls == []
    assert controller.calls == []
    assert window._current_index == 3
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
    assert window._current_index == 4
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
    assert window._current_index == 5

    window._message_widgets["backend"].set_active_id("redis")
    window._message_widgets["redis_url"].set_text("redis://localhost:6379/0")
    window._message_widgets["stream_prefix"].set_text("atlas")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
    ]
    assert window._current_index == 6

    window._kv_widgets["reuse"].set_active(False)
    window._kv_widgets["url"].set_text("postgresql://kv")

    window._on_next_clicked(None)

    assert [name for name, *_ in controller.calls] == [
        "database",
        "job_scheduling",
        "message_bus",
        "kv_store",
    ]
    assert window._current_index == 7

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
    assert window._current_index == 8

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
    assert window._current_index == 8

    assert on_success.calls
    assert marker_calls == [controller.summary]
    assert len(controller.register_calls) == 1
    assert (
        window._status_label.get_text()
        == "Speech settings saved. Administrator account created. You can now sign in with the new administrator account."
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

    _complete_setup_type_step(window)

    _complete_user_step(window)

    window._database_entries["port"].set_text("invalid")

    window._on_next_clicked(None)

    assert window._current_index == 3
    assert on_success.calls == []
    assert on_error.calls
    assert "Port must be a valid integer" in window._status_label.get_text()
    if hasattr(window._status_label, "get_css_classes"):
        assert "error-text" in window._status_label.get_css_classes()

    window.close()


def test_setup_type_hint_tracks_selection():
    application = Gtk.Application()
    controller = FakeController()
    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=lambda: None,
        on_error=lambda exc: None,
        controller=controller,
    )

    hint = window._optional_personal_hint
    assert isinstance(hint, Gtk.Label)
    assert not hint.get_visible()

    _complete_setup_type_step(window, mode="personal")
    assert hint.get_visible()

    enterprise_button = window._setup_type_buttons.get("enterprise")
    assert isinstance(enterprise_button, Gtk.CheckButton)
    enterprise_button.set_active(True)

    assert not hint.get_visible()

    window.close()


def test_setup_wizard_reuses_stored_sudo_password_for_bootstrap(monkeypatch):
    application = Gtk.Application()
    on_success = _CallbackRecorder()
    on_error = _CallbackRecorder()

    class RecordingController(FakeController):
        def __init__(self, *, atlas=None, request_privileged_password=None):
            super().__init__()
            self.request_privileged_password = request_privileged_password
            self.bootstrap_calls = []

        def apply_database_settings(self, state, privileged_credentials=None):
            self.calls.append(("database", state))
            password = None
            if self.request_privileged_password is not None:
                password = self.request_privileged_password()
            self.bootstrap_calls.append(password)
            self.state.database = state
            return "dsn"

    monkeypatch.setattr(
        "GTKUI.Setup.setup_wizard.CoreSetupWizardController",
        RecordingController,
    )

    window = SetupWizardWindow(
        application=application,
        atlas=None,
        on_success=on_success,
        on_error=on_error,
    )

    assert isinstance(window.controller, RecordingController)
    assert window.controller.request_privileged_password is window._request_sudo_password

    _complete_setup_type_step(window)
    _complete_user_step(window)

    _complete_database_step(window)

    assert window.controller.bootstrap_calls == ["SudoPass!"]
    assert window._current_index == 4

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

    _complete_setup_type_step(window)
    _complete_user_step(window)
    _complete_database_step(window)
    assert window._current_index == 4

    window._job_widgets["enabled"].set_active(True)
    window._job_widgets["retry_max_attempts"].set_text("not-a-number")

    window._on_next_clicked(None)

    assert window._current_index == 4
    assert on_success.calls == []
    assert on_error.calls
    assert "Max attempts must be an integer" in window._status_label.get_text()

    window.close()


def test_privileged_credentials_dialog_footer_buttons():
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

    dialog, _username_entry, _password_entry, apply_button, cancel_button = (
        window._create_privileged_credentials_dialog(
            existing=None,
            error=BootstrapError("superuser access required"),
        )
    )

    try:
        responses: list[int] = []
        dialog.connect("response", lambda _dlg, resp: responses.append(resp))

        assert isinstance(apply_button, Gtk.Button)
        assert isinstance(cancel_button, Gtk.Button)
        assert apply_button.get_label() == "Apply"
        assert cancel_button.get_label() == "Cancel"

        apply_button.emit("clicked")
        assert responses[-1] == Gtk.ResponseType.OK

        responses.clear()
        dialog.activate_default()
        assert responses[-1] == Gtk.ResponseType.OK

        responses.clear()
        cancel_button.emit("clicked")
        assert responses[-1] == Gtk.ResponseType.CANCEL
    finally:
        dialog.destroy()
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

    _complete_setup_type_step(window)
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
    assert window._current_index == 4
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

    _complete_setup_type_step(window, mode="enterprise")

    window._on_next_clicked(None)

    assert "Tenant ID is required" in window._status_label.get_text()

    window._optional_widgets["tenant_id"].set_text("enterprise.example")
    window._optional_widgets["retention_days"].set_text("invalid")
    window._on_next_clicked(None)

    _populate_user_entries(window)
    window._on_next_clicked(None)

    assert window._current_index == 2
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

    _complete_setup_type_step(window)

    _populate_user_entries(window, confirm_password="different")
    window._on_next_clicked(None)
    assert window._current_index == 2
    assert "Passwords do not match" in window._status_label.get_text()

    _populate_user_entries(window, confirm_password="changeme", date_of_birth="01-01-1990")
    window._on_next_clicked(None)
    assert window._current_index == 2
    assert "Date of birth must use YYYY-MM-DD format" in window._status_label.get_text()

    _populate_user_entries(
        window,
        date_of_birth="1990-01-01",
        confirm_sudo_password="Mismatch",
    )
    window._on_next_clicked(None)
    assert window._current_index == 2
    assert "Sudo passwords do not match" in window._status_label.get_text()

    _complete_user_step(window, domain=" @MyOrg.COM ")
    assert window._current_index == 3
    assert controller.state.user.domain == "myorg.com"
    tenant_widget = window._optional_widgets["tenant_id"]
    assert isinstance(tenant_widget, Gtk.Entry)
    assert tenant_widget.get_text() == "myorg.com"

    window.close()
