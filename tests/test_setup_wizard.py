import types

import pytest

gi = pytest.importorskip("gi")

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Setup.setup_wizard import SetupWizardWindow
from ATLAS.setup import DatabaseState, ProviderState, UserState


class FakeController:
    def __init__(self):
        self.state = types.SimpleNamespace(
            database=DatabaseState(),
            providers=ProviderState(),
            user=UserState(),
        )
        self.calls = []
        self.summary = {"status": "ok"}

    def apply_database_settings(self, state):
        self.calls.append(("database", state))
        self.state.database = state
        return "dsn"

    def apply_provider_settings(self, state):
        self.calls.append(("providers", state))
        self.state.providers = state
        return state

    def register_user(self, state):
        self.calls.append(("user", state))
        self.state.user = state
        return {"username": state.username}

    def build_summary(self):
        return self.summary


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

    window._database_entries["host"].set_text("db.example.com")
    window._database_entries["port"].set_text("5433")
    window._database_entries["database"].set_text("atlas_test")
    window._database_entries["user"].set_text("atlas_user")
    window._database_entries["password"].set_text("secret")

    window._on_next_clicked(None)

    assert controller.calls[0][0] == "database"
    assert window._current_index == 1
    assert not on_error.calls

    window._provider_entries["default_provider"].set_text("openai")
    window._provider_entries["default_model"].set_text("gpt-4o-mini")
    buffer = window._provider_buffer
    assert buffer is not None
    buffer.set_text("openai=sk-test")

    window._on_next_clicked(None)

    assert controller.calls[1][0] == "providers"
    assert window._current_index == 2

    window._user_entries["username"].set_text("admin")
    window._user_entries["email"].set_text("admin@example.com")
    window._user_entries["display_name"].set_text("Atlas Admin")
    window._user_entries["password"].set_text("changeme")
    window._user_entries["confirm_password"].set_text("changeme")

    window._on_next_clicked(None)

    assert controller.calls[2][0] == "user"
    assert on_success.calls
    assert marker_calls == [controller.summary]
    assert (
        window._status_label.get_text()
        == "Administrator account created. You can now sign in with the new administrator account."
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

    window._database_entries["port"].set_text("invalid")

    window._on_next_clicked(None)

    assert window._current_index == 0
    assert on_success.calls == []
    assert on_error.calls
    assert "Port must be a valid integer" in window._status_label.get_text()
    if hasattr(window._status_label, "get_css_classes"):
        assert "error-text" in window._status_label.get_css_classes()

    window.close()
