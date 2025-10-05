import asyncio
from types import SimpleNamespace

import pytest

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are registered
from GTKUI.UserAccounts.account_dialog import AccountDialog


class _AtlasStub:
    def __init__(self):
        self.logger = SimpleNamespace(error=lambda *args, **kwargs: None)
        self.active_username = None
        self.active_display_name = "Guest"
        self.last_factory = None
        self.last_success = None
        self.last_error = None
        self.last_thread_name = None
        self.login_result = True
        self.register_result = {"username": "new-user"}
        self.logout_called = False
        self.login_called = None
        self.register_called = None
        self.config_manager = SimpleNamespace(get_active_user=lambda: self.active_username)

    def add_active_user_change_listener(self, listener):
        self.active_listener = listener
        listener(self.active_username or "", self.active_display_name)

    def remove_active_user_change_listener(self, listener):
        if getattr(self, "active_listener", None) == listener:
            self.active_listener = None

    def run_in_background(self, factory, *, on_success=None, on_error=None, thread_name=None):
        self.last_factory = factory
        self.last_success = on_success
        self.last_error = on_error
        self.last_thread_name = thread_name
        return SimpleNamespace()

    async def login_user_account(self, username, password):
        self.login_called = (username, password)
        return self.login_result

    async def register_user_account(self, username, password, email, name, dob):
        self.register_called = (username, password, email, name, dob)
        return self.register_result

    async def logout_active_user(self):
        self.logout_called = True
        return None

    def get_user_display_name(self):
        return self.active_display_name


@pytest.fixture(autouse=True)
def disable_css(monkeypatch):
    monkeypatch.setattr("GTKUI.UserAccounts.account_dialog.apply_css", lambda: None)


def _drain_background(atlas: _AtlasStub):
    assert atlas.last_factory is not None
    coro = atlas.last_factory()
    result = asyncio.get_event_loop().run_until_complete(coro)
    if atlas.last_success is not None:
        atlas.last_success(result)
    return result


def test_login_uses_background_worker():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("secret")

    dialog._on_login_clicked(dialog.login_button)

    assert atlas.last_thread_name == "user-login"
    result = _drain_background(atlas)
    assert result is True
    assert atlas.login_called == ("alice", "secret")
    assert getattr(dialog, "closed", False)


def test_login_failure_displays_error():
    atlas = _AtlasStub()
    atlas.login_result = False
    dialog = AccountDialog(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("bad")

    dialog._on_login_clicked(dialog.login_button)
    _drain_background(atlas)

    assert dialog.login_feedback_label.get_text() == "Invalid username or password."
    assert not getattr(dialog, "closed", False)


def test_register_validates_and_submits():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("pw123")
    dialog.register_confirm_entry.set_text("pw123")
    dialog.register_name_entry.set_text("Test User")
    dialog.register_dob_entry.set_text("2000-01-01")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_thread_name == "user-register"
    _drain_background(atlas)
    assert atlas.register_called == (
        "newuser",
        "pw123",
        "user@example.com",
        "Test User",
        "2000-01-01",
    )
    assert getattr(dialog, "closed", False)


def test_register_rejects_mismatched_passwords():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("pw123")
    dialog.register_confirm_entry.set_text("other")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert dialog.register_feedback_label.get_text() == "Passwords do not match."


def test_logout_invokes_background_worker():
    atlas = _AtlasStub()
    atlas.active_username = "alice"
    dialog = AccountDialog(atlas)

    dialog._on_logout_clicked(dialog.logout_button)

    assert atlas.last_thread_name == "user-logout"
    _drain_background(atlas)
    assert atlas.logout_called is True
    assert dialog.status_label.get_text() == "Signed out."
