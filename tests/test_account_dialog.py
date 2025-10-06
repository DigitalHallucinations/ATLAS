import asyncio
import datetime as _dt
from typing import Optional
from types import MethodType, SimpleNamespace

import pytest

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are registered
from gi.repository import GLib, Gtk
from GTKUI.UserAccounts.account_dialog import AccountDialog
from modules.user_accounts.user_account_service import (
    AccountLockedError,
    DuplicateUserError,
    InvalidCurrentPasswordError,
    PasswordRequirements,
)
from tests.test_chat_async_helper import make_alert_dialog_future


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
        self.login_error = None
        self.register_result = {"username": "new-user"}
        self.logout_called = False
        self.login_called = None
        self.register_called = None
        self.list_accounts_result = []
        self.list_calls = 0
        self.search_accounts_result = []
        self.search_calls: list[str] = []
        self.search_error: Optional[Exception] = None
        self.activate_called = None
        self.delete_called = None
        self.details_called = None
        self.details_result = {}
        self.update_called = None
        self.update_result = {"username": "updated-user"}
        self.config_manager = SimpleNamespace(get_active_user=lambda: self.active_username)
        self._password_requirements = PasswordRequirements(
            min_length=10,
            require_uppercase=True,
            require_lowercase=True,
            require_digit=True,
            require_symbol=True,
            forbid_whitespace=True,
        )
        self._password_requirements_description: Optional[str] = None

    def set_password_requirements(
        self,
        requirements: PasswordRequirements,
        description: Optional[str] = None,
    ) -> None:
        self._password_requirements = requirements
        self._password_requirements_description = description

    def get_user_password_requirements(self):
        return self._password_requirements

    def describe_user_password_requirements(self):
        if self._password_requirements_description is not None:
            return self._password_requirements_description
        return self.get_user_password_requirements().describe()

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
        if self.login_error is not None:
            raise self.login_error
        return self.login_result

    async def register_user_account(self, username, password, email, name, dob):
        self.register_called = (username, password, email, name, dob)
        return self.register_result

    async def update_user_account(
        self,
        username,
        *,
        password=None,
        current_password=None,
        email=None,
        name=None,
        dob=None,
    ):
        self.update_called = {
            "username": username,
            "password": password,
            "current_password": current_password,
            "email": email,
            "name": name,
            "dob": dob,
        }
        return self.update_result

    async def logout_active_user(self):
        self.logout_called = True
        return None

    async def list_user_accounts(self):
        self.list_calls += 1
        return list(self.list_accounts_result)

    async def search_user_accounts(self, query_text: str):
        self.search_calls.append(query_text)
        if self.search_error is not None:
            raise self.search_error
        return list(self.search_accounts_result)

    async def activate_user_account(self, username):
        self.activate_called = username
        self.active_username = username
        if getattr(self, "active_listener", None):
            self.active_listener(username, username)
        return None

    async def delete_user_account(self, username):
        self.delete_called = username
        return None

    async def get_user_account_details(self, username):
        self.details_called = username
        return self.details_result.get(username)

    def get_user_display_name(self):
        return self.active_display_name


@pytest.fixture(autouse=True)
def disable_css(monkeypatch):
    monkeypatch.setattr("GTKUI.UserAccounts.account_dialog.apply_css", lambda: None)


def _drain_background(atlas: _AtlasStub):
    assert atlas.last_factory is not None
    factory = atlas.last_factory
    atlas.last_factory = None
    try:
        result_or_coro = factory()
        if asyncio.iscoroutine(result_or_coro):
            result = asyncio.run(result_or_coro)
        else:
            result = result_or_coro
    except Exception as exc:  # noqa: BLE001 - mimic background worker behaviour
        if atlas.last_error is not None:
            atlas.last_error(exc)
        return None

    if atlas.last_success is not None:
        atlas.last_success(result)
    return result


def _click(button):
    for signal, callback in getattr(button, "_callbacks", []):
        if signal == "clicked":
            callback(button)
            break


def _get_toggle_state(toggle):
    getter = getattr(toggle, "get_active", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:  # pragma: no cover - fallback for stub widgets
            pass
    return getattr(toggle, "_atlas_toggle_active", False)


def _is_sensitive(widget):
    if hasattr(widget, "_sensitive"):
        return bool(getattr(widget, "_sensitive"))
    getter = getattr(widget, "get_sensitive", None)
    if callable(getter):
        try:
            return bool(getter())
        except Exception:  # pragma: no cover - fallback for stub widgets
            pass
    return getattr(widget, "_sensitive", True)


def test_form_toggle_highlighting():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    assert dialog.login_toggle_button.has_css_class("suggested-action")
    assert not dialog.register_toggle_button.has_css_class("suggested-action")

    dialog._show_form("register")
    assert dialog.register_toggle_button.has_css_class("suggested-action")
    assert not dialog.login_toggle_button.has_css_class("suggested-action")

    dialog._show_form("edit")
    assert not dialog.login_toggle_button.has_css_class("suggested-action")
    assert not dialog.register_toggle_button.has_css_class("suggested-action")


def test_password_requirements_rendered():
    atlas = _AtlasStub()
    custom_requirements = PasswordRequirements(
        min_length=16,
        require_uppercase=False,
        require_lowercase=True,
        require_digit=True,
        require_symbol=True,
        forbid_whitespace=False,
    )
    custom_description = (
        "Passwords must be at least 16 characters long, include a number and symbol."
    )
    atlas.set_password_requirements(custom_requirements, custom_description)
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    expected_tooltip = atlas.describe_user_password_requirements()
    expected_label_text = "\n".join(
        f"• {line}" for line in custom_requirements.bullet_points()
    )

    assert dialog.register_password_requirements_label.get_text() == expected_label_text
    assert dialog.edit_password_requirements_label.get_text() == expected_label_text

    widgets = [
        dialog.register_password_entry,
        dialog.register_confirm_entry,
        dialog.edit_password_entry,
        dialog.edit_confirm_entry,
    ]

    for widget in widgets:
        tooltip = getattr(widget, "_tooltip", None)
        assert tooltip is not None
        assert tooltip == expected_tooltip

    dialog._password_requirements_text = "Use a memorable passphrase"
    dialog._update_password_requirement_labels()

    for widget in widgets:
        assert getattr(widget, "_tooltip", None) == "Use a memorable passphrase"


def test_password_requirements_refresh_updates_labels():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    initial_tooltip = getattr(dialog.register_password_entry, "_tooltip", None)
    assert initial_tooltip is not None

    updated_requirements = PasswordRequirements(
        min_length=8,
        require_uppercase=True,
        require_lowercase=True,
        require_digit=False,
        require_symbol=False,
        forbid_whitespace=True,
    )
    updated_description = "Passwords must be at least 8 characters and avoid spaces."
    atlas.set_password_requirements(updated_requirements, updated_description)

    dialog._initialise_password_requirements()

    expected_label_text = "\n".join(
        f"• {line}" for line in updated_requirements.bullet_points()
    )
    assert dialog.register_password_requirements_label.get_text() == expected_label_text
    assert dialog.edit_password_requirements_label.get_text() == expected_label_text

    refreshed_tooltip = getattr(dialog.register_password_entry, "_tooltip", None)
    assert refreshed_tooltip == updated_description
    assert refreshed_tooltip != initial_tooltip


def test_password_toggle_icon_press_updates_state_and_label():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    entry = dialog.login_password_entry
    toggles = dialog._password_toggle_buttons_by_entry.get(entry)
    assert toggles, "Password toggle should be registered for the login entry"
    toggle = toggles[0]

    initial_label = getattr(toggle, "label", "")
    assert initial_label == "Show"
    assert _get_toggle_state(toggle) is False

    dialog._on_password_icon_pressed(entry, None, None)
    assert entry.visible is True
    assert getattr(toggle, "label", "") == "Hide"
    assert _get_toggle_state(toggle) is True

    dialog._on_password_icon_pressed(entry, None, None)
    assert entry.visible is False
    assert getattr(toggle, "label", "") == "Show"
    assert _get_toggle_state(toggle) is False


def test_login_uses_background_worker():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("secret")

    dialog._on_login_clicked(dialog.login_button)

    assert atlas.last_thread_name == "user-login"
    result = _drain_background(atlas)
    assert result is True
    assert atlas.login_called == ("alice", "secret")
    assert getattr(dialog, "closed", False)


def test_login_entry_activate_triggers_login(monkeypatch):
    atlas = _AtlasStub()
    activate_callbacks = []

    original_connect = Gtk.Entry.connect

    def recording_connect(self, signal, callback, *args):
        if signal == "activate":
            activate_callbacks.append((self, callback, args))
        return original_connect(self, signal, callback, *args)

    monkeypatch.setattr(Gtk.Entry, "connect", recording_connect, raising=False)

    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("secret")

    for widget, callback, args in activate_callbacks:
        if widget is dialog.login_password_entry:
            callback(widget, *args)
            break
    else:  # pragma: no cover - defensive assertion aid
        pytest.fail("Password entry did not register an activate callback")

    assert atlas.last_thread_name == "user-login"
    _drain_background(atlas)
    assert atlas.login_called == ("alice", "secret")


def test_login_entry_activate_ignored_when_busy(monkeypatch):
    atlas = _AtlasStub()
    activate_callbacks = []

    original_connect = Gtk.Entry.connect

    def recording_connect(self, signal, callback, *args):
        if signal == "activate":
            activate_callbacks.append((self, callback, args))
        return original_connect(self, signal, callback, *args)

    monkeypatch.setattr(Gtk.Entry, "connect", recording_connect, raising=False)

    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("secret")

    # Trigger the first login attempt and capture the factory used.
    dialog._on_login_clicked(dialog.login_button)
    first_factory = atlas.last_factory

    assert dialog._login_busy is True
    assert getattr(dialog.login_username_entry, "_sensitive", True) is False
    assert getattr(dialog.login_password_entry, "_sensitive", True) is False

    # Attempt to trigger login again via the activate callback while busy.
    for widget, callback, args in activate_callbacks:
        if widget is dialog.login_password_entry:
            callback(widget, *args)
            break
    else:  # pragma: no cover - defensive assertion aid
        pytest.fail("Password entry did not register an activate callback")

    assert atlas.last_factory is first_factory


def test_register_user_validation():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    toggle_widgets = []
    for entry in (dialog.register_password_entry, dialog.register_confirm_entry):
        toggle_widgets.extend(dialog._password_toggle_buttons_by_entry.get(entry, []))

    widgets = [
        dialog.register_button,
        dialog.register_username_entry,
        dialog.register_email_entry,
        dialog.register_password_entry,
        dialog.register_confirm_entry,
        dialog.register_name_entry,
        dialog.register_dob_entry,
        *toggle_widgets,
    ]

    dialog._set_register_busy(True, "Processing…")

    assert dialog._register_busy is True
    assert dialog.register_feedback_label.get_text() == "Processing…"
    disabled_states = [_is_sensitive(widget) for widget in widgets]
    assert all(state is False for state in disabled_states), disabled_states

    dialog._set_register_busy(False)

    assert dialog._register_busy is False
    enabled_states = [_is_sensitive(widget) for widget in widgets]
    assert all(state is True for state in enabled_states), enabled_states


def test_edit_form_disabled_when_busy():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    toggle_widgets = []
    for entry in (
        dialog.edit_current_password_entry,
        dialog.edit_password_entry,
        dialog.edit_confirm_entry,
    ):
        toggle_widgets.extend(dialog._password_toggle_buttons_by_entry.get(entry, []))

    widgets = [
        dialog.edit_save_button,
        dialog.edit_username_entry,
        dialog.edit_email_entry,
        dialog.edit_current_password_entry,
        dialog.edit_password_entry,
        dialog.edit_confirm_entry,
        dialog.edit_name_entry,
        dialog.edit_dob_entry,
        *toggle_widgets,
    ]

    dialog._set_edit_busy(True, "Processing…")

    disabled_states = [_is_sensitive(widget) for widget in widgets]
    assert all(state is False for state in disabled_states), disabled_states

    dialog._set_edit_busy(False)

    enabled_states = [_is_sensitive(widget) for widget in widgets]
    assert all(state is True for state in enabled_states), enabled_states


def test_login_failure_displays_error():
    atlas = _AtlasStub()
    atlas.login_result = False
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("bad")

    dialog._on_login_clicked(dialog.login_button)
    _drain_background(atlas)

    assert dialog.login_feedback_label.get_text() == "Invalid username or password."
    assert not getattr(dialog, "closed", False)


def test_login_lockout_error_displays_message():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("bad")

    atlas.login_error = AccountLockedError(
        "alice",
        retry_at=_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=60),
        retry_after=60,
    )

    dialog._on_login_clicked(dialog.login_button)
    _drain_background(atlas)

    assert (
        dialog.login_feedback_label.get_text()
        == "Too many failed login attempts. Try again in 60 seconds."
    )
    assert not getattr(dialog, "closed", False)


def test_login_lockout_countdown_disables_controls(monkeypatch):
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    timeouts: dict[int, tuple[int, object, tuple[object, ...]]] = {}
    next_id = 1

    def fake_timeout_add_seconds(interval, callback, *args):
        nonlocal next_id
        handle = next_id
        next_id += 1
        timeouts[handle] = (interval, callback, args)
        return handle

    def fake_source_remove(handle):
        timeouts.pop(handle, None)
        return True

    monkeypatch.setattr(GLib, "timeout_add_seconds", fake_timeout_add_seconds, raising=False)
    monkeypatch.setattr(GLib, "source_remove", fake_source_remove, raising=False)

    dialog.login_username_entry.set_text("alice")
    dialog.login_password_entry.set_text("bad")

    atlas.login_error = AccountLockedError("alice", retry_after=3)

    dialog._on_login_clicked(dialog.login_button)
    _drain_background(atlas)

    assert dialog.login_feedback_label.get_text() == (
        "Too many failed login attempts. Try again in 3 seconds."
    )
    assert _is_sensitive(dialog.login_button) is False
    assert _is_sensitive(dialog.login_username_entry) is False
    assert _is_sensitive(dialog.login_password_entry) is False

    timeout_id = dialog._login_lockout_timeout_id
    assert timeout_id in timeouts

    def tick_once():
        interval, callback, args = timeouts[timeout_id]
        keep_running = callback(*args)
        if not keep_running:
            timeouts.pop(timeout_id, None)
        return keep_running

    assert tick_once() is True
    assert dialog.login_feedback_label.get_text() == (
        "Too many failed login attempts. Try again in 2 seconds."
    )
    assert _is_sensitive(dialog.login_button) is False

    assert tick_once() is True
    assert dialog.login_feedback_label.get_text() == (
        "Too many failed login attempts. Try again in 1 second."
    )
    assert _is_sensitive(dialog.login_button) is False

    assert tick_once() is False
    assert dialog.login_feedback_label.get_text() == "You can try signing in again."
    assert _is_sensitive(dialog.login_button) is True
    assert _is_sensitive(dialog.login_username_entry) is True
    assert _is_sensitive(dialog.login_password_entry) is True
    assert dialog._login_lockout_timeout_id is None
    assert dialog._login_lockout_remaining_seconds is None
    assert timeout_id not in timeouts


def test_close_cancels_login_lockout_timer(monkeypatch):
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    timeouts: dict[int, tuple[int, object, tuple[object, ...]]] = {}
    removed: list[int] = []
    next_id = 1

    def fake_timeout_add_seconds(interval, callback, *args):
        nonlocal next_id
        handle = next_id
        next_id += 1
        timeouts[handle] = (interval, callback, args)
        return handle

    def fake_source_remove(handle):
        removed.append(handle)
        timeouts.pop(handle, None)
        return True

    monkeypatch.setattr(GLib, "timeout_add_seconds", fake_timeout_add_seconds, raising=False)
    monkeypatch.setattr(GLib, "source_remove", fake_source_remove, raising=False)

    dialog._start_login_lockout_timer(5)
    timeout_id = dialog._login_lockout_timeout_id
    assert timeout_id is not None
    assert timeout_id in timeouts

    dialog._on_close_request()

    assert dialog._login_lockout_timeout_id is None
    assert dialog._login_lockout_remaining_seconds is None
    assert timeout_id not in timeouts
    assert removed == [timeout_id]


def test_register_validates_and_submits():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")
    dialog.register_name_entry.set_text("Test User")
    dialog.register_dob_entry.set_text("2000-01-01")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_thread_name == "user-register"
    _drain_background(atlas)
    assert atlas.register_called == (
        "newuser",
        "Password1!",
        "user@example.com",
        "Test User",
        "2000-01-01",
    )
    assert getattr(dialog, "closed", False)


def test_refresh_account_list_handles_async_accounts():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)

    # Drain the initial refresh triggered during dialog construction.
    if atlas.last_factory is not None:
        _drain_background(atlas)

    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
    ]

    dialog._refresh_account_list()
    _drain_background(atlas)

    assert atlas.list_calls >= 1
    assert [entry["username"] for entry in dialog._account_records] == ["alice", "bob"]


def test_refresh_account_list_failure_clears_rows_and_details():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
    ]
    atlas.details_result = {
        "alice": {
            "username": "alice",
            "display_name": "Alice",
            "email": "alice@example.com",
            "name": "Alice",  # legacy field fallback
            "dob": "1990-01-01",
            "last_login": "2024-05-20T10:00:00Z",
        }
    }

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    # Display account details to populate the panel with data.
    details_button = dialog._account_rows["alice"]["details_button"]
    _click(details_button)
    _drain_background(atlas)

    assert dialog.account_details_status_label.get_text() == ""
    assert dialog.account_details_username_value.get_text() == "alice"
    assert dialog.account_details_last_login_value.get_text() == "2024-05-20 10:00 UTC"

    async def failing_list(self):
        self.list_calls += 1
        raise RuntimeError("backend unavailable")

    atlas.list_user_accounts = MethodType(failing_list, atlas)

    dialog._refresh_account_list()
    _drain_background(atlas)

    assert dialog._account_rows == {}
    assert dialog._visible_usernames == []
    assert getattr(dialog.account_list_box, "children", []) == []
    assert dialog.account_details_status_label.get_text() == "Select an account to view details."
    placeholder = "—"
    assert dialog.account_details_username_value.get_text() == placeholder
    assert dialog.account_details_name_value.get_text() == placeholder
    assert dialog.account_details_email_value.get_text() == placeholder
    assert dialog.account_details_dob_value.get_text() == placeholder
    assert dialog.account_details_last_login_value.get_text() == placeholder


def test_register_rejects_mismatched_passwords():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("other")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert dialog.register_feedback_label.get_text() == "Passwords do not match."


def test_register_rejects_invalid_email_client_side():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("invalid-email")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert dialog.register_feedback_label.get_text() == "Enter a valid email address."
    assert getattr(dialog.register_email_entry, "_atlas_invalid", False) is True
    assert getattr(dialog.register_password_entry, "_atlas_invalid", False) is False


def test_register_rejects_invalid_username_client_side():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("ab")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert dialog.register_feedback_label.get_text().startswith("Username must be")
    assert getattr(dialog.register_username_entry, "_atlas_invalid", False) is True


def test_register_rejects_weak_password_client_side():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("short1")
    dialog.register_confirm_entry.set_text("short1")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert (
        dialog.register_feedback_label.get_text()
        == "Password must be at least 10 characters long."
    )
    assert getattr(dialog.register_password_entry, "_atlas_invalid", False) is True


def test_register_rejects_invalid_dob_client_side():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")
    dialog.register_dob_entry.set_text("01-01-2000")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_factory is None
    assert dialog.register_feedback_label.get_text() == "Enter date of birth as YYYY-MM-DD."
    assert getattr(dialog.register_dob_entry, "_atlas_invalid", False) is True


def test_register_surfaces_backend_value_errors():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("newuser")
    dialog.register_email_entry.set_text("user@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_error is not None
    atlas.last_error(ValueError("Email must be a valid email address."))

    assert dialog.register_feedback_label.get_text() == "Email must be a valid email address."


def test_register_duplicate_user_marks_fields_invalid():
    atlas = _AtlasStub()
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog.register_username_entry.set_text("existinguser")
    dialog.register_email_entry.set_text("exists@example.com")
    dialog.register_password_entry.set_text("Password1!")
    dialog.register_confirm_entry.set_text("Password1!")

    dialog._on_register_clicked(dialog.register_button)

    assert atlas.last_error is not None
    atlas.last_error(DuplicateUserError("duplicate"))

    assert dialog.register_feedback_label.get_text() == "Username or email already exists."
    assert getattr(dialog.register_username_entry, "_atlas_invalid", False) is True
    assert getattr(dialog.register_email_entry, "_atlas_invalid", False) is True


def test_logout_invokes_background_worker():
    atlas = _AtlasStub()
    atlas.active_username = "alice"
    dialog = AccountDialog(atlas)
    if atlas.last_factory is not None:
        _drain_background(atlas)

    dialog._on_logout_clicked(dialog.logout_button)

    assert atlas.last_thread_name == "user-logout"
    _drain_background(atlas)
    assert atlas.logout_called is True
    assert dialog.status_label.get_text() == "Signed out."


def test_account_list_populates_and_highlights_active():
    atlas = _AtlasStub()
    atlas.active_username = "bob"
    atlas.active_display_name = "Bobby"
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bobby", "last_login": "2024-05-20T10:00:00Z"},
    ]

    dialog = AccountDialog(atlas)
    assert atlas.last_thread_name == "user-account-list"
    _drain_background(atlas)

    assert atlas.list_calls == 1
    assert set(dialog._account_rows.keys()) == {"alice", "bob"}
    assert dialog._account_rows["bob"]["active_label"].get_text() == "Active"
    assert dialog._account_rows["alice"]["active_label"].get_text() == "No sign-in yet"
    assert dialog._account_rows["alice"]["last_login_label"].get_text() == "Last sign-in: never"
    assert (
        dialog._account_rows["bob"]["last_login_label"].get_text()
        == "Last sign-in: 2024-05-20 10:00 UTC"
    )

    dialog._apply_active_user_state("alice", "Alice")

    assert dialog._account_rows["alice"]["active_label"].get_text() == "Active"
    assert dialog._account_rows["alice"]["use_button"]._sensitive is False
    assert dialog._account_rows["bob"]["use_button"]._sensitive is True


def test_use_account_triggers_activation_and_disables_forms():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    dialog._confirm_delete_handler = None

    use_button = dialog._account_rows["alice"]["use_button"]
    assert dialog.login_box._sensitive is True

    _click(use_button)

    assert dialog._forms_busy is True
    assert dialog.login_box._sensitive is False
    assert atlas.last_thread_name == "user-account-activate"

    _drain_background(atlas)

    assert atlas.activate_called == "alice"
    assert dialog.account_feedback_label.get_text() == "Account activated."
    assert dialog._forms_busy is False
    assert dialog._account_rows["alice"]["active_label"].get_text() == "Active"


def test_delete_account_confirms_and_refreshes():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    dialog._confirm_delete_handler = None

    dialog._confirm_delete_handler = lambda username: True
    delete_button = dialog._account_rows["alice"]["delete_button"]

    _click(delete_button)

    assert dialog.account_feedback_label.get_text() == "Deleting alice…"
    assert atlas.last_thread_name == "user-account-delete"

    assert dialog._forms_busy is True
    assert atlas.last_thread_name == "user-account-delete"

    atlas.list_accounts_result = [{"username": "bob", "display_name": "Bob"}]
    _drain_background(atlas)

    assert atlas.delete_called == "alice"
    assert dialog._last_delete_prompt == "alice"
    assert dialog._forms_busy is False
    assert atlas.last_thread_name == "user-account-list"

    if atlas.last_factory is not None:
        _drain_background(atlas)

    assert atlas.list_calls >= 2
    assert dialog.account_feedback_label.get_text().startswith("Account deleted.")
    assert set(dialog._account_rows.keys()) == {"bob"}


def test_account_search_filters_rows():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
        {"username": "carol", "display_name": "Carol"},
    ]
    atlas.search_accounts_result = [
        {"username": "bob", "display_name": "Bob"},
        {"username": "bobbie", "display_name": "Bobbie"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    dialog.account_search_entry.set_text("bob")
    dialog._on_account_filter_changed(dialog.account_search_entry)

    assert atlas.last_thread_name == "user-account-search"
    assert dialog.account_feedback_label.get_text() == "Searching accounts…"
    assert atlas.search_calls == []

    _drain_background(atlas)

    assert atlas.search_calls == ["bob"]
    assert dialog._visible_usernames == ["bob", "bobbie"]
    assert dialog.account_feedback_label.get_text().startswith("Found 2 account")

    dialog.account_search_entry.set_text("")
    dialog._on_account_filter_changed(dialog.account_search_entry)

    assert atlas.last_thread_name == "user-account-list"
    assert dialog.account_feedback_label.get_text() == "Loading saved accounts…"

    _drain_background(atlas)

    assert set(dialog._visible_usernames) == {"alice", "bob", "carol"}


def test_account_details_fetches_and_displays():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
    ]
    atlas.details_result = {
        "alice": {
            "username": "alice",
            "email": "alice@example.com",
            "name": "Alice",
            "dob": "1990-01-01",
            "last_login": "2024-05-20T10:00:00Z",
        }
    }

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    details_button = dialog._account_rows["alice"]["details_button"]
    _click(details_button)

    assert atlas.last_thread_name == "user-account-details"
    _drain_background(atlas)

    assert atlas.details_called == "alice"
    assert dialog.account_details_email_value.get_text() == "alice@example.com"
    assert dialog.account_details_name_value.get_text() == "Alice"
    assert dialog.account_details_last_login_value.get_text() == "2024-05-20 10:00 UTC"
    assert (
        dialog._account_rows["alice"]["last_login_label"].get_text()
        == "Last sign-in: 2024-05-20 10:00 UTC"
    )

    # Cached results should avoid another background call
    atlas.last_thread_name = None
    _click(details_button)
    assert atlas.last_thread_name is None


def test_delete_account_async_confirmation_accepts(monkeypatch):
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
        {"username": "bob", "display_name": "Bob"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)
    dialog._confirm_delete_handler = None

    busy_calls = []
    original_set_busy = dialog._set_account_busy

    def tracking_set_busy(self, busy, message=None, *, disable_forms=True):
        busy_calls.append((busy, message))
        return original_set_busy(busy, message, disable_forms=disable_forms)

    monkeypatch.setattr(
        dialog,
        "_set_account_busy",
        MethodType(tracking_set_busy, dialog),
        raising=False,
    )

    future = make_alert_dialog_future("delete")
    wait_hooks: list[str] = []

    def wait_result():
        assert busy_calls == []
        wait_hooks.append("wait_result")
        return "delete"

    def wait():
        assert busy_calls == []
        wait_hooks.append("wait")
        return "delete"

    future.set_wait_result_hook(wait_result)
    future.set_wait_hook(wait)
    original_alert_dialog = Gtk.AlertDialog

    class FutureAlertDialog(original_alert_dialog):
        instances: list[object] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            FutureAlertDialog.instances.append(self)

        def choose(self, _parent):
            return future

    monkeypatch.setattr(Gtk, "AlertDialog", FutureAlertDialog, raising=False)

    delete_button = dialog._account_rows["alice"]["delete_button"]

    _click(delete_button)

    assert FutureAlertDialog.instances
    assert wait_hooks
    assert future.wait_result_calls + future.wait_calls >= 1
    assert busy_calls and busy_calls[0] == (True, "Deleting alice…")
    assert atlas.last_thread_name == "user-account-delete"
    assert dialog.account_feedback_label.get_text() == "Deleting alice…"

    atlas.list_accounts_result = [{"username": "bob", "display_name": "Bob"}]
    _drain_background(atlas)

    assert atlas.delete_called == "alice"
    assert dialog._forms_busy is False
    assert atlas.last_thread_name == "user-account-list"
    if atlas.last_factory is not None:
        _drain_background(atlas)
    assert dialog.account_feedback_label.get_text().startswith("Account deleted.")


def test_delete_account_async_confirmation_cancels(monkeypatch):
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)
    dialog._confirm_delete_handler = None

    busy_calls = []
    original_set_busy = dialog._set_account_busy

    def tracking_set_busy(self, busy, message=None, *, disable_forms=True):
        busy_calls.append((busy, message))
        return original_set_busy(busy, message, disable_forms=disable_forms)

    monkeypatch.setattr(
        dialog,
        "_set_account_busy",
        MethodType(tracking_set_busy, dialog),
        raising=False,
    )

    future = make_alert_dialog_future("cancel")
    wait_hooks: list[str] = []

    def wait_result():
        assert busy_calls == []
        wait_hooks.append("wait_result")
        return "cancel"

    def wait():
        assert busy_calls == []
        wait_hooks.append("wait")
        return "cancel"

    future.set_wait_result_hook(wait_result)
    future.set_wait_hook(wait)
    original_alert_dialog = Gtk.AlertDialog

    class FutureAlertDialog(original_alert_dialog):
        instances: list[object] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            FutureAlertDialog.instances.append(self)

        def choose(self, _parent):
            return future

    monkeypatch.setattr(Gtk, "AlertDialog", FutureAlertDialog, raising=False)

    delete_button = dialog._account_rows["alice"]["delete_button"]

    _click(delete_button)

    assert FutureAlertDialog.instances
    assert wait_hooks
    assert future.wait_result_calls + future.wait_calls >= 1
    assert busy_calls == []
    assert atlas.delete_called is None
    assert dialog.account_feedback_label.get_text() == "Deletion cancelled."
    assert dialog._forms_busy is False
    assert dialog._last_delete_prompt == "alice"
    assert atlas.last_factory is None


def test_delete_account_manual_dialog_requires_confirmation(monkeypatch):
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice"},
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)
    dialog._confirm_delete_handler = None

    monkeypatch.delattr(Gtk, "AlertDialog", raising=False)

    confirmations: list[tuple[str, bool]] = []
    state = {"allow": False}

    def fake_fallback(self, username):
        confirmations.append((username, state["allow"]))
        return state["allow"]

    monkeypatch.setattr(AccountDialog, "_show_fallback_delete_dialog", fake_fallback, raising=False)

    delete_button = dialog._account_rows["alice"]["delete_button"]

    _click(delete_button)

    assert confirmations == [("alice", False)]
    assert atlas.delete_called is None
    assert dialog.account_feedback_label.get_text() == "Deletion cancelled."

    state["allow"] = True

    _click(delete_button)

    assert confirmations == [("alice", False), ("alice", True)]
    assert atlas.last_thread_name == "user-account-delete"
    assert dialog.account_feedback_label.get_text() == "Deleting alice…"

    atlas.list_accounts_result = []
    _drain_background(atlas)

    assert atlas.delete_called == "alice"


def test_edit_account_updates_fields():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {
            "username": "alice",
            "display_name": "Alice",
            "email": "alice@example.com",
            "name": "Alice",
            "dob": "1990-01-01",
        }
    ]

    dialog = AccountDialog(atlas)
    assert atlas.last_thread_name == "user-account-list"
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    assert dialog._active_form == "edit"
    assert dialog.edit_username_entry.get_text() == "alice"
    assert dialog.edit_email_entry.get_text() == "alice@example.com"

    dialog.edit_email_entry.set_text("alice.new@example.com")
    dialog.edit_current_password_entry.set_text("Password0")
    dialog.edit_password_entry.set_text("Password1!")
    dialog.edit_confirm_entry.set_text("Password1!")
    dialog.edit_name_entry.set_text("Alice Cooper")
    dialog.edit_dob_entry.set_text("1990-01-02")

    atlas.update_result = {
        "username": "alice",
        "email": "alice.new@example.com",
        "name": "Alice Cooper",
        "dob": "1990-01-02",
    }

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_thread_name == "user-account-update"
    _drain_background(atlas)

    assert atlas.update_called == {
        "username": "alice",
        "password": "Password1!",
        "current_password": "Password0",
        "email": "alice.new@example.com",
        "name": "Alice Cooper",
        "dob": "1990-01-02",
    }

    assert dialog.edit_password_entry.get_text() == ""
    assert dialog.edit_confirm_entry.get_text() == ""
    assert dialog.edit_current_password_entry.get_text() == ""
    assert dialog.edit_feedback_label.get_text().startswith("Account updated")
    assert dialog.edit_title_label.get_text() == "Editing Alice Cooper"

    atlas.list_accounts_result = [
        {
            "username": "alice",
            "display_name": "Alice Cooper",
            "email": "alice.new@example.com",
            "name": "Alice Cooper",
            "dob": "1990-01-02",
        }
    ]

    assert atlas.last_thread_name == "user-account-list"
    _drain_background(atlas)

    assert dialog.account_feedback_label.get_text().startswith("Account updated.")
    assert dialog._account_rows["alice"]["metadata"]["email"] == "alice.new@example.com"


def test_edit_account_validation_blocks_submission():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "email": "alice@example.com"}
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    dialog.edit_email_entry.set_text("invalid-email")
    dialog.edit_password_entry.set_text("short")
    dialog.edit_confirm_entry.set_text("different")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_factory is None
    assert dialog.edit_feedback_label.get_text() == "Enter a valid email address."
    assert getattr(dialog.edit_email_entry, "_atlas_invalid", False) is True

    dialog.edit_email_entry.set_text("alice@example.com")
    dialog.edit_password_entry.set_text("")
    dialog.edit_confirm_entry.set_text("Password1!")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert dialog.edit_feedback_label.get_text() == "Enter the new password before confirming it."
    assert getattr(dialog.edit_password_entry, "_atlas_invalid", False) is True


def test_edit_account_requires_current_password_to_change_it():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "email": "alice@example.com"}
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    dialog.edit_password_entry.set_text("Password2!")
    dialog.edit_confirm_entry.set_text("Password2!")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_factory is None
    assert dialog.edit_feedback_label.get_text() == "Enter your current password to change it."
    assert getattr(dialog.edit_current_password_entry, "_atlas_invalid", False) is True


def test_edit_account_backend_incorrect_password_marks_field():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "email": "alice@example.com"}
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    dialog.edit_current_password_entry.set_text("WrongPass")
    dialog.edit_password_entry.set_text("Password2!")
    dialog.edit_confirm_entry.set_text("Password2!")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_thread_name == "user-account-update"
    assert atlas.last_error is not None

    atlas.last_error(InvalidCurrentPasswordError("Current password is incorrect."))

    assert dialog.edit_feedback_label.get_text() == "Current password is incorrect."
    assert getattr(dialog.edit_current_password_entry, "_atlas_invalid", False) is True


def test_edit_account_rejects_invalid_profile_fields():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "display_name": "Alice", "email": "alice@example.com"}
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    dialog.edit_name_entry.set_text("X" * 81)
    dialog.edit_dob_entry.set_text("2030-01-01")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_factory is None
    assert dialog.edit_feedback_label.get_text().startswith("Display name must be")
    assert getattr(dialog.edit_name_entry, "_atlas_invalid", False) is True
    assert getattr(dialog.edit_dob_entry, "_atlas_invalid", False) is True


def test_edit_account_duplicate_email_shows_error():
    atlas = _AtlasStub()
    atlas.list_accounts_result = [
        {"username": "alice", "email": "alice@example.com"}
    ]

    dialog = AccountDialog(atlas)
    _drain_background(atlas)

    edit_button = dialog._account_rows["alice"]["edit_button"]
    _click(edit_button)

    dialog.edit_email_entry.set_text("taken@example.com")

    dialog._on_edit_save_clicked(dialog.edit_save_button)

    assert atlas.last_thread_name == "user-account-update"

    atlas.last_error(DuplicateUserError("duplicate"))

    assert dialog.edit_feedback_label.get_text() == "Username or email already exists."
    assert getattr(dialog.edit_email_entry, "_atlas_invalid", False) is True
