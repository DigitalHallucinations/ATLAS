"""Dialog for managing ATLAS user accounts."""

from __future__ import annotations

import logging
import re
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Utils.utils import apply_css, create_box
from modules.user_accounts.user_account_service import DuplicateUserError


class AccountDialog(Gtk.Window):
    """Provide login, registration, and logout flows for ATLAS accounts."""

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def __init__(self, atlas, parent: Optional[Gtk.Window] = None) -> None:
        super().__init__(title="Account Management")
        self.logger = logging.getLogger(__name__)
        self.ATLAS = atlas
        self._active_user_listener = None
        self._active_username: Optional[str] = None
        self._active_display_name: Optional[str] = None
        self._editing_username: Optional[str] = None
        self._editing_metadata: dict[str, object] = {}

        self.set_modal(True)
        if parent is not None:
            try:
                self.set_transient_for(parent)
            except Exception:  # pragma: no cover - defensive for stub environments
                pass

        self.set_default_size(420, 420)

        apply_css()
        context = self.get_style_context()
        context.add_class("chat-page")
        context.add_class("sidebar")

        self._build_ui()
        self._subscribe_to_active_user()
        self._refresh_account_list()
        self.connect("close-request", self._on_close_request)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=18)
        self.set_child(container)

        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0.0)
        container.append(self.status_label)

        self._account_rows: dict[str, dict[str, Gtk.Widget]] = {}
        self._forms_busy = False
        self._account_busy = False
        self._account_forms_locked = False
        self._post_refresh_feedback: Optional[str] = None

        account_section = self._build_account_section()
        container.append(account_section)

        toggle_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        container.append(toggle_box)

        self.login_toggle_button = Gtk.Button(label="Sign in")
        self.login_toggle_button.connect("clicked", lambda *_: self._show_form("login"))
        toggle_box.append(self.login_toggle_button)

        self.register_toggle_button = Gtk.Button(label="Create account")
        self.register_toggle_button.connect("clicked", lambda *_: self._show_form("register"))
        toggle_box.append(self.register_toggle_button)

        self.login_box = self._build_login_form()
        container.append(self.login_box)

        self.register_box = self._build_registration_form()
        self._set_widget_visible(self.register_box, False)
        container.append(self.register_box)

        self.edit_box = self._build_edit_form()
        self._set_widget_visible(self.edit_box, False)
        container.append(self.edit_box)

        self.logout_button = Gtk.Button(label="Log out")
        self.logout_button.connect("clicked", self._on_logout_clicked)
        container.append(self.logout_button)

        self._forms = {
            "login": self.login_box,
            "register": self.register_box,
            "edit": self.edit_box,
        }
        self._active_form = "login"
        self._update_toggle_buttons()

        self._forms_sensitive_widgets = [
            self.login_box,
            self.register_box,
            self.edit_box,
            self.login_toggle_button,
            self.register_toggle_button,
            self.login_button,
            self.register_button,
            self.edit_save_button,
            self.logout_button,
        ]
        self._set_forms_busy(False)

    def _build_account_section(self) -> Gtk.Widget:
        wrapper = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=0)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        wrapper.append(header)

        title = Gtk.Label(label="Saved accounts")
        title.set_xalign(0.0)
        header.append(title)

        spacer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        try:
            spacer.set_hexpand(True)
        except Exception:  # pragma: no cover - stub compatibility
            pass
        header.append(spacer)

        self.account_refresh_button = Gtk.Button(label="Refresh")
        self.account_refresh_button.connect("clicked", self._refresh_account_list)
        header.append(self.account_refresh_button)

        scroller = Gtk.ScrolledWindow()
        try:
            scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        except Exception:  # pragma: no cover - stub environments may not implement policy
            pass

        self.account_list_box = getattr(Gtk, "ListBox", Gtk.Box)()
        scroller.set_child(self.account_list_box)
        wrapper.append(scroller)

        self.account_empty_label = Gtk.Label()
        self.account_empty_label.set_xalign(0.0)
        wrapper.append(self.account_empty_label)

        self.account_feedback_label = Gtk.Label()
        self.account_feedback_label.set_xalign(0.0)
        wrapper.append(self.account_feedback_label)

        return wrapper

    # ------------------------------------------------------------------
    # Account list handling
    # ------------------------------------------------------------------
    def _refresh_account_list(self, *_args) -> None:
        if self._account_busy:
            return

        self._set_account_busy(True, "Loading saved accounts…", disable_forms=False)

        def factory():
            return self.ATLAS.list_user_accounts()

        def on_success(result) -> None:
            GLib.idle_add(self._handle_account_list_result, result)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_list_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-list",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to refresh account list: %s", exc, exc_info=True)
            self._handle_account_list_error(exc)

    def _handle_account_list_result(self, accounts) -> bool:
        self._account_rows.clear()

        removed = False
        remover = getattr(self.account_list_box, "remove_all", None)
        if callable(remover):
            try:
                remover()
                removed = True
            except Exception:  # pragma: no cover - fallback to manual removal
                removed = False

        if not removed:
            get_children = getattr(self.account_list_box, "get_children", None)
            children = []
            if callable(get_children):
                try:
                    children = list(get_children())
                except Exception:  # pragma: no cover - fallback for stubs
                    children = []
            elif hasattr(self.account_list_box, "children"):
                children = list(getattr(self.account_list_box, "children", []))

            for child in children:
                try:
                    self.account_list_box.remove(child)
                except Exception:  # pragma: no cover - stub removal safety
                    pass

            if hasattr(self.account_list_box, "children"):
                self.account_list_box.children = []

        count = 0
        iterable = accounts if isinstance(accounts, (list, tuple)) else []
        for entry in iterable:
            if isinstance(entry, dict):
                source = dict(entry)
            else:
                source = {}
                for attr in ("username", "display_name", "name", "email", "dob"):
                    if hasattr(entry, attr):
                        source[attr] = getattr(entry, attr)

            username = str(source.get("username") or "").strip()
            if not username:
                continue

            display_name = source.get("display_name") or source.get("name")

            def _string_or_empty(value):
                if value is None:
                    return ""
                return str(value)

            account_data = {
                "username": username,
                "display_name": display_name,
                "email": _string_or_empty(source.get("email")),
                "name": _string_or_empty(source.get("name")),
                "dob": _string_or_empty(source.get("dob")),
            }

            row = self._create_account_row(account_data)
            self.account_list_box.append(row["row"])
            self._account_rows[username] = row
            count += 1

        if count:
            self.account_empty_label.set_text("")
            message = f"Loaded {count} account{'s' if count != 1 else ''}."
        else:
            self.account_empty_label.set_text("No saved accounts yet.")
            message = "No saved accounts found."

        if self._post_refresh_feedback:
            message = f"{self._post_refresh_feedback} {message}".strip()
            self._post_refresh_feedback = None

        self._set_account_busy(False, message, disable_forms=False)
        self._highlight_active_account()
        return False

    def _handle_account_list_error(self, exc: Exception) -> bool:
        self.account_empty_label.set_text("Failed to load saved accounts.")
        message = f"Account load failed: {exc}"
        if self._post_refresh_feedback:
            message = f"{self._post_refresh_feedback} {message}".strip()
            self._post_refresh_feedback = None
        self._set_account_busy(False, message, disable_forms=False)
        return False

    def _create_account_row(self, account_data: dict[str, object]) -> dict[str, Gtk.Widget]:
        username = str(account_data.get("username") or "").strip()
        display_name = account_data.get("display_name") or account_data.get("name")

        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row_box.username = username  # type: ignore[attr-defined]

        name_label = Gtk.Label(label=str(display_name or username))
        name_label.set_xalign(0.0)
        row_box.append(name_label)

        badge_label = Gtk.Label()
        badge_label.set_xalign(0.0)
        row_box.append(badge_label)

        spacer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        try:
            spacer.set_hexpand(True)
        except Exception:  # pragma: no cover - stub compatibility
            pass
        row_box.append(spacer)

        use_button = Gtk.Button(label="Use")
        use_button.connect("clicked", lambda _btn, user=username: self._on_use_account_clicked(user))
        row_box.append(use_button)

        edit_button = Gtk.Button(label="Edit")
        edit_button.connect("clicked", lambda _btn, user=username: self._on_edit_account_clicked(user))
        row_box.append(edit_button)

        delete_button = Gtk.Button(label="Delete")
        delete_button.connect("clicked", lambda _btn, user=username: self._on_delete_account_clicked(user))
        row_box.append(delete_button)

        return {
            "row": row_box,
            "name_label": name_label,
            "active_label": badge_label,
            "use_button": use_button,
            "edit_button": edit_button,
            "delete_button": delete_button,
            "metadata": account_data,
        }

    def _on_use_account_clicked(self, username: str) -> None:
        if self._account_busy:
            return

        self._set_account_busy(True, f"Activating {username}…", disable_forms=True)

        def factory():
            return self.ATLAS.activate_user_account(username)

        def on_success(_result) -> None:
            GLib.idle_add(self._handle_activate_success)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_action_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-activate",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start activate task: %s", exc, exc_info=True)
            self._handle_account_action_error(exc)

    def _on_edit_account_clicked(self, username: str) -> None:
        if self._account_busy:
            return

        row = self._account_rows.get(username)
        if not row:
            return

        metadata = row.get("metadata") or {}
        self._load_edit_form(username, metadata)

    def _on_delete_account_clicked(self, username: str) -> None:
        if self._account_busy:
            return

        if not self._confirm_account_delete(username):
            self.account_feedback_label.set_text("Deletion cancelled.")
            return

        self._set_account_busy(True, f"Deleting {username}…", disable_forms=True)

        def factory():
            return self.ATLAS.delete_user_account(username)

        def on_success(_result) -> None:
            GLib.idle_add(self._handle_delete_success, username)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_action_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-delete",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start delete task: %s", exc, exc_info=True)
            self._handle_account_action_error(exc)

    def _confirm_account_delete(self, username: str) -> bool:
        self._last_delete_prompt = username  # type: ignore[attr-defined]

        handler = getattr(self, "_confirm_delete_handler", None)
        if callable(handler):
            try:
                return bool(handler(username))
            except Exception:  # pragma: no cover - defensive confirmation hooks
                return False

        dialog_cls = getattr(Gtk, "AlertDialog", None)
        if dialog_cls is not None:
            try:
                dialog = dialog_cls(title="Delete account?", body=f'Delete saved account "{username}"?')
                setter = getattr(dialog, "set_buttons", None)
                if callable(setter):
                    setter(["Cancel", "Delete"])
                response = dialog.choose(self)
                if isinstance(response, str):
                    return response.lower() in {"delete", "accept"}
                accept_value = getattr(getattr(Gtk, "ResponseType", object), "ACCEPT", None)
                return response == accept_value
            except Exception:  # pragma: no cover - alert dialogs are best-effort
                pass

        return True

    def _handle_activate_success(self) -> bool:
        self._set_account_busy(False, "Account activated.", disable_forms=True)
        self._highlight_active_account()
        return False

    def _handle_delete_success(self, username: str) -> bool:
        self._set_account_busy(False, "Account deleted.", disable_forms=True)
        self._post_refresh_feedback = "Account deleted."
        self._refresh_account_list()
        return False

    def _handle_account_action_error(self, exc: Exception) -> bool:
        self._set_account_busy(False, f"Account action failed: {exc}", disable_forms=True)
        return False

    def _set_forms_busy(self, busy: bool) -> None:
        self._forms_busy = bool(busy)
        for widget in getattr(self, "_forms_sensitive_widgets", []):
            self._set_widget_sensitive(widget, not busy)

    def _set_account_busy(self, busy: bool, message: str | None = None, *, disable_forms: bool = True) -> None:
        self._account_busy = bool(busy)
        if message is not None:
            self.account_feedback_label.set_text(message)

        if busy:
            if disable_forms:
                self._account_forms_locked = True
                self._set_forms_busy(True)
        else:
            if self._account_forms_locked or disable_forms:
                self._account_forms_locked = False
                self._set_forms_busy(False)

        self._set_widget_sensitive(self.account_refresh_button, not busy)
        self._update_account_buttons_sensitive()

    def _update_account_buttons_sensitive(self) -> None:
        active_username = self._active_username or ""
        for username, widgets in self._account_rows.items():
            allow_use = not self._account_busy and username != active_username
            self._set_widget_sensitive(widgets["use_button"], allow_use)
            self._set_widget_sensitive(widgets["edit_button"], not self._account_busy)
            self._set_widget_sensitive(widgets["delete_button"], not self._account_busy)

    def _highlight_active_account(self) -> None:
        active_username = self._active_username or ""
        for username, widgets in self._account_rows.items():
            self._set_row_active(widgets, username == active_username)

    def _set_row_active(self, widgets: dict[str, Gtk.Widget], active: bool) -> None:
        label = widgets["active_label"]
        label.set_text("Active" if active else "")
        row = widgets["row"]
        if active:
            try:
                row.add_css_class("suggested-action")
            except Exception:  # pragma: no cover - stub environments may not implement CSS APIs
                pass
        else:
            remover = getattr(row, "remove_css_class", None)
            if callable(remover):
                try:
                    remover("suggested-action")
                except Exception:  # pragma: no cover - stub safety
                    pass

        row.is_active = active  # type: ignore[attr-defined]
        self._update_account_buttons_sensitive()

    @staticmethod
    def _set_widget_sensitive(widget, sensitive: bool) -> None:
        setter = getattr(widget, "set_sensitive", None)
        if callable(setter):
            try:
                setter(bool(sensitive))
            except Exception:  # pragma: no cover - stub environments may not accept bools
                pass
        setattr(widget, "_sensitive", bool(sensitive))

    def _build_login_form(self) -> Gtk.Widget:
        wrapper = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=0)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        wrapper.append(grid)

        username_label = Gtk.Label(label="Username")
        username_label.set_xalign(0.0)
        grid.attach(username_label, 0, 0, 1, 1)

        self.login_username_entry = Gtk.Entry()
        self.login_username_entry.set_placeholder_text("Your username")
        grid.attach(self.login_username_entry, 1, 0, 1, 1)

        password_label = Gtk.Label(label="Password")
        password_label.set_xalign(0.0)
        grid.attach(password_label, 0, 1, 1, 1)

        self.login_password_entry = Gtk.Entry()
        self.login_password_entry.set_visibility(False)
        self.login_password_entry.set_invisible_char("•")
        self.login_password_entry.set_placeholder_text("Your password")
        grid.attach(self.login_password_entry, 1, 1, 1, 1)

        self.login_feedback_label = Gtk.Label()
        self.login_feedback_label.set_xalign(0.0)
        wrapper.append(self.login_feedback_label)

        self.login_button = Gtk.Button(label="Sign in")
        self.login_button.connect("clicked", self._on_login_clicked)
        wrapper.append(self.login_button)

        return wrapper

    def _build_registration_form(self) -> Gtk.Widget:
        wrapper = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=0)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        wrapper.append(grid)

        username_label = Gtk.Label(label="Username")
        username_label.set_xalign(0.0)
        grid.attach(username_label, 0, 0, 1, 1)

        self.register_username_entry = Gtk.Entry()
        self.register_username_entry.set_placeholder_text("Choose a username")
        grid.attach(self.register_username_entry, 1, 0, 1, 1)

        email_label = Gtk.Label(label="Email")
        email_label.set_xalign(0.0)
        grid.attach(email_label, 0, 1, 1, 1)

        self.register_email_entry = Gtk.Entry()
        self.register_email_entry.set_placeholder_text("name@example.com")
        grid.attach(self.register_email_entry, 1, 1, 1, 1)

        password_label = Gtk.Label(label="Password")
        password_label.set_xalign(0.0)
        grid.attach(password_label, 0, 2, 1, 1)

        self.register_password_entry = Gtk.Entry()
        self.register_password_entry.set_visibility(False)
        self.register_password_entry.set_invisible_char("•")
        self.register_password_entry.set_placeholder_text("Create a password")
        grid.attach(self.register_password_entry, 1, 2, 1, 1)

        confirm_label = Gtk.Label(label="Confirm password")
        confirm_label.set_xalign(0.0)
        grid.attach(confirm_label, 0, 3, 1, 1)

        self.register_confirm_entry = Gtk.Entry()
        self.register_confirm_entry.set_visibility(False)
        self.register_confirm_entry.set_invisible_char("•")
        self.register_confirm_entry.set_placeholder_text("Repeat password")
        grid.attach(self.register_confirm_entry, 1, 3, 1, 1)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 4, 1, 1)

        self.register_name_entry = Gtk.Entry()
        self.register_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.register_name_entry, 1, 4, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 5, 1, 1)

        self.register_dob_entry = Gtk.Entry()
        self.register_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.register_dob_entry, 1, 5, 1, 1)

        self.register_feedback_label = Gtk.Label()
        self.register_feedback_label.set_xalign(0.0)
        wrapper.append(self.register_feedback_label)

        self.register_button = Gtk.Button(label="Create account")
        self.register_button.connect("clicked", self._on_register_clicked)
        wrapper.append(self.register_button)

        return wrapper

    def _build_edit_form(self) -> Gtk.Widget:
        wrapper = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=0)

        self.edit_title_label = Gtk.Label()
        self.edit_title_label.set_xalign(0.0)
        wrapper.append(self.edit_title_label)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        wrapper.append(grid)

        username_label = Gtk.Label(label="Username")
        username_label.set_xalign(0.0)
        grid.attach(username_label, 0, 0, 1, 1)

        self.edit_username_entry = Gtk.Entry()
        try:
            self.edit_username_entry.set_editable(False)
            self.edit_username_entry.set_can_focus(False)
        except Exception:  # pragma: no cover - GTK stub compatibility
            pass
        grid.attach(self.edit_username_entry, 1, 0, 1, 1)

        email_label = Gtk.Label(label="Email")
        email_label.set_xalign(0.0)
        grid.attach(email_label, 0, 1, 1, 1)

        self.edit_email_entry = Gtk.Entry()
        self.edit_email_entry.set_placeholder_text("name@example.com")
        grid.attach(self.edit_email_entry, 1, 1, 1, 1)

        password_label = Gtk.Label(label="New password")
        password_label.set_xalign(0.0)
        grid.attach(password_label, 0, 2, 1, 1)

        self.edit_password_entry = Gtk.Entry()
        self.edit_password_entry.set_visibility(False)
        self.edit_password_entry.set_invisible_char("•")
        self.edit_password_entry.set_placeholder_text("Leave blank to keep current password")
        grid.attach(self.edit_password_entry, 1, 2, 1, 1)

        confirm_label = Gtk.Label(label="Confirm password")
        confirm_label.set_xalign(0.0)
        grid.attach(confirm_label, 0, 3, 1, 1)

        self.edit_confirm_entry = Gtk.Entry()
        self.edit_confirm_entry.set_visibility(False)
        self.edit_confirm_entry.set_invisible_char("•")
        self.edit_confirm_entry.set_placeholder_text("Repeat new password")
        grid.attach(self.edit_confirm_entry, 1, 3, 1, 1)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 4, 1, 1)

        self.edit_name_entry = Gtk.Entry()
        self.edit_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.edit_name_entry, 1, 4, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 5, 1, 1)

        self.edit_dob_entry = Gtk.Entry()
        self.edit_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.edit_dob_entry, 1, 5, 1, 1)

        self.edit_feedback_label = Gtk.Label()
        self.edit_feedback_label.set_xalign(0.0)
        wrapper.append(self.edit_feedback_label)

        self.edit_save_button = Gtk.Button(label="Save changes")
        self.edit_save_button.connect("clicked", self._on_edit_save_clicked)
        wrapper.append(self.edit_save_button)

        return wrapper

    # ------------------------------------------------------------------
    # Active user synchronisation
    # ------------------------------------------------------------------
    def _subscribe_to_active_user(self) -> None:
        def _listener(username: str, display_name: str) -> None:
            GLib.idle_add(self._apply_active_user_state, username, display_name)

        try:
            self._active_user_listener = _listener
            self.ATLAS.add_active_user_change_listener(_listener)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Unable to subscribe to active user changes: %s", exc, exc_info=True)
            self._active_user_listener = None

    def _apply_active_user_state(self, username: str, display_name: str) -> bool:
        self._active_username = username
        self._active_display_name = display_name

        persisted = None
        try:
            persisted = self.ATLAS.config_manager.get_active_user()
        except Exception:  # pragma: no cover - config lookups are best-effort
            persisted = None

        if persisted:
            message = f"Signed in as {display_name}"
        else:
            message = "No active account"

        self.status_label.set_text(message)
        try:
            self.logout_button.set_sensitive(bool(persisted))
        except Exception:  # pragma: no cover - stub safety
            pass

        self._highlight_active_account()

        return False

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    def _show_form(self, name: str) -> None:
        if name == self._active_form:
            return

        for form_name, widget in self._forms.items():
            should_show = form_name == name
            if not self._set_widget_visible(widget, should_show):
                if should_show:
                    getattr(widget, "show", lambda: None)()
                else:
                    getattr(widget, "hide", lambda: None)()

        self._active_form = name
        self._update_toggle_buttons()

    def _update_toggle_buttons(self) -> None:
        if self._active_form == "login":
            self.login_toggle_button.add_css_class("suggested-action")
            self.register_toggle_button.remove_css_class("suggested-action")
        else:
            self.register_toggle_button.add_css_class("suggested-action")
            self.login_toggle_button.remove_css_class("suggested-action")

    # ------------------------------------------------------------------
    # Login flow
    # ------------------------------------------------------------------
    def _set_login_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self.login_button.set_sensitive(not busy)
        if message is not None:
            self.login_feedback_label.set_text(message)

    def _on_login_clicked(self, _button) -> None:
        username = (self.login_username_entry.get_text() or "").strip()
        password = self.login_password_entry.get_text() or ""

        if not username or not password:
            self.login_feedback_label.set_text("Username and password are required.")
            return

        self._set_login_busy(True, "Signing in…")

        def factory():
            return self.ATLAS.login_user_account(username, password)

        def on_success(result: bool) -> None:
            GLib.idle_add(self._handle_login_result, bool(result))

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_login_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-login",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start login task: %s", exc, exc_info=True)
            self._handle_login_error(exc)

    def _handle_login_result(self, success: bool) -> bool:
        self._set_login_busy(False)
        if success:
            self.login_feedback_label.set_text("Signed in successfully.")
            self.close()
        else:
            self.login_feedback_label.set_text("Invalid username or password.")
        return False

    def _handle_login_error(self, exc: Exception) -> bool:
        self._set_login_busy(False)
        self.login_feedback_label.set_text(f"Login failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Registration flow
    # ------------------------------------------------------------------
    def _set_register_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self.register_button.set_sensitive(not busy)
        if message is not None:
            self.register_feedback_label.set_text(message)

    def _mark_field_valid(self, widget) -> None:
        remover = getattr(widget, "remove_css_class", None)
        if callable(remover):
            try:
                remover("error")
            except Exception:  # pragma: no cover - GTK fallback
                pass
        setattr(widget, "_atlas_invalid", False)

    def _mark_field_invalid(self, widget) -> None:
        adder = getattr(widget, "add_css_class", None)
        if callable(adder):
            try:
                adder("error")
            except Exception:  # pragma: no cover - GTK fallback
                pass
        setattr(widget, "_atlas_invalid", True)

    def _password_validation_error(self, password: str) -> Optional[str]:
        if len(password) < 8:
            return "Password must be at least 8 characters and include letters and numbers."

        has_letter = any(char.isalpha() for char in password)
        has_digit = any(char.isdigit() for char in password)

        if not (has_letter and has_digit):
            return "Password must be at least 8 characters and include letters and numbers."

        return None

    def _clear_registration_validation(self) -> None:
        for widget in (
            self.register_username_entry,
            self.register_email_entry,
            self.register_password_entry,
            self.register_confirm_entry,
        ):
            self._mark_field_valid(widget)

    def _on_register_clicked(self, _button) -> None:
        username = (self.register_username_entry.get_text() or "").strip()
        email = (self.register_email_entry.get_text() or "").strip()
        password = self.register_password_entry.get_text() or ""
        confirm = self.register_confirm_entry.get_text() or ""
        name = (self.register_name_entry.get_text() or "").strip() or None
        dob = (self.register_dob_entry.get_text() or "").strip() or None

        self._clear_registration_validation()

        errors: list[tuple[str, object]] = []

        if not username:
            errors.append(("Username is required.", self.register_username_entry))

        if not email:
            errors.append(("Email is required.", self.register_email_entry))
        elif not self._EMAIL_PATTERN.fullmatch(email):
            errors.append(("Enter a valid email address.", self.register_email_entry))

        if not password:
            errors.append(("Password is required.", self.register_password_entry))
        else:
            password_error = self._password_validation_error(password)
            if password_error:
                errors.append((password_error, self.register_password_entry))

        if password != confirm:
            errors.append(("Passwords do not match.", self.register_confirm_entry))

        if errors:
            for _message, widget in errors:
                self._mark_field_invalid(widget)

            self.register_feedback_label.set_text(errors[0][0])
            first_widget = errors[0][1]
            grabber = getattr(first_widget, "grab_focus", None)
            if callable(grabber):
                grabber()
            return

        self._set_register_busy(True, "Creating account…")

        def factory():
            return self.ATLAS.register_user_account(username, password, email, name, dob)

        def on_success(result: dict) -> None:
            GLib.idle_add(self._handle_register_result, result)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_register_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-register",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start registration task: %s", exc, exc_info=True)
            self._handle_register_error(exc)

    def _handle_register_result(self, result: dict) -> bool:
        self._set_register_busy(False)
        username = result.get("username") if isinstance(result, dict) else None
        self.register_feedback_label.set_text(
            f"Account created for {username or 'new user'}." if username else "Account created successfully."
        )
        self.close()
        return False

    def _handle_register_error(self, exc: Exception) -> bool:
        self._set_register_busy(False)
        if isinstance(exc, DuplicateUserError):
            message = "Username or email already exists."
            self.register_feedback_label.set_text(message)
            self._mark_field_invalid(self.register_username_entry)
            self._mark_field_invalid(self.register_email_entry)
            focus_widget = self.register_username_entry
            grabber = getattr(focus_widget, "grab_focus", None)
            if callable(grabber):
                grabber()
        elif isinstance(exc, ValueError):
            self.register_feedback_label.set_text(str(exc))
        else:
            self.register_feedback_label.set_text(f"Registration failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Account editing flow
    # ------------------------------------------------------------------
    def _clear_edit_validation(self) -> None:
        for widget in (
            self.edit_email_entry,
            self.edit_password_entry,
            self.edit_confirm_entry,
        ):
            self._mark_field_valid(widget)

    def _load_edit_form(self, username: str, metadata: dict[str, object]) -> None:
        self._editing_username = username
        self._editing_metadata = dict(metadata)

        display_name = metadata.get("display_name") or metadata.get("name")
        heading = display_name or username
        self.edit_title_label.set_text(f"Editing {heading}")

        self.edit_username_entry.set_text(username)
        self.edit_email_entry.set_text(str(metadata.get("email") or ""))
        self.edit_name_entry.set_text(str(metadata.get("name") or ""))
        self.edit_dob_entry.set_text(str(metadata.get("dob") or ""))
        self.edit_password_entry.set_text("")
        self.edit_confirm_entry.set_text("")
        self.edit_feedback_label.set_text("")
        self._clear_edit_validation()

        self._show_form("edit")

        grabber = getattr(self.edit_email_entry, "grab_focus", None)
        if callable(grabber):
            grabber()

    def _set_edit_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self.edit_save_button.set_sensitive(not busy)
        if message is not None:
            self.edit_feedback_label.set_text(message)

    def _on_edit_save_clicked(self, _button) -> None:
        username = self._editing_username
        if not username:
            self.edit_feedback_label.set_text("Select an account to edit.")
            return

        row = self._account_rows.get(username)
        metadata = dict(self._editing_metadata)
        if row and isinstance(row.get("metadata"), dict):
            metadata = dict(row["metadata"])

        email = (self.edit_email_entry.get_text() or "").strip()
        password = self.edit_password_entry.get_text() or ""
        confirm = self.edit_confirm_entry.get_text() or ""
        name = (self.edit_name_entry.get_text() or "").strip()
        dob = (self.edit_dob_entry.get_text() or "").strip()

        original_email = str(metadata.get("email") or "").strip()
        original_name = str(metadata.get("name") or "").strip()
        original_dob = str(metadata.get("dob") or "").strip()

        self._clear_edit_validation()

        errors: list[tuple[str, object]] = []

        if not email:
            errors.append(("Email is required.", self.edit_email_entry))
        elif not self._EMAIL_PATTERN.fullmatch(email):
            errors.append(("Enter a valid email address.", self.edit_email_entry))

        if confirm and not password:
            errors.append(("Enter the new password before confirming it.", self.edit_password_entry))

        if password:
            password_error = self._password_validation_error(password)
            if password_error:
                errors.append((password_error, self.edit_password_entry))

        if password != confirm:
            if password or confirm:
                errors.append(("Passwords do not match.", self.edit_confirm_entry))

        if errors:
            for _message, widget in errors:
                self._mark_field_invalid(widget)

            self.edit_feedback_label.set_text(errors[0][0])
            first_widget = errors[0][1]
            grabber = getattr(first_widget, "grab_focus", None)
            if callable(grabber):
                grabber()
            return

        self._set_edit_busy(True, "Saving changes…")

        password_arg = password or None

        def _derive_update(new_value: str, original_value: str) -> Optional[str]:
            if new_value == original_value:
                return None
            if not new_value:
                return "" if original_value else None
            return new_value

        name_arg = _derive_update(name, original_name)
        dob_arg = _derive_update(dob, original_dob)

        def factory():
            return self.ATLAS.update_user_account(
                username,
                password=password_arg,
                email=email,
                name=name_arg,
                dob=dob_arg,
            )

        def on_success(result) -> None:
            GLib.idle_add(self._handle_edit_success, result)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_edit_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-update",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start update task: %s", exc, exc_info=True)
            self._handle_edit_error(exc)

    def _handle_edit_success(self, result: dict | object) -> bool:
        self._set_edit_busy(False)
        username = None
        if isinstance(result, dict):
            username = result.get("username")
            self.edit_email_entry.set_text(str(result.get("email") or ""))
            self.edit_name_entry.set_text(str(result.get("name") or ""))
            self.edit_dob_entry.set_text(str(result.get("dob") or ""))
            self._editing_metadata.update(
                {
                    "username": result.get("username") or self._editing_metadata.get("username"),
                    "email": str(result.get("email") or ""),
                    "name": str(result.get("name") or ""),
                    "dob": str(result.get("dob") or ""),
                    "display_name": result.get("name") or self._editing_metadata.get("display_name"),
                }
            )
            heading = self._editing_metadata.get("display_name") or self._editing_metadata.get("username")
            if heading:
                self.edit_title_label.set_text(f"Editing {heading}")

        self.edit_password_entry.set_text("")
        self.edit_confirm_entry.set_text("")

        message = "Account updated."
        if username:
            message = f"Account updated for {username}."
        self.edit_feedback_label.set_text(message)

        self._post_refresh_feedback = "Account updated."
        self._refresh_account_list()
        return False

    def _handle_edit_error(self, exc: Exception) -> bool:
        self._set_edit_busy(False)
        if isinstance(exc, DuplicateUserError):
            self.edit_feedback_label.set_text("Username or email already exists.")
            self._mark_field_invalid(self.edit_email_entry)
            grabber = getattr(self.edit_email_entry, "grab_focus", None)
            if callable(grabber):
                grabber()
        elif isinstance(exc, ValueError):
            self.edit_feedback_label.set_text(str(exc))
        else:
            self.edit_feedback_label.set_text(f"Update failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Logout flow
    # ------------------------------------------------------------------
    def _on_logout_clicked(self, _button) -> None:
        self.logout_button.set_sensitive(False)
        self.status_label.set_text("Signing out…")

        def factory():
            return self.ATLAS.logout_active_user()

        def on_success(_result) -> None:
            GLib.idle_add(self._handle_logout_complete)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_logout_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-logout",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start logout task: %s", exc, exc_info=True)
            self._handle_logout_error(exc)

    def _handle_logout_complete(self) -> bool:
        self.status_label.set_text("Signed out.")
        return False

    def _handle_logout_error(self, exc: Exception) -> bool:
        self.status_label.set_text(f"Logout failed: {exc}")
        self.logout_button.set_sensitive(True)
        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _on_close_request(self, *_args) -> bool:
        if self._active_user_listener is not None:
            try:
                self.ATLAS.remove_active_user_change_listener(self._active_user_listener)
            except Exception:  # pragma: no cover - defensive logging
                self.logger.debug("Failed to remove active user listener", exc_info=True)
            self._active_user_listener = None
        return False

    @staticmethod
    def _set_widget_visible(widget, visible: bool) -> bool:
        setter = getattr(widget, "set_visible", None)
        if callable(setter):
            setter(bool(visible))
            return True
        return False
