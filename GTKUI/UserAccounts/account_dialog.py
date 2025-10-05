"""Dialog for managing ATLAS user accounts."""

from __future__ import annotations

import logging
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Utils.utils import apply_css, create_box


class AccountDialog(Gtk.Window):
    """Provide login, registration, and logout flows for ATLAS accounts."""

    def __init__(self, atlas, parent: Optional[Gtk.Window] = None) -> None:
        super().__init__(title="Account Management")
        self.logger = logging.getLogger(__name__)
        self.ATLAS = atlas
        self._active_user_listener = None
        self._active_username: Optional[str] = None
        self._active_display_name: Optional[str] = None

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

        self.logout_button = Gtk.Button(label="Log out")
        self.logout_button.connect("clicked", self._on_logout_clicked)
        container.append(self.logout_button)

        self._forms = {
            "login": self.login_box,
            "register": self.register_box,
        }
        self._active_form = "login"
        self._update_toggle_buttons()

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

    def _on_register_clicked(self, _button) -> None:
        username = (self.register_username_entry.get_text() or "").strip()
        email = (self.register_email_entry.get_text() or "").strip()
        password = self.register_password_entry.get_text() or ""
        confirm = self.register_confirm_entry.get_text() or ""
        name = (self.register_name_entry.get_text() or "").strip() or None
        dob = (self.register_dob_entry.get_text() or "").strip() or None

        if not username or not email or not password:
            self.register_feedback_label.set_text("Username, email, and password are required.")
            return

        if password != confirm:
            self.register_feedback_label.set_text("Passwords do not match.")
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
        self.register_feedback_label.set_text(f"Registration failed: {exc}")
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
