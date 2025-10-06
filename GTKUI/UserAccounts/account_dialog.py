"""Dialog for managing ATLAS user accounts."""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

try:  # pragma: no cover - Gio may be unavailable in some environments
    from gi.repository import Gio
except Exception:  # pragma: no cover - fall back gracefully when Gio is missing
    Gio = None  # type: ignore[assignment]

from GTKUI.Utils.utils import apply_css, create_box
from modules.user_accounts.user_account_service import DuplicateUserError


class AccountDialog(Gtk.Window):
    """Provide login, registration, and logout flows for ATLAS accounts."""

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
    _DOB_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _MAX_DISPLAY_NAME_LENGTH = 80

    def __init__(self, atlas, parent: Optional[Gtk.Window] = None) -> None:
        super().__init__(title="Account Management")
        self.logger = logging.getLogger(__name__)
        self.ATLAS = atlas
        self._active_user_listener = None
        self._active_username: Optional[str] = None
        self._active_display_name: Optional[str] = None
        self._editing_username: Optional[str] = None
        self._editing_metadata: dict[str, object] = {}
        self._account_records: list[dict[str, object]] = []
        self._account_details_cache: dict[str, dict[str, object]] = {}
        self._account_filter_text: str = ""
        self._selected_username: Optional[str] = None
        self._details_busy = False
        self._visible_usernames: list[str] = []
        self._password_toggle_buttons: list[Gtk.Widget] = []

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
            self.account_search_entry,
        ]
        self._forms_sensitive_widgets.extend(self._password_toggle_buttons)
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

        search_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        wrapper.append(search_row)

        search_entry_cls = getattr(Gtk, "SearchEntry", Gtk.Entry)
        self.account_search_entry = search_entry_cls()
        placeholder_text = "Search by username, email or name"
        set_placeholder = getattr(self.account_search_entry, "set_placeholder_text", None)
        if callable(set_placeholder):
            set_placeholder(placeholder_text)
        else:  # pragma: no cover - compatibility fallback
            try:
                self.account_search_entry.set_property("placeholder-text", placeholder_text)
            except Exception:
                pass
        connectable_signals = ["changed", "search-changed"]
        for signal in connectable_signals:
            connect = getattr(self.account_search_entry, "connect", None)
            if callable(connect):
                try:
                    connect(signal, self._on_account_filter_changed)
                except TypeError:
                    continue
        search_row.append(self.account_search_entry)

        clear_button = Gtk.Button(label="Clear")
        clear_button.connect("clicked", self._on_account_search_clear)
        search_row.append(clear_button)

        scroller = Gtk.ScrolledWindow()
        try:
            scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        except Exception:  # pragma: no cover - stub environments may not implement policy
            pass

        self.account_list_box = getattr(Gtk, "ListBox", Gtk.Box)()
        try:
            self.account_list_box.connect("row-selected", self._on_account_row_selected)
        except Exception:  # pragma: no cover - signal not available on fallback widgets
            pass
        scroller.set_child(self.account_list_box)
        wrapper.append(scroller)

        self.account_empty_label = Gtk.Label()
        self.account_empty_label.set_xalign(0.0)
        wrapper.append(self.account_empty_label)

        self.account_feedback_label = Gtk.Label()
        self.account_feedback_label.set_xalign(0.0)
        wrapper.append(self.account_feedback_label)

        details_panel = self._build_account_details_panel()
        wrapper.append(details_panel)

        return wrapper

    def _build_account_details_panel(self) -> Gtk.Widget:
        frame = Gtk.Frame(label="Account details")
        content = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=12)
        frame.set_child(content)

        self.account_details_status_label = Gtk.Label(label="Select an account to view details.")
        self.account_details_status_label.set_xalign(0.0)
        content.append(self.account_details_status_label)

        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        content.append(grid)

        def _add_row(row_index: int, title: str) -> Gtk.Label:
            label = Gtk.Label(label=title)
            label.set_xalign(0.0)
            grid.attach(label, 0, row_index, 1, 1)

            value = Gtk.Label()
            value.set_xalign(0.0)
            grid.attach(value, 1, row_index, 1, 1)
            return value

        self.account_details_username_value = _add_row(0, "Username")
        self.account_details_name_value = _add_row(1, "Display name")
        self.account_details_email_value = _add_row(2, "Email")
        self.account_details_dob_value = _add_row(3, "Date of birth")

        return frame

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
        normalised = self._normalise_account_payload(accounts)
        self._account_records = normalised

        known_usernames = {entry["username"] for entry in normalised}
        self._account_details_cache = {
            username: details
            for username, details in self._account_details_cache.items()
            if username in known_usernames
        }
        if self._selected_username not in known_usernames:
            self._selected_username = None

        visible_accounts = self._apply_account_filter(normalised)
        message = self._render_account_rows(visible_accounts, context="load")

        if self._post_refresh_feedback:
            message = f"{self._post_refresh_feedback} {message}".strip()
            self._post_refresh_feedback = None

        self._set_account_busy(False, message, disable_forms=False)
        return False

    def _handle_account_list_error(self, exc: Exception) -> bool:
        self.account_empty_label.set_text("Failed to load saved accounts.")
        message = f"Account load failed: {exc}"
        if self._post_refresh_feedback:
            message = f"{self._post_refresh_feedback} {message}".strip()
            self._post_refresh_feedback = None
        self._set_account_busy(False, message, disable_forms=False)
        return False

    def _normalise_account_payload(self, accounts) -> list[dict[str, object]]:
        iterable = list(accounts) if isinstance(accounts, (list, tuple)) else []
        normalised: list[dict[str, object]] = []

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
                "display_name": _string_or_empty(display_name) or "",
                "email": _string_or_empty(source.get("email")),
                "name": _string_or_empty(source.get("name")),
                "dob": _string_or_empty(source.get("dob")),
            }
            normalised.append(account_data)

        normalised.sort(key=lambda item: item["username"].lower())
        return normalised

    def _apply_account_filter(self, accounts: list[dict[str, object]]) -> list[dict[str, object]]:
        if not self._account_filter_text:
            return list(accounts)

        needle = self._account_filter_text.lower()
        filtered: list[dict[str, object]] = []

        for account in accounts:
            haystack_parts = [
                account.get("username", ""),
                account.get("email", ""),
                account.get("name", ""),
                account.get("display_name", ""),
            ]
            haystack = " \n".join(str(part).lower() for part in haystack_parts if part)
            if needle in haystack:
                filtered.append(account)

        return filtered

    def _render_account_rows(self, accounts: list[dict[str, object]], *, context: str) -> str:
        self._account_rows.clear()
        self._visible_usernames = []

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
        for account_data in accounts:
            row = self._create_account_row(account_data)
            self.account_list_box.append(row["row"])
            username = account_data["username"]
            self._account_rows[username] = row
            self._visible_usernames.append(username)
            count += 1

        if count:
            self.account_empty_label.set_text("")
            if context == "search":
                message = f"Found {count} account{'s' if count != 1 else ''} matching your search."
            else:
                message = f"Loaded {count} account{'s' if count != 1 else ''}."
        else:
            if context == "search":
                empty_message = "No accounts match your search."
                message = empty_message
            else:
                empty_message = "No saved accounts yet."
                message = "No saved accounts found."
            self.account_empty_label.set_text(empty_message)

        self._highlight_active_account()
        self._restore_account_details()
        self._update_account_buttons_sensitive()
        return message

    def _restore_account_details(self) -> None:
        if self._selected_username and self._selected_username in self._visible_usernames:
            cached = self._account_details_cache.get(self._selected_username)
            if cached:
                self._apply_account_detail_mapping(cached)
            else:
                self._set_account_detail_message("Loading account details…")
                self._on_account_details_clicked(self._selected_username)
        else:
            self._selected_username = None
            self._set_account_detail_message("Select an account to view details.")

    def _set_account_detail_message(self, message: str) -> None:
        self.account_details_status_label.set_text(message)
        if message:
            placeholder = "—"
            self.account_details_username_value.set_text(placeholder)
            self.account_details_name_value.set_text(placeholder)
            self.account_details_email_value.set_text(placeholder)
            self.account_details_dob_value.set_text(placeholder)

    def _apply_account_detail_mapping(self, details: dict[str, object]) -> None:
        username = self._format_detail_value(details.get("username"))
        display_name = details.get("display_name") or details.get("name")
        email = self._format_detail_value(details.get("email"))
        dob = self._format_detail_value(details.get("dob"))

        self.account_details_status_label.set_text("")
        self.account_details_username_value.set_text(username)
        self.account_details_name_value.set_text(self._format_detail_value(display_name))
        self.account_details_email_value.set_text(email)
        self.account_details_dob_value.set_text(dob)

    @staticmethod
    def _format_detail_value(value: object, fallback: str = "—") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text or fallback

    def _on_account_filter_changed(self, entry, *_args) -> None:
        text = getattr(entry, "get_text", lambda: "")()
        self._account_filter_text = (text or "").strip()
        accounts = self._apply_account_filter(self._account_records)
        context = "search" if self._account_filter_text else "load"
        message = self._render_account_rows(accounts, context=context)
        self.account_feedback_label.set_text(message)

    def _on_account_search_clear(self, _button) -> None:
        setter = getattr(self.account_search_entry, "set_text", None)
        if callable(setter):
            setter("")
        self._account_filter_text = ""
        accounts = self._apply_account_filter(self._account_records)
        message = self._render_account_rows(accounts, context="load")
        self.account_feedback_label.set_text(message)

    def _on_account_row_selected(self, _listbox, row) -> None:
        if row is None:
            return

        username = getattr(row, "username", None)
        if username is None:
            child_getter = getattr(row, "get_child", None)
            if callable(child_getter):
                child = child_getter()
                username = getattr(child, "username", None)

        if not username:
            return

        self._on_account_details_clicked(username)

    def _on_account_details_clicked(self, username: str) -> None:
        username = str(username or "").strip()
        if not username:
            return

        if self._account_busy:
            return

        self._selected_username = username

        cached = self._account_details_cache.get(username)
        if cached:
            self._apply_account_detail_mapping(cached)
            return

        self._set_account_details_busy(True, f"Loading {username}…")

        def factory():
            return self.ATLAS.get_user_account_details(username)

        def on_success(result) -> None:
            GLib.idle_add(self._handle_account_details_result, username, result)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_details_error, username, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-details",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start details task: %s", exc, exc_info=True)
            self._handle_account_details_error(username, exc)

    def _set_account_details_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self._details_busy = bool(busy)
        if message is not None:
            self._set_account_detail_message(message)
        self._update_account_buttons_sensitive()

    def _handle_account_details_result(self, username: str, result) -> bool:
        if self._selected_username != username:
            return False

        self._set_account_details_busy(False)

        if isinstance(result, dict) and result.get("username"):
            mapping = dict(result)
            mapping.setdefault("display_name", mapping.get("name"))
            self._account_details_cache[username] = mapping
            self._apply_account_detail_mapping(mapping)
        else:
            self._set_account_detail_message("No additional details available.")

        return False

    def _handle_account_details_error(self, username: str, exc: Exception) -> bool:
        if self._selected_username != username:
            return False

        self._set_account_details_busy(False)
        self._set_account_detail_message(f"Failed to load details: {exc}")
        return False

    def _create_account_row(self, account_data: dict[str, object]) -> dict[str, Gtk.Widget]:
        username = str(account_data.get("username") or "").strip()
        display_name = account_data.get("display_name") or account_data.get("name")

        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row_box.username = username  # type: ignore[attr-defined]

        container = row_box
        list_box_cls = getattr(Gtk, "ListBox", None)
        list_box_row_cls = getattr(Gtk, "ListBoxRow", None)
        try:
            if (
                list_box_cls is not None
                and list_box_row_cls is not None
                and isinstance(self.account_list_box, list_box_cls)
            ):
                row_container = list_box_row_cls()
                set_child = getattr(row_container, "set_child", None)
                if callable(set_child):
                    set_child(row_box)
                    row_container.username = username  # type: ignore[attr-defined]
                    container = row_container
        except Exception:  # pragma: no cover - fallback when rows unsupported
            container = row_box

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

        details_button = Gtk.Button(label="Details")
        details_button.connect(
            "clicked", lambda _btn, user=username: self._on_account_details_clicked(user)
        )
        row_box.append(details_button)

        edit_button = Gtk.Button(label="Edit")
        edit_button.connect("clicked", lambda _btn, user=username: self._on_edit_account_clicked(user))
        row_box.append(edit_button)

        delete_button = Gtk.Button(label="Delete")
        delete_button.connect("clicked", lambda _btn, user=username: self._on_delete_account_clicked(user))
        row_box.append(delete_button)

        return {
            "row": container,
            "row_box": row_box,
            "name_label": name_label,
            "active_label": badge_label,
            "use_button": use_button,
            "details_button": details_button,
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
                response = self._resolve_alert_dialog_response(dialog.choose(self))
                if isinstance(response, str):
                    return response.lower() in {"delete", "accept"}
                accept_value = getattr(getattr(Gtk, "ResponseType", object), "ACCEPT", None)
                return response == accept_value
            except Exception:  # pragma: no cover - alert dialogs are best-effort
                pass

        return True

    def _resolve_alert_dialog_response(self, response):
        wait_methods = ("wait_result", "wait")

        future_type = getattr(Gio, "Future", None) if Gio is not None else None

        for name in wait_methods:
            wait = getattr(response, name, None)
            if callable(wait):
                try:
                    result = wait()
                except Exception:  # pragma: no cover - fall back to raw response on failure
                    continue
                if isinstance(result, tuple) and len(result) == 1:
                    return result[0]
                return result

        if future_type is not None:
            try:
                if isinstance(response, future_type):
                    try:
                        result = response.get()
                    except Exception:  # pragma: no cover - not all futures expose get
                        return None
                    if isinstance(result, tuple) and len(result) == 1:
                        return result[0]
                    return result
            except TypeError:  # pragma: no cover - defensive for unusual future types
                pass

        return response

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
            details_button = widgets.get("details_button")
            if details_button is not None:
                allow_details = not self._account_busy and not self._details_busy
                self._set_widget_sensitive(details_button, allow_details)

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

    def _configure_password_entry(
        self,
        entry: Gtk.Entry,
        *,
        strength_label: Optional[Gtk.Label] = None,
    ) -> None:
        entry.set_visibility(False)
        entry.set_invisible_char("•")
        setattr(entry, "_atlas_password_visible", False)

        if strength_label is not None:
            entry.connect(
                "changed",
                lambda widget: self._update_password_strength_label(widget, strength_label),
            )

        entry.connect("icon-press", self._on_password_icon_pressed)
        self._set_password_entry_icon(entry)

    def _create_password_toggle(self, entry: Gtk.Entry) -> Gtk.Widget:
        toggle_cls = getattr(Gtk, "ToggleButton", None)

        if toggle_cls is not None:
            toggle = toggle_cls(label="Show")

            def _on_toggled(button):
                active = getattr(button, "get_active", lambda: False)()
                self._set_password_entry_visibility(entry, active)
                setter = getattr(button, "set_label", None)
                if callable(setter):
                    setter("Hide" if active else "Show")
                self._set_password_entry_icon(entry)

            toggle.connect("toggled", _on_toggled)
        else:
            toggle = Gtk.Button(label="Show")
            setattr(toggle, "_atlas_toggle_active", False)

            def _on_clicked(button):
                current = getattr(button, "_atlas_toggle_active", False)
                new_state = not current
                setattr(button, "_atlas_toggle_active", new_state)
                self._set_password_entry_visibility(entry, new_state)
                setter = getattr(button, "set_label", None)
                if callable(setter):
                    setter("Hide" if new_state else "Show")
                self._set_password_entry_icon(entry)

            toggle.connect("clicked", _on_clicked)

        self._password_toggle_buttons.append(toggle)
        return toggle

    def _set_password_entry_icon(self, entry: Gtk.Entry) -> None:
        icon_setter = getattr(entry, "set_icon_from_icon_name", None)
        icon_position = getattr(Gtk, "EntryIconPosition", None)
        if not callable(icon_setter) or icon_position is None:
            return

        visible = getattr(entry, "_atlas_password_visible", False)
        icon_name = "view-conceal-symbolic" if visible else "view-reveal-symbolic"

        try:
            icon_setter(icon_position.SECONDARY, icon_name)
        except Exception:  # pragma: no cover - optional icon support
            pass

    def _set_password_entry_visibility(self, entry: Gtk.Entry, visible: bool) -> None:
        entry.set_visibility(bool(visible))
        setattr(entry, "_atlas_password_visible", bool(visible))
        self._set_password_entry_icon(entry)

    def _on_password_icon_pressed(self, entry: Gtk.Entry, icon_pos, _event) -> None:
        icon_position = getattr(Gtk, "EntryIconPosition", None)
        if icon_position is not None and icon_pos != icon_position.SECONDARY:
            return

        current = getattr(entry, "_atlas_password_visible", False)
        self._set_password_entry_visibility(entry, not current)

    def _update_password_strength_label(self, entry: Gtk.Entry, label: Gtk.Label) -> None:
        password = entry.get_text() or ""
        label.set_text(self._describe_password_strength(password))

    @staticmethod
    def _describe_password_strength(password: str) -> str:
        password = password or ""
        if not password:
            return ""

        score = 0

        length = len(password)
        if length >= 8:
            score += 1
        if length >= 12:
            score += 1

        has_lower = any(char.islower() for char in password)
        has_upper = any(char.isupper() for char in password)
        has_digit = any(char.isdigit() for char in password)
        has_symbol = any(not char.isalnum() for char in password)

        if has_lower and has_upper:
            score += 1
        if has_digit:
            score += 1
        if has_symbol:
            score += 1

        descriptions = {
            0: "Weak password",
            1: "Weak password",
            2: "Fair password",
            3: "Strong password",
        }

        if score >= 4:
            return "Very strong password"

        return descriptions.get(score, "Strong password")

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
        self.login_password_entry.set_placeholder_text("Your password")
        grid.attach(self.login_password_entry, 1, 1, 1, 1)
        self._configure_password_entry(self.login_password_entry)

        login_toggle = self._create_password_toggle(self.login_password_entry)
        grid.attach(login_toggle, 2, 1, 1, 1)

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
        self.register_password_entry.set_placeholder_text("Create a password")
        grid.attach(self.register_password_entry, 1, 2, 1, 1)
        register_password_toggle = self._create_password_toggle(self.register_password_entry)
        grid.attach(register_password_toggle, 2, 2, 1, 1)

        confirm_label = Gtk.Label(label="Confirm password")
        confirm_label.set_xalign(0.0)
        grid.attach(confirm_label, 0, 3, 1, 1)

        self.register_confirm_entry = Gtk.Entry()
        self.register_confirm_entry.set_placeholder_text("Repeat password")
        grid.attach(self.register_confirm_entry, 1, 3, 1, 1)
        register_confirm_toggle = self._create_password_toggle(self.register_confirm_entry)
        grid.attach(register_confirm_toggle, 2, 3, 1, 1)

        self.register_password_strength_label = Gtk.Label()
        self.register_password_strength_label.set_xalign(0.0)
        grid.attach(self.register_password_strength_label, 1, 4, 2, 1)

        self._configure_password_entry(
            self.register_password_entry,
            strength_label=self.register_password_strength_label,
        )
        self._configure_password_entry(self.register_confirm_entry)
        self.register_password_entry.connect("changed", self._on_register_password_changed)
        self.register_confirm_entry.connect("changed", self._on_register_confirm_changed)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 5, 1, 1)

        self.register_name_entry = Gtk.Entry()
        self.register_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.register_name_entry, 1, 5, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 6, 1, 1)

        self.register_dob_entry = Gtk.Entry()
        self.register_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.register_dob_entry, 1, 6, 1, 1)

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
        self.edit_password_entry.set_placeholder_text("Leave blank to keep current password")
        grid.attach(self.edit_password_entry, 1, 2, 1, 1)

        edit_password_toggle = self._create_password_toggle(self.edit_password_entry)
        grid.attach(edit_password_toggle, 2, 2, 1, 1)

        confirm_label = Gtk.Label(label="Confirm password")
        confirm_label.set_xalign(0.0)
        grid.attach(confirm_label, 0, 3, 1, 1)

        self.edit_confirm_entry = Gtk.Entry()
        self.edit_confirm_entry.set_placeholder_text("Repeat new password")
        grid.attach(self.edit_confirm_entry, 1, 3, 1, 1)

        edit_confirm_toggle = self._create_password_toggle(self.edit_confirm_entry)
        grid.attach(edit_confirm_toggle, 2, 3, 1, 1)

        self.edit_password_strength_label = Gtk.Label()
        self.edit_password_strength_label.set_xalign(0.0)
        grid.attach(self.edit_password_strength_label, 1, 4, 2, 1)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 5, 1, 1)

        self.edit_name_entry = Gtk.Entry()
        self.edit_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.edit_name_entry, 1, 5, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 6, 1, 1)

        self.edit_dob_entry = Gtk.Entry()
        self.edit_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.edit_dob_entry, 1, 6, 1, 1)

        self._configure_password_entry(
            self.edit_password_entry,
            strength_label=self.edit_password_strength_label,
        )
        self._configure_password_entry(self.edit_confirm_entry)
        self.edit_password_entry.connect("changed", self._on_edit_password_changed)
        self.edit_confirm_entry.connect("changed", self._on_edit_confirm_changed)

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
            self._update_toggle_buttons()
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
        login_active = self._active_form == "login"
        register_active = self._active_form == "register"

        if login_active:
            self.login_toggle_button.add_css_class("suggested-action")
        else:
            self.login_toggle_button.remove_css_class("suggested-action")

        if register_active:
            self.register_toggle_button.add_css_class("suggested-action")
        else:
            self.register_toggle_button.remove_css_class("suggested-action")

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

    def _username_validation_error(self, username: str) -> Optional[str]:
        if not username:
            return "Username is required."

        if not self._USERNAME_PATTERN.fullmatch(username):
            return (
                "Username must be 3-32 characters using letters, numbers, dots, hyphens or underscores."
            )

        return None

    def _display_name_validation_error(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None

        if len(name) > self._MAX_DISPLAY_NAME_LENGTH:
            return f"Display name must be {self._MAX_DISPLAY_NAME_LENGTH} characters or fewer."

        return None

    def _dob_validation_error(self, dob: Optional[str]) -> Optional[str]:
        if not dob:
            return None

        if not self._DOB_PATTERN.fullmatch(dob):
            return "Enter date of birth as YYYY-MM-DD."

        try:
            parsed = _dt.date.fromisoformat(dob)
        except ValueError:
            return "Enter date of birth as YYYY-MM-DD."

        if parsed > _dt.date.today():
            return "Date of birth cannot be in the future."

        return None

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
            self.register_name_entry,
            self.register_dob_entry,
        ):
            self._mark_field_valid(widget)
        self.register_password_strength_label.set_text("")

    def _on_register_clicked(self, _button) -> None:
        username = (self.register_username_entry.get_text() or "").strip()
        email = (self.register_email_entry.get_text() or "").strip()
        password = self.register_password_entry.get_text() or ""
        confirm = self.register_confirm_entry.get_text() or ""
        name = (self.register_name_entry.get_text() or "").strip() or None
        dob = (self.register_dob_entry.get_text() or "").strip() or None

        self._clear_registration_validation()

        errors: list[tuple[str, object]] = []

        username_error = self._username_validation_error(username)
        if username_error:
            errors.append((username_error, self.register_username_entry))

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

        name_error = self._display_name_validation_error(name)
        if name_error:
            errors.append((name_error, self.register_name_entry))

        dob_error = self._dob_validation_error(dob)
        if dob_error:
            errors.append((dob_error, self.register_dob_entry))

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

    def _on_register_password_changed(self, _entry) -> None:
        confirm_text = self.register_confirm_entry.get_text() or ""
        password = self.register_password_entry.get_text() or ""

        if not confirm_text:
            self._mark_field_valid(self.register_confirm_entry)
            return

        if password == confirm_text:
            self._mark_field_valid(self.register_confirm_entry)
        else:
            self._mark_field_invalid(self.register_confirm_entry)

    def _on_register_confirm_changed(self, entry) -> None:
        confirm_text = entry.get_text() or ""
        password = self.register_password_entry.get_text() or ""

        if not confirm_text:
            self._mark_field_valid(entry)
            return

        if confirm_text == password:
            self._mark_field_valid(entry)
        else:
            self._mark_field_invalid(entry)

    # ------------------------------------------------------------------
    # Account editing flow
    # ------------------------------------------------------------------
    def _clear_edit_validation(self) -> None:
        for widget in (
            self.edit_email_entry,
            self.edit_password_entry,
            self.edit_confirm_entry,
            self.edit_name_entry,
            self.edit_dob_entry,
        ):
            self._mark_field_valid(widget)
        self.edit_password_strength_label.set_text("")

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

        name_error = self._display_name_validation_error(name or None)
        if name_error:
            errors.append((name_error, self.edit_name_entry))

        dob_error = self._dob_validation_error(dob or None)
        if dob_error:
            errors.append((dob_error, self.edit_dob_entry))

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

    def _on_edit_password_changed(self, _entry) -> None:
        confirm_text = self.edit_confirm_entry.get_text() or ""
        password = self.edit_password_entry.get_text() or ""

        if not confirm_text:
            self._mark_field_valid(self.edit_confirm_entry)
            return

        if password == confirm_text:
            self._mark_field_valid(self.edit_confirm_entry)
        else:
            self._mark_field_invalid(self.edit_confirm_entry)

    def _on_edit_confirm_changed(self, entry) -> None:
        confirm_text = entry.get_text() or ""
        password = self.edit_password_entry.get_text() or ""

        if not confirm_text:
            self._mark_field_valid(entry)
            return

        if password == confirm_text:
            self._mark_field_valid(entry)
        else:
            self._mark_field_invalid(entry)

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
