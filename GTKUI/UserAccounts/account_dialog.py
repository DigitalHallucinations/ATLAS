"""Dialog for managing ATLAS user accounts."""

from __future__ import annotations

import datetime as _dt
import logging
import re
from concurrent.futures import Future
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

try:  # pragma: no cover - Gio may be unavailable in some environments
    from gi.repository import Gio
except Exception:  # pragma: no cover - fall back gracefully when Gio is missing
    Gio = None  # type: ignore[assignment]

from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css, create_box
from modules.user_accounts.user_account_service import (
    AccountLockedError,
    DuplicateUserError,
    InvalidCurrentPasswordError,
    PasswordRequirements,
)


class AccountDialog(AtlasWindow):
    """Provide login, registration, and logout flows for ATLAS accounts."""

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
    _DOB_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _MAX_DISPLAY_NAME_LENGTH = 80
    _STALE_ACCOUNT_THRESHOLD_DAYS = 90

    def __init__(self, atlas, parent: Optional[Gtk.Window] = None) -> None:
        super().__init__(
            title="Account Management",
            modal=True,
            transient_for=parent,
            default_size=(420, 420),
        )
        self.logger = logging.getLogger(__name__)
        self.ATLAS = atlas
        self._active_user_listener = None
        self._active_username: Optional[str] = None
        self._active_display_name: Optional[str] = None
        self._editing_username: Optional[str] = None
        self._editing_metadata: dict[str, object] = {}
        self._account_records: list[dict[str, object]] = []
        self._account_details_cache: dict[str, dict[str, object]] = {}
        self._account_details_history_strings: list[str] = []
        self._status_filter_value: str = "all"
        self._status_filter_options: list[tuple[str, str]] = []
        self._account_filter_text: str = ""
        self._active_account_request: Optional[tuple[str, str]] = None
        self._active_account_task: Optional[Future] = None
        self._active_summary_task: Optional[Future] = None
        self._selected_username: Optional[str] = None
        self._details_busy = False
        self._login_busy = False
        self._register_busy = False
        self._visible_usernames: list[str] = []
        self._password_toggle_buttons: list[Gtk.Widget] = []
        self._password_toggle_buttons_by_entry: dict[Gtk.Entry, list[Gtk.Widget]] = {}
        self._password_requirements: Optional[PasswordRequirements] = None
        self._password_requirements_text: str = ""
        self._password_requirement_labels: list[Gtk.Label] = []
        self._password_requirement_tooltip_widgets: list[Gtk.Widget] = []
        self._login_lockout_timeout_id: Optional[int] = None
        self._login_lockout_remaining_seconds: Optional[int] = None
        self._is_closing = False
        self._is_closed = False
        self._password_reset_prompt_queue: Optional[list[str]] = None
        self._last_password_reset_message: Optional[str] = None
        self._account_summary: dict[str, object] = {}
        self._last_account_context: str = "list"

        self._build_ui()
        self._initialise_password_requirements()
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
            self.forgot_password_button,
            self.register_button,
            self.edit_save_button,
            self.logout_button,
            self.account_search_entry,
            self.account_status_filter,
        ]
        self._forms_sensitive_widgets.extend(self._password_toggle_buttons)
        self._set_forms_busy(False)
        self._set_account_summary_message("Loading summary…")

    def _default_password_requirements(self) -> PasswordRequirements:
        return PasswordRequirements(
            min_length=10,
            require_uppercase=True,
            require_lowercase=True,
            require_digit=True,
            require_symbol=True,
            forbid_whitespace=True,
        )

    def _get_password_requirements(self) -> PasswordRequirements:
        if self._password_requirements is None:
            self._password_requirements = self._default_password_requirements()
        return self._password_requirements

    def _password_requirement_lines(self) -> list[str]:
        requirements = self._get_password_requirements()
        return requirements.bullet_points()

    def _password_requirement_text(self) -> str:
        lines = self._password_requirement_lines()
        if not lines:
            return ""
        return "\n".join(f"• {line}" for line in lines)

    def _update_password_requirement_labels(self) -> None:
        text = self._password_requirement_text()
        for label in self._password_requirement_labels:
            label.set_text(text)
        tooltip = self._password_requirements_text or text
        for widget in self._password_requirement_tooltip_widgets:
            setter = getattr(widget, "set_tooltip_text", None)
            if callable(setter):
                try:
                    setter(tooltip)
                except Exception:  # pragma: no cover - stub compatibility
                    continue

    def _initialise_password_requirements(self) -> None:
        requirements = self._default_password_requirements()
        description: Optional[str] = None

        getter = getattr(self.ATLAS, "get_user_password_requirements", None)
        if callable(getter):
            try:
                candidate = getter()
                if isinstance(candidate, PasswordRequirements):
                    requirements = candidate
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.debug("Failed to fetch password requirements: %s", exc)

        describer = getattr(self.ATLAS, "describe_user_password_requirements", None)
        if callable(describer):
            try:
                description = str(describer()) or None
            except Exception as exc:  # pragma: no cover - descriptive text optional
                self.logger.debug("Failed to describe password requirements: %s", exc)

        resolved_description = description or requirements.describe()

        if (
            self._password_requirements is None
            or requirements != self._password_requirements
            or resolved_description != self._password_requirements_text
        ):
            self._password_requirements = requirements
            self._password_requirements_text = resolved_description
            self._update_password_requirement_labels()

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

        self.account_summary_label = Gtk.Label()
        self.account_summary_label.set_xalign(0.0)
        wrapper.append(self.account_summary_label)

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

        status_label = Gtk.Label(label="Status:")
        status_label.set_xalign(0.0)
        search_row.append(status_label)

        self.account_status_filter = self._create_status_filter_widget()
        search_row.append(self.account_status_filter)

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

    def _create_status_filter_widget(self) -> Gtk.Widget:
        options = [
            ("all", "All accounts"),
            ("active", "Active"),
            ("locked", "Locked"),
            ("inactive", f"Inactive {self._STALE_ACCOUNT_THRESHOLD_DAYS}+ days"),
            ("never", "Never signed in"),
        ]

        self._status_filter_options = options
        self._status_filter_value = "all"

        drop_down_cls = getattr(Gtk, "DropDown", None)
        if drop_down_cls is not None:
            labels = [label for _value, label in options]
            constructor = getattr(drop_down_cls, "new_from_strings", None)
            widget = None
            if callable(constructor):
                try:
                    widget = constructor(labels)
                except Exception:
                    widget = None
            if widget is None:
                string_list_cls = getattr(Gtk, "StringList", None)
                if string_list_cls is not None:
                    try:
                        model = string_list_cls.new(labels)  # type: ignore[attr-defined]
                    except AttributeError:
                        model = string_list_cls()
                        append = getattr(model, "append", None)
                        if callable(append):
                            for label in labels:
                                append(label)
                    try:
                        widget = drop_down_cls(model=model)
                    except Exception:
                        widget = None

            if widget is not None:
                set_selected = getattr(widget, "set_selected", None)
                if callable(set_selected):
                    try:
                        set_selected(0)
                    except Exception:
                        pass
                widget.connect("notify::selected", self._on_status_filter_notify)
                return widget

        combo_cls = getattr(Gtk, "ComboBoxText", None)
        if combo_cls is not None:
            combo = combo_cls()
            for value, label in options:
                append = getattr(combo, "append", None)
                if callable(append):
                    append(value, label)
            set_active = getattr(combo, "set_active", None)
            if callable(set_active):
                try:
                    set_active(0)
                except Exception:
                    pass
            combo.connect("changed", self._on_status_filter_combo_changed)
            return combo

        fallback = Gtk.Button(label="All accounts")
        fallback.connect("clicked", lambda *_: None)
        return fallback

    def _on_status_filter_notify(self, widget, _param) -> None:
        getter = getattr(widget, "get_selected", None)
        if callable(getter):
            try:
                index = int(getter())
            except Exception:
                index = 0
            self._set_status_filter_from_index(index)

    def _on_status_filter_combo_changed(self, widget) -> None:
        get_active_id = getattr(widget, "get_active_id", None)
        if callable(get_active_id):
            value = get_active_id()
            if value is not None:
                self._update_status_filter_value(str(value))
                return

        get_active = getattr(widget, "get_active", None)
        if callable(get_active):
            try:
                index = int(get_active())
            except Exception:
                index = 0
            self._set_status_filter_from_index(index)

    def _set_status_filter_from_index(self, index: int) -> None:
        if 0 <= index < len(self._status_filter_options):
            value = self._status_filter_options[index][0]
        else:
            value = "all"
        self._update_status_filter_value(value)

    def _update_status_filter_value(self, value: str) -> None:
        value = value or "all"
        if value == self._status_filter_value:
            self._refresh_filtered_accounts()
            return
        self._status_filter_value = value
        self._refresh_filtered_accounts()

    def _set_account_summary_message(self, message: str) -> None:
        label = getattr(self, "account_summary_label", None)
        if isinstance(label, Gtk.Label):
            label.set_text(message)

    def _format_account_summary(self, summary: dict[str, object]) -> str:
        total = int(summary.get("total_accounts", 0) or 0)
        parts = [f"Total accounts: {total}"]

        active_display = str(summary.get("active_display_name") or "").strip()
        if active_display:
            parts.append(f"Active: {active_display}")

        locked = int(summary.get("locked_accounts", 0) or 0)
        if locked:
            parts.append(f"Locked: {locked}")

        stale = int(summary.get("stale_accounts", 0) or 0)
        if stale:
            parts.append(
                f"Inactive {self._STALE_ACCOUNT_THRESHOLD_DAYS}+ days: {stale}"
            )

        never_signed_in = int(summary.get("never_signed_in", 0) or 0)
        if never_signed_in:
            parts.append(f"Never signed in: {never_signed_in}")

        latest_username = str(summary.get("latest_sign_in_username") or "").strip()
        latest_timestamp = summary.get("latest_sign_in_at")
        if latest_username and latest_timestamp:
            formatted = self._format_last_login_detail(latest_timestamp)
            parts.append(f"Latest sign-in: {latest_username} ({formatted})")

        return " • ".join(parts)

    def _update_account_summary_display(self, summary: dict[str, object]) -> None:
        self._account_summary = dict(summary)
        message = self._format_account_summary(summary)
        self._set_account_summary_message(message)

    def _cancel_active_summary_task(self) -> None:
        task = self._active_summary_task
        self._active_summary_task = None
        if task is None:
            return
        try:
            if not task.done():
                task.cancel()
        except Exception:  # pragma: no cover - defensive cancellation
            pass

    def _request_account_summary(self) -> None:
        getter = getattr(self.ATLAS, "get_user_account_overview", None)
        if not callable(getter):
            self._set_account_summary_message("Summary unavailable.")
            return

        self._cancel_active_summary_task()
        self._set_account_summary_message("Loading summary…")

        future: Optional[Future] = None

        def factory():
            return getter()

        def on_success(result) -> None:
            GLib.idle_add(self._handle_account_summary_result, result, future)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_summary_error, exc, future)

        try:
            future = self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-summary",
            )
            self._active_summary_task = future
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to request account summary: %s", exc, exc_info=True)
            self._handle_account_summary_error(exc)

    def _handle_account_summary_result(
        self, summary, future: Optional[Future] = None
    ) -> bool:
        if future is not None:
            if future.cancelled():
                return False
            if future is not self._active_summary_task:
                return False
            self._active_summary_task = None

        if not isinstance(summary, dict):
            summary = {}

        self._update_account_summary_display(summary)
        return False

    def _handle_account_summary_error(
        self, exc: Exception, future: Optional[Future] = None
    ) -> bool:
        if future is not None and future is not self._active_summary_task:
            return False
        if future is self._active_summary_task:
            self._active_summary_task = None

        self.logger.debug("Account summary failed: %s", exc)
        self._set_account_summary_message("Summary unavailable.")
        return False
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
        self.account_details_age_value = _add_row(4, "Age")
        self.account_details_last_login_value = _add_row(5, "Last sign-in")

        history_header = Gtk.Label(label="Recent sign-in attempts")
        history_header.set_xalign(0.0)
        content.append(history_header)

        self.account_details_history_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=4,
            margin=0,
        )
        setter = getattr(self.account_details_history_box, "set_margin_top", None)
        if callable(setter):
            try:
                setter(6)
            except Exception:  # pragma: no cover - defensive for stub environments
                pass
        content.append(self.account_details_history_box)

        self.account_details_history_placeholder = Gtk.Label(
            label="No recent activity."
        )
        self.account_details_history_placeholder.set_xalign(0.0)
        self.account_details_history_box.append(
            self.account_details_history_placeholder
        )

        return frame

    # ------------------------------------------------------------------
    # Account list handling
    # ------------------------------------------------------------------
    def _refresh_account_list(self, *_args) -> None:
        if self._account_busy:
            return

        self._request_account_summary()
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

    def _handle_account_list_result(
        self, accounts, future: Optional[Future] = None
    ) -> bool:
        if future is not None:
            if future.cancelled():
                return False
            if future is not self._active_account_task:
                return False

        if self._is_closed:
            return False
        if future is self._active_account_task:
            self._active_account_task = None
        self._active_account_request = None
        normalised = self._normalise_account_payload(accounts)
        self._last_account_context = "list"
        self._account_records = normalised

        known_usernames = {entry["username"] for entry in normalised}
        self._account_details_cache = {
            username: details
            for username, details in self._account_details_cache.items()
            if username in known_usernames
        }
        if self._selected_username not in known_usernames:
            self._selected_username = None

        previous_selection = self._selected_username or None
        active_username = self._active_username or ""
        visible_accounts = self._filter_accounts_for_display(normalised, context="list")
        active_visible = bool(active_username) and any(
            account.get("username") == active_username for account in visible_accounts
        )
        should_focus_active = active_visible and previous_selection is None
        message = self._render_account_rows(visible_accounts, context="load")

        if self._post_refresh_feedback:
            message = f"{self._post_refresh_feedback} {message}".strip()
            self._post_refresh_feedback = None

        self._set_account_busy(False, message, disable_forms=False)

        if should_focus_active:
            self._focus_active_account_row(trigger_details=True)
        return False

    def _handle_account_list_error(
        self, exc: Exception, future: Optional[Future] = None
    ) -> bool:
        if future is not None:
            if future.cancelled():
                return False
            if future is not self._active_account_task:
                return False

        if self._is_closed:
            return False
        if future is self._active_account_task:
            self._active_account_task = None
        self._active_account_request = None
        self._account_rows.clear()
        self._visible_usernames = []

        removed = False
        remover = getattr(self.account_list_box, "remove_all", None)
        if callable(remover):
            try:
                remover()
                removed = True
            except Exception:  # pragma: no cover - fallback for stub environments
                removed = False

        if not removed:
            get_children = getattr(self.account_list_box, "get_children", None)
            children = []
            if callable(get_children):
                try:
                    children = list(get_children())
                except Exception:  # pragma: no cover - compatibility with stubs
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

        self._selected_username = None
        self._set_account_detail_message("Select an account to view details.")
        self._update_account_buttons_sensitive()
        self.account_empty_label.set_text("Failed to load saved accounts.")
        self._set_account_summary_message("Summary unavailable.")
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

            account_data = dict(source)
            account_data["username"] = username
            account_data["display_name"] = _string_or_empty(display_name) or username
            account_data["email"] = _string_or_empty(source.get("email"))
            account_data["name"] = _string_or_empty(source.get("name"))
            account_data["dob"] = _string_or_empty(source.get("dob"))
            account_data["last_login"] = _string_or_empty(source.get("last_login"))

            if "status_badge" in source:
                account_data["status_badge"] = str(source.get("status_badge") or "")

            if "is_active" in source:
                account_data["is_active"] = bool(source.get("is_active"))

            if "is_locked" in source:
                account_data["is_locked"] = bool(source.get("is_locked"))

            if "last_login_age_days" in source:
                age_value = source.get("last_login_age_days")
                try:
                    account_data["last_login_age_days"] = float(age_value)
                except (TypeError, ValueError):
                    account_data["last_login_age_days"] = None

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

    def _apply_status_filter(self, accounts: list[dict[str, object]]) -> list[dict[str, object]]:
        value = self._status_filter_value or "all"
        if value in {"all", ""}:
            return list(accounts)

        filtered: list[dict[str, object]] = []
        threshold_days = self._STALE_ACCOUNT_THRESHOLD_DAYS

        for account in accounts:
            metadata = account if isinstance(account, dict) else {}
            if value == "active":
                if bool(metadata.get("is_active")):
                    filtered.append(account)
                continue

            if value == "locked":
                if bool(metadata.get("is_locked")):
                    filtered.append(account)
                continue

            if value == "never":
                last_login = str(metadata.get("last_login") or "").strip()
                if not last_login:
                    filtered.append(account)
                continue

            if value == "inactive":
                age_value = metadata.get("last_login_age_days")
                try:
                    age = float(age_value)
                except (TypeError, ValueError):
                    age = None
                if age is not None and age >= threshold_days:
                    filtered.append(account)
                continue

        return filtered

    def _filter_accounts_for_display(
        self, accounts: list[dict[str, object]], *, context: str
    ) -> list[dict[str, object]]:
        filtered = self._apply_account_filter(accounts)
        filtered = self._apply_status_filter(filtered)
        return filtered

    def _refresh_filtered_accounts(self) -> None:
        if self._account_busy:
            return

        records = getattr(self, "_account_records", None)
        if not isinstance(records, list):
            return

        filtered = self._filter_accounts_for_display(records, context=self._last_account_context)
        message = self._render_account_rows(filtered, context=self._last_account_context)
        self.account_feedback_label.set_text(message)

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

        ordered_accounts = self._sort_accounts_for_display(list(accounts))

        count = 0
        for account_data in ordered_accounts:
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
            self.account_details_age_value.set_text(placeholder)
            self.account_details_last_login_value.set_text(placeholder)
            self._show_login_history_placeholder()

    def _apply_account_detail_mapping(self, details: dict[str, object]) -> None:
        username = self._format_detail_value(details.get("username"))
        display_name = details.get("display_name") or details.get("name")
        email = self._format_detail_value(details.get("email"))
        dob = self._format_detail_value(details.get("dob"))
        age = self._format_age_detail(details.get("dob"))
        last_login = self._format_last_login_detail(details.get("last_login"))

        self.account_details_status_label.set_text("")
        self.account_details_username_value.set_text(username)
        self.account_details_name_value.set_text(self._format_detail_value(display_name))
        self.account_details_email_value.set_text(email)
        self.account_details_dob_value.set_text(dob)
        self.account_details_age_value.set_text(age)
        self.account_details_last_login_value.set_text(last_login)
        self._update_login_history(details.get("login_attempts"))

    @staticmethod
    def _format_detail_value(value: object, fallback: str = "—") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text or fallback

    def _update_login_history(self, attempts: object) -> None:
        entries = self._format_login_history_entries(attempts)
        if not entries:
            self._show_login_history_placeholder()
            return

        self._clear_login_history_widgets()

        for text in entries:
            label = Gtk.Label(label=text)
            label.set_xalign(0.0)
            self._append_to_history_container(label)

        self._account_details_history_strings = entries

    def _format_login_history_entries(self, attempts: object) -> list[str]:
        if not isinstance(attempts, list):
            return []

        entries: list[str] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue

            timestamp_text = self._format_last_login_detail(attempt.get("timestamp"))
            successful = bool(attempt.get("successful"))

            if successful:
                status = "Signed in"
            else:
                reason_text = self._describe_login_attempt_reason(attempt.get("reason"))
                status = "Failed"
                if reason_text:
                    status = f"Failed ({reason_text})"

            entries.append(f"{timestamp_text} — {status}")

        return entries

    def _describe_login_attempt_reason(self, reason: object) -> str:
        if reason is None:
            return ""

        text = str(reason).strip()
        if not text:
            return ""

        mapping = {
            "invalid-credentials": "Invalid credentials",
            "unknown-identifier": "Unknown account",
            "invalid-identifier": "Invalid identifier",
            "missing-password": "Missing password",
            "account-locked": "Account locked",
            "error": "Error",
        }

        if text in mapping:
            return mapping[text]

        return text.replace("-", " ").title()

    def _append_to_history_container(self, widget: Gtk.Widget) -> None:
        container = getattr(self, "account_details_history_box", None)
        if container is None:
            return

        append = getattr(container, "append", None)
        if callable(append):
            append(widget)
            return

        add = getattr(container, "add", None)
        if callable(add):
            add(widget)
            return

        children = getattr(container, "children", None)
        if isinstance(children, list):
            children.append(widget)

    def _clear_login_history_widgets(self) -> None:
        container = getattr(self, "account_details_history_box", None)
        if container is None:
            self._account_details_history_strings = []
            return

        remove = getattr(container, "remove", None)
        get_first_child = getattr(container, "get_first_child", None)
        if callable(remove) and callable(get_first_child):
            child = get_first_child()
            while child is not None:
                remove(child)
                child = get_first_child()
        else:
            children = getattr(container, "children", None)
            if isinstance(children, list):
                children.clear()

        self._account_details_history_strings = []

    def _show_login_history_placeholder(self) -> None:
        placeholder = getattr(self, "account_details_history_placeholder", None)
        if placeholder is None:
            placeholder = Gtk.Label(label="No recent activity.")
            placeholder.set_xalign(0.0)
            self.account_details_history_placeholder = placeholder

        self._clear_login_history_widgets()
        self._append_to_history_container(placeholder)
        self._account_details_history_strings = []

    @staticmethod
    def _parse_last_login(value: object) -> Optional[_dt.datetime]:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        candidate = text.replace("Z", "+00:00")

        try:
            parsed = _dt.datetime.fromisoformat(candidate)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        else:
            parsed = parsed.astimezone(_dt.timezone.utc)

        return parsed

    def _format_last_login_for_row(self, value: object) -> str:
        parsed = self._parse_last_login(value)
        if parsed is None:
            return "Last sign-in: never"

        return f"Last sign-in: {parsed.strftime('%Y-%m-%d %H:%M')} UTC"

    def _format_last_login_detail(self, value: object) -> str:
        parsed = self._parse_last_login(value)
        if parsed is None:
            return "—"

        return parsed.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _current_utc_time() -> _dt.datetime:
        return _dt.datetime.now(_dt.timezone.utc)

    def _compute_age_from_dob(self, dob_value: object) -> Optional[int]:
        if dob_value is None:
            return None

        text = str(dob_value).strip()
        if not text:
            return None

        try:
            dob_date = _dt.date.fromisoformat(text)
        except ValueError:
            return None

        today = self._current_utc_time().date()
        computed = today.year - dob_date.year - (
            (today.month, today.day) < (dob_date.month, dob_date.day)
        )
        return max(computed, 0)

    def _format_age_detail(self, dob_value: object) -> str:
        age = self._compute_age_from_dob(dob_value)
        if age is None:
            return "—"
        return str(age)

    def _determine_account_badge(self, metadata: dict[str, object]) -> str:
        if bool(metadata.get("is_locked")):
            return "Locked"

        if bool(metadata.get("is_active")):
            return "Active"

        last_login_value = metadata.get("last_login") if isinstance(metadata, dict) else None
        parsed = self._parse_last_login(last_login_value)
        if parsed is None:
            return "Never signed in"

        threshold = self._current_utc_time() - _dt.timedelta(days=self._STALE_ACCOUNT_THRESHOLD_DAYS)
        if parsed < threshold:
            return f"Inactive {self._STALE_ACCOUNT_THRESHOLD_DAYS}+ days"

        return ""

    def _sort_accounts_for_display(self, accounts: list[dict[str, object]]) -> list[dict[str, object]]:
        def _sort_key(account: dict[str, object]) -> tuple[float, str]:
            parsed = self._parse_last_login(account.get("last_login"))
            timestamp = parsed.timestamp() if parsed is not None else float("-inf")
            username = str(account.get("username") or "").lower()
            return (-timestamp, username)

        return sorted(accounts, key=_sort_key)

    def _on_account_filter_changed(self, entry, *_args) -> None:
        text = getattr(entry, "get_text", lambda: "")()
        query = (text or "").strip()
        self._account_filter_text = query

        if query:
            self._start_account_search(query)
        else:
            self._start_account_list_refresh()

    def _on_account_search_clear(self, _button) -> None:
        setter = getattr(self.account_search_entry, "set_text", None)
        if callable(setter):
            setter("")
        self._on_account_filter_changed(self.account_search_entry)

    def _cancel_active_account_task(self) -> None:
        task = self._active_account_task
        self._active_account_task = None
        if task is None:
            return

        try:
            if not task.done():
                task.cancel()
        except Exception:  # pragma: no cover - defensive cancellation handling
            pass

    def _start_account_search(self, query: str) -> None:
        self._cancel_active_account_task()
        self._active_account_request = ("search", query)
        self._set_account_busy(True, "Searching accounts…", disable_forms=False)
        self.account_empty_label.set_text("Searching…")

        future: Optional[Future] = None

        def factory():
            return self.ATLAS.search_user_accounts(query)

        def on_success(result) -> None:
            GLib.idle_add(self._handle_account_search_result, query, result, future)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_search_error, query, exc, future)

        try:
            future = self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-search",
            )
            self._active_account_task = future
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start account search: %s", exc, exc_info=True)
            self._handle_account_search_error(query, exc)

    def _start_account_list_refresh(self) -> None:
        self._cancel_active_account_task()
        self._active_account_request = ("list", "")
        self._set_account_busy(True, "Loading saved accounts…", disable_forms=False)

        self._request_account_summary()

        future: Optional[Future] = None

        def factory():
            return self.ATLAS.list_user_accounts()

        def on_success(result) -> None:
            GLib.idle_add(self._handle_account_list_result, result, future)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_account_list_error, exc, future)

        try:
            future = self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="user-account-list",
            )
            self._active_account_task = future
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to refresh account list: %s", exc, exc_info=True)
            self._handle_account_list_error(exc)

    def _handle_account_search_result(
        self, query: str, accounts, future: Optional[Future] = None
    ) -> bool:
        if future is not None:
            if future.cancelled():
                return False
            if future is not self._active_account_task:
                return False

        if query != self._account_filter_text or self._active_account_request != ("search", query):
            return False

        if future is self._active_account_task:
            self._active_account_task = None
        self._active_account_request = None
        normalised = self._normalise_account_payload(accounts)
        self._last_account_context = "search"
        self._account_records = normalised
        filtered_accounts = self._filter_accounts_for_display(normalised, context="search")
        message = self._render_account_rows(filtered_accounts, context="search")
        self._set_account_busy(False, message, disable_forms=False)
        return False

    def _handle_account_search_error(
        self, query: str, exc: Exception, future: Optional[Future] = None
    ) -> bool:
        if future is not None:
            if future.cancelled():
                return False
            if future is not self._active_account_task:
                return False

        if query != self._account_filter_text or self._active_account_request != ("search", query):
            return False

        if future is self._active_account_task:
            self._active_account_task = None
        self._active_account_request = None
        self.account_empty_label.set_text("Search failed.")
        message = f"Search failed: {exc}"
        self._set_account_busy(False, message, disable_forms=False)
        return False

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
        if self._is_closed:
            return False
        if self._selected_username != username:
            return False

        self._set_account_details_busy(False)

        if isinstance(result, dict) and result.get("username"):
            mapping = dict(result)
            mapping.setdefault("display_name", mapping.get("name"))
            self._account_details_cache[username] = mapping
            self._apply_account_detail_mapping(mapping)
            row_widgets = self._account_rows.get(username)
            if row_widgets:
                combined = dict(row_widgets.get("metadata") or {})
                combined.update(mapping)
                combined["username"] = username
                self._apply_account_row_metadata(row_widgets, combined)
        else:
            self._set_account_detail_message("No additional details available.")

        return False

    def _handle_account_details_error(self, username: str, exc: Exception) -> bool:
        if self._is_closed:
            return False
        if self._selected_username != username:
            return False

        self._set_account_details_busy(False)
        self._set_account_detail_message(f"Failed to load details: {exc}")
        return False

    def _create_account_row(self, account_data: dict[str, object]) -> dict[str, Gtk.Widget]:
        username = str(account_data.get("username") or "").strip()

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

        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        row_box.append(info_box)

        name_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        info_box.append(name_row)

        name_label = Gtk.Label()
        name_label.set_xalign(0.0)
        try:
            name_label.set_hexpand(True)
        except Exception:  # pragma: no cover - compatibility for stubs lacking hexpand
            pass
        name_row.append(name_label)

        badge_label = Gtk.Label()
        badge_label.set_xalign(0.0)
        name_row.append(badge_label)

        email_label = Gtk.Label()
        email_label.set_xalign(0.0)
        add_dim = getattr(email_label, "add_css_class", None)
        if callable(add_dim):
            try:
                add_dim("dim-label")
            except Exception:  # pragma: no cover - CSS support varies in tests
                pass
        info_box.append(email_label)

        last_login_label = Gtk.Label()
        last_login_label.set_xalign(0.0)
        add_dim = getattr(last_login_label, "add_css_class", None)
        if callable(add_dim):
            try:
                add_dim("dim-label")
            except Exception:  # pragma: no cover - CSS support varies in tests
                pass
        info_box.append(last_login_label)

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

        row_widgets = {
            "row": container,
            "row_box": row_box,
            "name_label": name_label,
            "active_label": badge_label,
            "email_label": email_label,
            "last_login_label": last_login_label,
            "use_button": use_button,
            "details_button": details_button,
            "edit_button": edit_button,
            "delete_button": delete_button,
        }
        self._apply_account_row_metadata(row_widgets, account_data)
        return row_widgets

    def _apply_account_row_metadata(
        self, widgets: dict[str, Gtk.Widget], account_data: dict[str, object]
    ) -> None:
        metadata = dict(account_data) if isinstance(account_data, dict) else {}
        username = str(metadata.get("username") or "").strip()
        metadata["username"] = username
        widgets["metadata"] = metadata

        display_name = metadata.get("display_name") or metadata.get("name") or username
        widgets["name_label"].set_text(str(display_name or username))

        email_label = widgets.get("email_label")
        if isinstance(email_label, Gtk.Label):
            email_value = str(metadata.get("email") or "").strip()
            email_label.set_text(email_value or "—")

        last_login_label = widgets.get("last_login_label")
        if isinstance(last_login_label, Gtk.Label):
            last_login_label.set_text(self._format_last_login_for_row(metadata.get("last_login")))

        badge_text = str(metadata.get("status_badge") or metadata.get("_status_badge") or "")
        if not badge_text:
            badge_text = self._determine_account_badge(metadata)
        metadata["status_badge"] = badge_text

        badge_label = widgets.get("active_label")
        if isinstance(badge_label, Gtk.Label):
            badge_label.set_text(badge_text)
            setter = getattr(badge_label, "set_tooltip_text", None)
            if callable(setter):
                try:
                    setter(badge_text or None)
                except Exception:  # pragma: no cover - defensive tooltip handling
                    pass

        row_box = widgets.get("row_box")
        self._set_css_class(row_box, "warning", bool(metadata.get("is_locked")))

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
        try:
            return bool(self._show_fallback_delete_dialog(username))
        except Exception:  # pragma: no cover - defensive fallback
            return False

    def _style_dialog(self, dialog) -> None:
        """Apply ATLAS styling to dialogs in a defensive manner."""
        try:
            apply_css()
        except Exception:  # pragma: no cover - styling is best-effort
            pass

        get_style_context = getattr(dialog, "get_style_context", None)
        if not callable(get_style_context):
            return

        try:
            style_context = get_style_context()
        except Exception:  # pragma: no cover - widget quirks
            return

        if style_context is None:
            return

        for class_name in ("chat-page", "sidebar"):
            for method_name in ("add_css_class", "add_class"):
                add_class = getattr(style_context, method_name, None)
                if callable(add_class):
                    try:
                        add_class(class_name)
                    except Exception:  # pragma: no cover - tolerate stub quirks
                        pass
                    else:
                        break

    def _close_dialog(self) -> None:
        closer = getattr(self, "close", None)
        if callable(closer):
            try:
                closer()
                return
            except Exception:  # pragma: no cover - stub environments may lack close support
                pass
        setattr(self, "closed", True)

    def _show_fallback_delete_dialog(self, username: str) -> bool:
        message = f'Delete saved account "{username}"?'

        dialog = None
        confirm_responses: set[object] = set()
        cancel_responses: set[object] = {None, False}

        message_dialog_cls = getattr(Gtk, "MessageDialog", None)
        if message_dialog_cls is not None:
            try:
                kwargs = {
                    "transient_for": self if isinstance(self, Gtk.Window) else None,
                    "modal": True,
                    "text": "Delete account?",
                }
                message_type = getattr(getattr(Gtk, "MessageType", object), "WARNING", None)
                if message_type is not None:
                    kwargs["message_type"] = message_type
                buttons_type = getattr(getattr(Gtk, "ButtonsType", object), "NONE", None)
                if buttons_type is not None:
                    kwargs["buttons"] = buttons_type
                dialog = message_dialog_cls(**kwargs)
                self._style_dialog(dialog)
                setter = getattr(dialog, "set_secondary_text", None)
                if callable(setter):
                    try:
                        setter(message)
                    except Exception:
                        pass
            except Exception:
                dialog = None

        if dialog is None:
            dialog_cls = getattr(Gtk, "Dialog", None)
            if dialog_cls is None:
                return False
            try:
                dialog = dialog_cls(transient_for=self if isinstance(self, Gtk.Window) else None, modal=True)
                self._style_dialog(dialog)
            except Exception:
                return False
            content_area = getattr(dialog, "get_content_area", None)
            if callable(content_area):
                try:
                    container = content_area()
                except Exception:
                    container = None
                if container is not None:
                    label_cls = getattr(Gtk, "Label", None)
                    if label_cls is not None:
                        try:
                            label = label_cls(label="Delete account?")
                        except Exception:
                            label = None
                        if label is not None:
                            try:
                                setter = getattr(label, "set_text", None)
                                if callable(setter):
                                    setter(message)
                            except Exception:
                                pass
                            try:
                                container.append(label)
                            except Exception:
                                pass

        if dialog is None:
            return False

        response_type = getattr(Gtk, "ResponseType", object)
        accept_value = getattr(response_type, "ACCEPT", None)
        ok_value = getattr(response_type, "OK", None)
        yes_value = getattr(response_type, "YES", None)
        confirm_responses.update({accept_value, ok_value, yes_value, True, "delete", "accept", "ok", "yes"})
        cancel_value = getattr(response_type, "CANCEL", None)
        no_value = getattr(response_type, "NO", None)
        reject_value = getattr(response_type, "REJECT", None)
        cancel_responses.update({cancel_value, no_value, reject_value, "cancel", "reject", "no"})

        add_button = getattr(dialog, "add_button", None)
        if callable(add_button):
            cancel_resp = cancel_value if cancel_value is not None else -1
            confirm_resp = accept_value if accept_value is not None else (-2 if cancel_resp != -2 else -3)
            try:
                add_button("Cancel", cancel_resp)
                add_button("Delete", confirm_resp)
                confirm_responses.add(confirm_resp)
                cancel_responses.add(cancel_resp)
            except Exception:
                pass

        set_buttons = getattr(dialog, "set_buttons", None)
        if callable(set_buttons):
            try:
                set_buttons(["Cancel", "Delete"])
            except Exception:
                pass

        present = getattr(dialog, "present", None) or getattr(dialog, "show", None)
        if callable(present):
            try:
                present()
            except Exception:
                pass

        run = getattr(dialog, "run", None)
        response = None
        if callable(run):
            try:
                response = run()
            except Exception:
                response = None

        destroy = getattr(dialog, "destroy", None) or getattr(dialog, "close", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass

        confirm_responses.discard(None)
        cancel_responses.discard(None)

        if isinstance(response, str):
            return response.lower() in {"delete", "accept", "ok", "yes"}
        if isinstance(response, bool):
            return response
        if response in confirm_responses:
            return True
        if response in cancel_responses:
            return False
        if isinstance(response, int):
            if response == -2 and -2 in confirm_responses:
                return True
            if response == -1 and -1 in cancel_responses:
                return False
        return False

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
        if self._is_closed:
            return False
        self._set_account_busy(False, "Account activated.", disable_forms=True)
        self._highlight_active_account()
        return False

    def _handle_delete_success(self, username: str) -> bool:
        if self._is_closed:
            return False
        self._set_account_busy(False, "Account deleted.", disable_forms=True)
        self._post_refresh_feedback = "Account deleted."
        self._refresh_account_list()
        return False

    def _handle_account_action_error(self, exc: Exception) -> bool:
        if self._is_closed:
            return False
        self._set_account_busy(False, f"Account action failed: {exc}", disable_forms=True)
        return False

    def _set_forms_busy(self, busy: bool) -> None:
        if self._is_closed:
            return
        self._forms_busy = bool(busy)
        for widget in getattr(self, "_forms_sensitive_widgets", []):
            self._set_widget_sensitive(widget, not busy)

    def _set_account_busy(self, busy: bool, message: str | None = None, *, disable_forms: bool = True) -> None:
        if self._is_closed:
            return
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

    def _select_list_box_row(self, row: Gtk.Widget) -> None:
        list_box = getattr(self, "account_list_box", None)
        if not list_box or row is None:
            return

        select_row = getattr(list_box, "select_row", None)
        if callable(select_row):
            try:
                select_row(row)
            except TypeError:
                try:
                    select_row(row=row)
                except Exception:  # pragma: no cover - fall back quietly when signature differs
                    pass
            except Exception:  # pragma: no cover - compatibility with stub widgets
                pass

        scroll_to_row = getattr(list_box, "scroll_to_row", None)
        if callable(scroll_to_row):
            try:
                scroll_to_row(row)
            except TypeError:
                try:
                    scroll_to_row(row, None, False, 0.0, 0.0)
                except Exception:  # pragma: no cover - tolerate differing GTK signatures
                    pass
            except Exception:  # pragma: no cover - stub compatibility
                pass

    def _focus_active_account_row(self, *, trigger_details: bool = False) -> None:
        active_username = self._active_username or ""
        active_row = None

        for username, widgets in self._account_rows.items():
            is_active = username == active_username
            self._set_row_active(widgets, is_active)
            if is_active:
                active_row = widgets.get("row")

        if active_row is not None:
            self._select_list_box_row(active_row)

            if trigger_details and active_username:
                if self._selected_username != active_username:
                    self._on_account_details_clicked(active_username)

    def _highlight_active_account(self) -> None:
        self._focus_active_account_row()

    def _set_row_active(self, widgets: dict[str, Gtk.Widget], active: bool) -> None:
        metadata = widgets.get("metadata") if isinstance(widgets, dict) else {}
        badge_text = ""
        if isinstance(metadata, dict):
            badge_text = str(metadata.get("status_badge") or metadata.get("_status_badge") or "")

        label = widgets["active_label"]
        label.set_text("Active" if active else badge_text)
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

    @staticmethod
    def _set_css_class(widget, css_class: str, enabled: bool) -> None:
        if widget is None:
            return

        if enabled:
            adder = getattr(widget, "add_css_class", None)
            if callable(adder):
                try:
                    adder(css_class)
                except Exception:  # pragma: no cover - CSS support may vary
                    pass
            return

        remover = getattr(widget, "remove_css_class", None)
        if callable(remover):
            try:
                remover(css_class)
            except Exception:  # pragma: no cover - defensive removal
                pass

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

    def _register_password_toggle(self, entry: Gtk.Entry, toggle: Gtk.Widget) -> None:
        toggles = self._password_toggle_buttons_by_entry.setdefault(entry, [])
        if toggle not in toggles:
            toggles.append(toggle)
        if toggle not in self._password_toggle_buttons:
            self._password_toggle_buttons.append(toggle)

    def _create_password_toggle(self, entry: Gtk.Entry) -> Gtk.Widget:
        toggle_cls = getattr(Gtk, "ToggleButton", None)

        if toggle_cls is not None:
            toggle = toggle_cls(label="Show")

            def _on_toggled(button):
                self._register_password_toggle(entry, button)
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

            def _stub_set_active(state: bool, *, _toggle=toggle) -> None:
                setattr(_toggle, "_atlas_toggle_active", bool(state))

            def _stub_get_active(*, _toggle=toggle) -> bool:
                return bool(getattr(_toggle, "_atlas_toggle_active", False))

            setattr(toggle, "set_active", _stub_set_active)
            setattr(toggle, "get_active", _stub_get_active)

            def _on_clicked(button):
                self._register_password_toggle(entry, button)
                current = getattr(button, "_atlas_toggle_active", False)
                new_state = not current
                setattr(button, "_atlas_toggle_active", new_state)
                self._set_password_entry_visibility(entry, new_state)
                setter = getattr(button, "set_label", None)
                if callable(setter):
                    setter("Hide" if new_state else "Show")
                self._set_password_entry_icon(entry)

            toggle.connect("clicked", _on_clicked)

        self._register_password_toggle(entry, toggle)
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
        toggles = self._password_toggle_buttons_by_entry.get(entry, [])
        for toggle in toggles:
            label_setter = getattr(toggle, "set_label", None)
            if callable(label_setter):
                try:
                    label_setter("Hide" if visible else "Show")
                except Exception:  # pragma: no cover - defensive for stub widgets
                    pass

            set_active = getattr(toggle, "set_active", None)
            if callable(set_active):
                getter = getattr(toggle, "get_active", None)
                try:
                    current_state = bool(getter()) if callable(getter) else None
                except Exception:  # pragma: no cover - fall back to unconditional set
                    current_state = None
                try:
                    if current_state is None or current_state != bool(visible):
                        set_active(bool(visible))
                except Exception:  # pragma: no cover - defensive for stub widgets
                    pass

            setattr(toggle, "_atlas_toggle_active", bool(visible))

    def _on_password_icon_pressed(self, entry: Gtk.Entry, icon_pos, _event) -> None:
        icon_position = getattr(Gtk, "EntryIconPosition", None)
        if icon_position is not None and icon_pos != icon_position.SECONDARY:
            return

        current = getattr(entry, "_atlas_password_visible", False)
        self._set_password_entry_visibility(entry, not current)

    def _update_password_strength_label(self, entry: Gtk.Entry, label: Gtk.Label) -> None:
        password = entry.get_text() or ""
        label.set_text(self._describe_password_strength(password))

    def _describe_password_strength(self, password: str) -> str:
        password = password or ""
        if not password:
            return ""

        requirements = self._get_password_requirements()

        missing: list[str] = []
        if requirements.forbid_whitespace and any(char.isspace() for char in password):
            missing.append("remove spaces")
        if requirements.require_lowercase and not any(char.islower() for char in password):
            missing.append("add a lowercase letter")
        if requirements.require_uppercase and not any(char.isupper() for char in password):
            missing.append("add an uppercase letter")
        if requirements.require_digit and not any(char.isdigit() for char in password):
            missing.append("add a number")
        if requirements.require_symbol and not any(
            (not char.isalnum()) and (not char.isspace()) for char in password
        ):
            missing.append("add a symbol")

        if len(password) < requirements.min_length:
            missing.append(f"use {requirements.min_length} or more characters")

        if missing:
            if len(missing) == 1:
                return f"Weak password – {missing[0]}."
            return "Weak password – " + ", ".join(missing[:-1]) + f", and {missing[-1]}."

        length = len(password)
        diversity = sum(
            1
            for check in (
                any(char.islower() for char in password),
                any(char.isupper() for char in password),
                any(char.isdigit() for char in password),
                any(
                    (not char.isalnum()) and (not char.isspace()) for char in password
                ),
            )
            if check
        )

        score = 0
        if length >= requirements.min_length:
            score += 1
        if length >= requirements.min_length + 4:
            score += 1
        score += min(diversity, 3)

        if score >= 5:
            return "Very strong password"
        if score >= 4:
            return "Strong password"
        return "Fair password"

    def _build_login_form(self) -> Gtk.Widget:
        wrapper = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=0)

        grid = Gtk.Grid(column_spacing=8, row_spacing=8)
        wrapper.append(grid)

        username_label = Gtk.Label(label="Username or email")
        username_label.set_xalign(0.0)
        grid.attach(username_label, 0, 0, 1, 1)

        self.login_username_entry = Gtk.Entry()
        self.login_username_entry.set_placeholder_text("Your username or email")
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

        self.forgot_password_button = Gtk.Button(label="Forgot password?")
        self.forgot_password_button.connect(
            "clicked",
            self._on_forgot_password_clicked,
        )
        wrapper.append(self.forgot_password_button)

        def trigger_login_from_entry(*_args) -> None:
            self._on_login_clicked(self.login_button)

        self.login_username_entry.connect("activate", trigger_login_from_entry)
        self.login_password_entry.connect("activate", trigger_login_from_entry)

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
        self._password_requirement_tooltip_widgets.extend(
            [self.register_password_entry, self.register_confirm_entry]
        )

        self.register_password_strength_label = Gtk.Label()
        self.register_password_strength_label.set_xalign(0.0)
        grid.attach(self.register_password_strength_label, 1, 4, 2, 1)

        self.register_password_requirements_label = Gtk.Label()
        self.register_password_requirements_label.set_xalign(0.0)
        try:
            self.register_password_requirements_label.set_wrap(True)
            justification = getattr(Gtk.Justification, "LEFT", None)
            if justification is not None:
                self.register_password_requirements_label.set_justify(justification)
        except Exception:  # pragma: no cover - wrap support varies in tests
            pass
        grid.attach(self.register_password_requirements_label, 1, 5, 2, 1)
        self._password_requirement_labels.append(self.register_password_requirements_label)

        self._configure_password_entry(
            self.register_password_entry,
            strength_label=self.register_password_strength_label,
        )
        self._configure_password_entry(self.register_confirm_entry)
        self.register_password_entry.connect("changed", self._on_register_password_changed)
        self.register_confirm_entry.connect("changed", self._on_register_confirm_changed)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 6, 1, 1)

        self.register_name_entry = Gtk.Entry()
        self.register_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.register_name_entry, 1, 6, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 7, 1, 1)

        self.register_dob_entry = Gtk.Entry()
        self.register_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.register_dob_entry, 1, 7, 1, 1)

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

        current_password_label = Gtk.Label(label="Current password")
        current_password_label.set_xalign(0.0)
        grid.attach(current_password_label, 0, 2, 1, 1)

        self.edit_current_password_entry = Gtk.Entry()
        self.edit_current_password_entry.set_placeholder_text("Required to change password")
        grid.attach(self.edit_current_password_entry, 1, 2, 1, 1)

        current_password_toggle = self._create_password_toggle(self.edit_current_password_entry)
        grid.attach(current_password_toggle, 2, 2, 1, 1)

        password_label = Gtk.Label(label="New password")
        password_label.set_xalign(0.0)
        grid.attach(password_label, 0, 3, 1, 1)

        self.edit_password_entry = Gtk.Entry()
        self.edit_password_entry.set_placeholder_text("Leave blank to keep current password")
        grid.attach(self.edit_password_entry, 1, 3, 1, 1)

        edit_password_toggle = self._create_password_toggle(self.edit_password_entry)
        grid.attach(edit_password_toggle, 2, 3, 1, 1)

        confirm_label = Gtk.Label(label="Confirm password")
        confirm_label.set_xalign(0.0)
        grid.attach(confirm_label, 0, 4, 1, 1)

        self.edit_confirm_entry = Gtk.Entry()
        self.edit_confirm_entry.set_placeholder_text("Repeat new password")
        grid.attach(self.edit_confirm_entry, 1, 4, 1, 1)

        edit_confirm_toggle = self._create_password_toggle(self.edit_confirm_entry)
        grid.attach(edit_confirm_toggle, 2, 4, 1, 1)
        self._password_requirement_tooltip_widgets.extend(
            [self.edit_password_entry, self.edit_confirm_entry]
        )

        self.edit_password_strength_label = Gtk.Label()
        self.edit_password_strength_label.set_xalign(0.0)
        grid.attach(self.edit_password_strength_label, 1, 5, 2, 1)

        self.edit_password_requirements_label = Gtk.Label()
        self.edit_password_requirements_label.set_xalign(0.0)
        try:
            self.edit_password_requirements_label.set_wrap(True)
            justification = getattr(Gtk.Justification, "LEFT", None)
            if justification is not None:
                self.edit_password_requirements_label.set_justify(justification)
        except Exception:  # pragma: no cover - wrap support varies in tests
            pass
        grid.attach(self.edit_password_requirements_label, 1, 6, 2, 1)
        self._password_requirement_labels.append(self.edit_password_requirements_label)

        name_label = Gtk.Label(label="Display name (optional)")
        name_label.set_xalign(0.0)
        grid.attach(name_label, 0, 7, 1, 1)

        self.edit_name_entry = Gtk.Entry()
        self.edit_name_entry.set_placeholder_text("How should we greet you?")
        grid.attach(self.edit_name_entry, 1, 7, 1, 1)

        dob_label = Gtk.Label(label="Date of birth (optional)")
        dob_label.set_xalign(0.0)
        grid.attach(dob_label, 0, 8, 1, 1)

        self.edit_dob_entry = Gtk.Entry()
        self.edit_dob_entry.set_placeholder_text("YYYY-MM-DD")
        grid.attach(self.edit_dob_entry, 1, 8, 1, 1)

        self._configure_password_entry(
            self.edit_password_entry,
            strength_label=self.edit_password_strength_label,
        )
        self._configure_password_entry(self.edit_confirm_entry)
        self._configure_password_entry(self.edit_current_password_entry)
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
        entry = getattr(self, "login_username_entry", None)
        if entry is not None:
            if username:
                entry.set_text(username)
            elif not persisted:
                entry.set_text("")
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
    def _login_lockout_active(self) -> bool:
        return bool(self._login_lockout_remaining_seconds and self._login_lockout_remaining_seconds > 0)

    def _set_login_controls_sensitive(self, sensitive: bool) -> None:
        self._set_widget_sensitive(self.login_button, sensitive)
        self._set_widget_sensitive(self.login_username_entry, sensitive)
        self._set_widget_sensitive(self.login_password_entry, sensitive)
        self._set_widget_sensitive(getattr(self, "forgot_password_button", None), sensitive)

    def _cancel_login_lockout_timer(self) -> None:
        timeout_id = self._login_lockout_timeout_id
        self._login_lockout_timeout_id = None
        self._login_lockout_remaining_seconds = None
        if timeout_id is not None:
            remover = getattr(GLib, "source_remove", None)
            if callable(remover):
                try:
                    remover(timeout_id)
                except Exception:  # pragma: no cover - defensive for stub environments
                    pass
        if not self._login_busy and not self._is_closing:
            self._set_login_controls_sensitive(True)

    def _update_login_lockout_feedback(self) -> None:
        remaining = self._login_lockout_remaining_seconds
        if remaining is None or remaining <= 0:
            return
        plural = "s" if remaining != 1 else ""
        self.login_feedback_label.set_text(
            f"Too many failed login attempts. Try again in {remaining} second{plural}."
        )

    def _start_login_lockout_timer(self, seconds: int) -> None:
        seconds = max(int(seconds), 0)
        if seconds <= 0:
            self._cancel_login_lockout_timer()
            return

        self._cancel_login_lockout_timer()
        self._login_lockout_remaining_seconds = seconds
        self._set_login_controls_sensitive(False)
        self._update_login_lockout_feedback()

        try:
            timeout_id = GLib.timeout_add_seconds(1, self._on_login_lockout_tick)
        except Exception:  # pragma: no cover - stub environments may not implement timers
            timeout_id = None
        self._login_lockout_timeout_id = timeout_id

    def _on_login_lockout_tick(self) -> bool:
        if self._login_lockout_remaining_seconds is None:
            self._login_lockout_timeout_id = None
            return False

        remaining = self._login_lockout_remaining_seconds - 1
        if remaining <= 0:
            self._login_lockout_remaining_seconds = None
            self._login_lockout_timeout_id = None
            if not self._login_busy:
                self._set_login_controls_sensitive(True)
            self.login_feedback_label.set_text("You can try signing in again.")
            return False

        self._login_lockout_remaining_seconds = remaining
        self._update_login_lockout_feedback()
        return True

    def _set_login_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self._login_busy = bool(busy)
        controls_sensitive = not busy and not self._login_lockout_active()
        self._set_login_controls_sensitive(controls_sensitive)

        shared_sensitive = not busy and not self._forms_busy
        self._set_widget_sensitive(getattr(self, "login_toggle_button", None), shared_sensitive)
        self._set_widget_sensitive(getattr(self, "register_toggle_button", None), shared_sensitive)

        register_sensitive = shared_sensitive and not self._register_busy
        register_widgets = [
            getattr(self, "register_box", None),
            getattr(self, "register_button", None),
            getattr(self, "register_username_entry", None),
            getattr(self, "register_email_entry", None),
            getattr(self, "register_password_entry", None),
            getattr(self, "register_confirm_entry", None),
            getattr(self, "register_name_entry", None),
            getattr(self, "register_dob_entry", None),
        ]

        for widget in register_widgets:
            self._set_widget_sensitive(widget, register_sensitive)

        for entry in (
            getattr(self, "register_password_entry", None),
            getattr(self, "register_confirm_entry", None),
        ):
            if entry is None:
                continue
            for toggle in self._password_toggle_buttons_by_entry.get(entry, []):
                self._set_widget_sensitive(toggle, register_sensitive)
        if message is not None:
            self.login_feedback_label.set_text(message)

    def _on_login_clicked(self, _button) -> None:
        if self._login_busy or self._login_lockout_active():
            return

        self._cancel_login_lockout_timer()

        # Clear any previous validation errors when starting a new attempt.
        self._mark_field_valid(self.login_username_entry)
        self._mark_field_valid(self.login_password_entry)

        username = (self.login_username_entry.get_text() or "").strip()
        password = self.login_password_entry.get_text() or ""

        if not username or not password:
            self.login_feedback_label.set_text("Username or email and password are required.")
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
        if self._is_closed:
            return False
        self._set_login_busy(False)
        self._cancel_login_lockout_timer()
        if success:
            self.login_feedback_label.set_text("Signed in successfully.")
            self._close_dialog()
        else:
            self.login_feedback_label.set_text("Invalid username/email or password.")
            self._mark_field_invalid(self.login_username_entry)
            self._mark_field_invalid(self.login_password_entry)
            try:
                self.login_password_entry.grab_focus()
            except Exception:  # pragma: no cover - grab_focus may be unavailable in stubs
                pass
        return False

    def _handle_login_error(self, exc: Exception) -> bool:
        if self._is_closed:
            return False
        self._set_login_busy(False)
        if isinstance(exc, AccountLockedError):
            retry_after = getattr(exc, "retry_after", None)
            seconds = 0
            if retry_after is not None:
                try:
                    seconds = int(retry_after)
                except (TypeError, ValueError):  # pragma: no cover - defensive casting
                    seconds = 0
            if seconds > 0:
                self._start_login_lockout_timer(seconds)
            else:
                self._cancel_login_lockout_timer()
                self.login_feedback_label.set_text(str(exc))
        else:
            self._cancel_login_lockout_timer()
            self.login_feedback_label.set_text(f"Login failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Password reset flow
    # ------------------------------------------------------------------
    def _prompt_for_value(
        self,
        title: str,
        message: str,
        *,
        placeholder: str = "",
        is_secret: bool = False,
    ) -> Optional[str]:
        queue = getattr(self, "_password_reset_prompt_queue", None)
        if isinstance(queue, list):
            if queue:
                return queue.pop(0)
            return None

        alert_cls = getattr(Gtk, "AlertDialog", None)
        if alert_cls is None:
            self.logger.warning("Password reset prompt unavailable: Gtk.AlertDialog missing.")
            return None

        entry = Gtk.Entry()
        if placeholder:
            setter = getattr(entry, "set_placeholder_text", None)
            if callable(setter):
                try:
                    setter(placeholder)
                except Exception:  # pragma: no cover - stub compatibility
                    pass
        if is_secret:
            visibility = getattr(entry, "set_visibility", None)
            if callable(visibility):
                try:
                    visibility(False)
                except Exception:  # pragma: no cover - stub compatibility
                    pass

        dialog = alert_cls(title=title, body=message)
        setter = getattr(dialog, "set_extra_child", None)
        if callable(setter):
            try:
                setter(entry)
            except Exception:  # pragma: no cover - stub compatibility
                pass
        button_setter = getattr(dialog, "set_buttons", None)
        if callable(button_setter):
            try:
                button_setter(["Cancel", "Submit"])
            except Exception:  # pragma: no cover
                pass

        try:
            future = dialog.choose(self)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to present password reset dialog: %s", exc, exc_info=True)
            return None

        result: Optional[object] = None
        try:
            wait_result = getattr(future, "wait_result", None)
            if callable(wait_result):
                result = wait_result()
            else:
                wait = getattr(future, "wait", None)
                result = wait() if callable(wait) else None
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Password reset dialog wait failed: %s", exc, exc_info=True)
            return None

        accept_values: set[object] = {"Submit", 1}
        response_type = getattr(Gtk, "ResponseType", None)
        if response_type is not None:
            accept_values.add(getattr(response_type, "ACCEPT", None))
            accept_values.add(getattr(response_type, "OK", None))

        if result not in accept_values:
            if isinstance(result, str) and result.lower() in {"ok", "accept"}:
                pass
            else:
                return None

        getter = getattr(entry, "get_text", None)
        text = getter() if callable(getter) else ""
        cleaned = (text or "").strip()
        return cleaned or None

    def _show_password_reset_message(self, title: str, message: str) -> None:
        self._last_password_reset_message = message
        dialog_cls = getattr(Gtk, "AlertDialog", None)
        if dialog_cls is None:
            self.login_feedback_label.set_text(message)
            return

        dialog = dialog_cls(title=title, body=message)
        button_setter = getattr(dialog, "set_buttons", None)
        if callable(button_setter):
            try:
                button_setter(["OK"])
            except Exception:  # pragma: no cover
                pass
        try:
            future = dialog.choose(self)
            waiter = getattr(future, "wait_result", None) or getattr(future, "wait", None)
            if callable(waiter):
                waiter()
        except Exception:  # pragma: no cover - dialog best-effort
            pass

    def _on_forgot_password_clicked(self, _button) -> None:
        if self._login_busy:
            return

        identifier = self._prompt_for_value(
            "Reset password",
            "Enter your username or email to reset your password.",
            placeholder="Username or email",
        )
        if not identifier:
            self.login_feedback_label.set_text("Password reset cancelled.")
            return

        self._set_login_busy(True, "Requesting password reset…")

        def factory():
            return self.ATLAS.request_password_reset(identifier)

        def on_success(result: Optional[dict]) -> None:
            GLib.idle_add(self._handle_password_reset_challenge, identifier, result)

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_password_reset_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="password-reset-request",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to start password reset request: %s", exc, exc_info=True)
            self._handle_password_reset_error(exc)

    def _handle_password_reset_challenge(
        self,
        identifier: str,
        result: Optional[dict[str, object]],
    ) -> bool:
        if self._is_closed:
            return False

        self._set_login_busy(False)

        if not result:
            self.login_feedback_label.set_text(
                "If the account exists, a password reset token has been generated."
            )
            return False

        username = result.get("username") if isinstance(result, dict) else None
        token = result.get("token") if isinstance(result, dict) else None
        expires_at = result.get("expires_at") if isinstance(result, dict) else None

        parts = []
        if username:
            parts.append(f"A reset token has been generated for {username}.")
        if token:
            parts.append(f"Token: {token}")
        if expires_at:
            parts.append(f"Expires at: {expires_at}")

        if parts:
            self._show_password_reset_message("Password reset", "\n".join(parts))
        else:
            self._show_password_reset_message(
                "Password reset",
                "A password reset token has been generated.",
            )

        self.login_feedback_label.set_text(
            "Enter the reset token to update your password."
        )

        if not username:
            username = identifier

        self._prompt_for_reset_token(str(username))
        return False

    def _prompt_for_reset_token(self, username: str) -> None:
        token = self._prompt_for_value(
            "Enter reset token",
            f"Enter the reset token for {username}.",
            placeholder="Reset token",
        )
        if not token:
            self.login_feedback_label.set_text("Password reset cancelled.")
            return

        self._verify_reset_token(username, token)

    def _verify_reset_token(self, username: str, token: str) -> None:
        self._set_login_busy(True, "Verifying reset token…")

        def factory():
            return self.ATLAS.verify_password_reset_token(username, token)

        def on_success(valid: bool) -> None:
            GLib.idle_add(self._handle_password_reset_token_verified, username, token, bool(valid))

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_password_reset_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="password-reset-verify",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to verify password reset token: %s", exc, exc_info=True)
            self._handle_password_reset_error(exc)

    def _handle_password_reset_token_verified(
        self,
        username: str,
        token: str,
        valid: bool,
    ) -> bool:
        if self._is_closed:
            return False

        self._set_login_busy(False)

        if not valid:
            self.login_feedback_label.set_text("Invalid or expired password reset token.")
            return False

        new_password = self._prompt_for_value(
            "Choose a new password",
            "Enter your new password.",
            is_secret=True,
        )
        if not new_password:
            self.login_feedback_label.set_text("Password reset cancelled.")
            return False

        confirm_password = self._prompt_for_value(
            "Confirm new password",
            "Re-enter your new password.",
            is_secret=True,
        )
        if new_password != confirm_password:
            self.login_feedback_label.set_text("Passwords do not match.")
            return False

        validation_error = self._password_validation_error(new_password)
        if validation_error:
            self.login_feedback_label.set_text(validation_error)
            return False

        self._finalise_password_reset(username, token, new_password)
        return False

    def _finalise_password_reset(self, username: str, token: str, password: str) -> None:
        self._set_login_busy(True, "Updating password…")

        def factory():
            return self.ATLAS.complete_password_reset(username, token, password)

        def on_success(success: bool) -> None:
            GLib.idle_add(self._handle_password_reset_complete, username, bool(success))

        def on_error(exc: Exception) -> None:
            GLib.idle_add(self._handle_password_reset_error, exc)

        try:
            self.ATLAS.run_in_background(
                factory,
                on_success=on_success,
                on_error=on_error,
                thread_name="password-reset-complete",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to finalise password reset: %s", exc, exc_info=True)
            self._handle_password_reset_error(exc)

    def _handle_password_reset_complete(self, username: str, success: bool) -> bool:
        if self._is_closed:
            return False

        self._set_login_busy(False)

        if not success:
            self.login_feedback_label.set_text("Password reset failed. Please try again.")
            return False

        self.login_feedback_label.set_text(
            "Password reset complete. You can sign in with your new password."
        )
        try:
            self.login_username_entry.set_text(username)
            self.login_password_entry.grab_focus()
        except Exception:  # pragma: no cover - stubs may not implement setters
            pass
        return False

    def _handle_password_reset_error(self, exc: Exception) -> bool:
        if self._is_closed:
            return False

        self._set_login_busy(False)
        self.login_feedback_label.set_text(f"Password reset failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # Registration flow
    # ------------------------------------------------------------------
    def _set_register_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self._register_busy = bool(busy)
        sensitive = not busy

        widgets = [
            self.register_button,
            self.register_username_entry,
            self.register_email_entry,
            self.register_password_entry,
            self.register_confirm_entry,
            self.register_name_entry,
            self.register_dob_entry,
        ]

        for widget in widgets:
            self._set_widget_sensitive(widget, sensitive)

        for entry in (self.register_password_entry, self.register_confirm_entry):
            for toggle in self._password_toggle_buttons_by_entry.get(entry, []):
                self._set_widget_sensitive(toggle, sensitive)

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
        requirements = self._get_password_requirements()

        if len(password) < requirements.min_length:
            return f"Password must be at least {requirements.min_length} characters long."

        if requirements.forbid_whitespace and any(char.isspace() for char in password):
            return "Password cannot contain spaces."

        if requirements.require_lowercase and not any(char.islower() for char in password):
            return "Password must include a lowercase letter."

        if requirements.require_uppercase and not any(char.isupper() for char in password):
            return "Password must include an uppercase letter."

        if requirements.require_digit and not any(char.isdigit() for char in password):
            return "Password must include a number."

        if requirements.require_symbol and not any(
            (not char.isalnum()) and (not char.isspace()) for char in password
        ):
            return "Password must include a symbol such as ! or #."

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
        if self._is_closed:
            return False
        self._set_register_busy(False)
        username = result.get("username") if isinstance(result, dict) else None
        self.register_feedback_label.set_text(
            f"Account created for {username or 'new user'}." if username else "Account created successfully."
        )
        self._close_dialog()
        return False

    def _handle_register_error(self, exc: Exception) -> bool:
        if self._is_closed:
            return False
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
            self.edit_current_password_entry,
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
        self.edit_current_password_entry.set_text("")
        self.edit_password_entry.set_text("")
        self.edit_confirm_entry.set_text("")
        self.edit_feedback_label.set_text("")
        self._clear_edit_validation()

        self._show_form("edit")

        grabber = getattr(self.edit_email_entry, "grab_focus", None)
        if callable(grabber):
            grabber()

    def _set_edit_busy(self, busy: bool, message: Optional[str] = None) -> None:
        sensitive = not busy

        widgets = [
            self.edit_save_button,
            self.edit_username_entry,
            self.edit_email_entry,
            self.edit_current_password_entry,
            self.edit_password_entry,
            self.edit_confirm_entry,
            self.edit_name_entry,
            self.edit_dob_entry,
        ]

        for widget in widgets:
            self._set_widget_sensitive(widget, sensitive)

        password_entries = (
            self.edit_current_password_entry,
            self.edit_password_entry,
            self.edit_confirm_entry,
        )

        for entry in password_entries:
            for toggle in self._password_toggle_buttons_by_entry.get(entry, []):
                self._set_widget_sensitive(toggle, sensitive)

        if message is not None:
            self.edit_feedback_label.set_text(message)
        elif not busy:
            self.edit_feedback_label.set_text("")

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
        current_password = self.edit_current_password_entry.get_text() or ""
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

            if not current_password:
                errors.append(
                    (
                        "Enter your current password to change it.",
                        self.edit_current_password_entry,
                    )
                )

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
        current_password_arg = current_password if password_arg else None

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
                current_password=current_password_arg,
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
        if self._is_closed:
            return False
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
        self.edit_current_password_entry.set_text("")

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
        if self._is_closed:
            return False
        self._set_edit_busy(False)
        if isinstance(exc, DuplicateUserError):
            self.edit_feedback_label.set_text("Username or email already exists.")
            self._mark_field_invalid(self.edit_email_entry)
            grabber = getattr(self.edit_email_entry, "grab_focus", None)
            if callable(grabber):
                grabber()
        elif isinstance(exc, InvalidCurrentPasswordError):
            self.edit_feedback_label.set_text(str(exc))
            self._mark_field_invalid(self.edit_current_password_entry)
            grabber = getattr(self.edit_current_password_entry, "grab_focus", None)
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
        self._is_closing = True
        self._is_closed = True
        self._cancel_login_lockout_timer()
        self._cancel_active_account_task()
        self._cancel_active_summary_task()
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
