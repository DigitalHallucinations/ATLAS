"""Interactive GTK front-end for the ATLAS setup workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gtk

from ATLAS.setup import (
    DatabaseState,
    ProviderState,
    SetupWizardController as CoreSetupWizardController,
    UserState,
)
from ATLAS.setup_marker import write_setup_marker
from modules.conversation_store.bootstrap import BootstrapError

Callback = Callable[[], None]
ErrorCallback = Callable[[BaseException], None]


@dataclass
class WizardStep:
    name: str
    widget: Gtk.Widget
    apply: Callable[[], str]


class SetupWizardWindow(Gtk.Window):
    """A small multi-step wizard bound to :class:`SetupWizardController`."""

    def __init__(
        self,
        *,
        application: Gtk.Application,
        atlas: Any | None,
        on_success: Callback,
        on_error: ErrorCallback,
        error: BaseException | None = None,
        controller: CoreSetupWizardController | None = None,
    ) -> None:
        super().__init__()
        if hasattr(self, "set_title"):
            self.set_title("ATLAS Setup Utility")
        self.set_application(application)
        self._on_success = on_success
        self._on_error = on_error

        self.controller = controller or CoreSetupWizardController(atlas=atlas)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(18)
        root.set_margin_bottom(18)
        root.set_margin_start(18)
        root.set_margin_end(18)
        self.set_child(root)

        header = Gtk.Label(label="Complete the following steps to finish configuring ATLAS.")
        header.set_wrap(True)
        header.set_xalign(0.0)
        root.append(header)

        self._status_label = Gtk.Label()
        self._status_label.set_wrap(True)
        self._status_label.set_xalign(0.0)
        root.append(self._status_label)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._stack.set_vexpand(True)
        root.append(self._stack)

        self._switcher = Gtk.StackSwitcher()
        self._switcher.set_stack(self._stack)
        root.append(self._switcher)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.set_halign(Gtk.Align.END)
        root.append(controls)

        self._back_button = Gtk.Button(label="Back")
        self._back_button.connect("clicked", self._on_back_clicked)
        controls.append(self._back_button)

        self._next_button = Gtk.Button(label="Next")
        self._next_button.connect("clicked", self._on_next_clicked)
        controls.append(self._next_button)

        self._steps: List[WizardStep] = []
        self._current_index = 0

        self._database_entries: Dict[str, Gtk.Entry] = {}
        self._provider_entries: Dict[str, Gtk.Entry] = {}
        self._provider_buffer: Optional[Gtk.TextBuffer] = None
        self._user_entries: Dict[str, Gtk.Entry] = {}
        self._setup_persisted = False
        self._privileged_credentials: Optional[Tuple[Optional[str], Optional[str]]] = None

        self._build_steps()
        if error is not None:
            self.display_error(error)
        else:
            self._set_status("Welcome to the ATLAS setup wizard.")
        self._go_to_step(0)

    def display_error(self, error: BaseException) -> None:
        text = str(error) or error.__class__.__name__
        self._status_label.set_text(text)
        if hasattr(self._status_label, "add_css_class"):
            self._status_label.add_css_class("error-text")

    # -- UI helpers -----------------------------------------------------

    def _set_status(self, message: str) -> None:
        self._status_label.set_text(message)
        if hasattr(self._status_label, "remove_css_class"):
            self._status_label.remove_css_class("error-text")

    def _build_steps(self) -> None:
        self._steps = [
            WizardStep(
                name="Database",
                widget=self._build_database_page(),
                apply=self._apply_database,
            ),
            WizardStep(
                name="Providers",
                widget=self._build_provider_page(),
                apply=self._apply_providers,
            ),
            WizardStep(
                name="Administrator",
                widget=self._build_user_page(),
                apply=self._apply_user,
            ),
        ]

        for step in self._steps:
            self._stack.add_titled(step.widget, step.name.lower(), step.name)

    def _build_database_page(self) -> Gtk.Widget:
        state = self.controller.state.database
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        self._database_entries["host"] = self._create_labeled_entry(grid, 0, "Host", state.host)
        self._database_entries["port"] = self._create_labeled_entry(grid, 1, "Port", str(state.port))
        self._database_entries["database"] = self._create_labeled_entry(
            grid, 2, "Database", state.database
        )
        self._database_entries["user"] = self._create_labeled_entry(grid, 3, "User", state.user)
        self._database_entries["password"] = self._create_labeled_entry(
            grid, 4, "Password", state.password, visibility=False
        )

        help_label = Gtk.Label(
            label=(
                "Enter the PostgreSQL connection details. If the conversation store "
                "already exists, the wizard will reuse it."
            )
        )
        help_label.set_wrap(True)
        help_label.set_xalign(0.0)
        grid.attach(help_label, 0, 5, 2, 1)

        return grid

    def _build_provider_page(self) -> Gtk.Widget:
        state = self.controller.state.providers
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        self._provider_entries["default_provider"] = self._create_entry(
            "Default provider", state.default_provider or ""
        )
        self._provider_entries["default_model"] = self._create_entry(
            "Default model", state.default_model or ""
        )

        for entry in self._provider_entries.values():
            box.append(entry)

        keys_label = Gtk.Label(
            label=(
                "API keys (one per line using provider=key format). Existing values "
                "will be pre-populated when available."
            )
        )
        keys_label.set_wrap(True)
        keys_label.set_xalign(0.0)
        box.append(keys_label)

        view = Gtk.TextView()
        view.set_monospace(True)
        self._provider_buffer = view.get_buffer()
        self._provider_buffer.set_text(self._format_api_keys(state.api_keys))
        box.append(view)

        return box

    def _build_user_page(self) -> Gtk.Widget:
        state = self.controller.state.user
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        self._user_entries["username"] = self._create_labeled_entry(grid, 0, "Username", state.username)
        self._user_entries["email"] = self._create_labeled_entry(grid, 1, "Email", state.email)
        self._user_entries["display_name"] = self._create_labeled_entry(
            grid, 2, "Display name", state.display_name
        )
        self._user_entries["password"] = self._create_labeled_entry(
            grid, 3, "Password", state.password, visibility=False
        )
        self._user_entries["confirm_password"] = self._create_labeled_entry(
            grid, 4, "Confirm password", state.password, visibility=False
        )

        hint = Gtk.Label(
            label="Create the administrator account that will be used to sign in after setup."
        )
        hint.set_wrap(True)
        hint.set_xalign(0.0)
        grid.attach(hint, 0, 5, 2, 1)

        return grid

    def _create_labeled_entry(
        self,
        grid: Gtk.Grid,
        row: int,
        label: str,
        value: str,
        *,
        visibility: bool = True,
    ) -> Gtk.Entry:
        label_widget = Gtk.Label(label=label)
        label_widget.set_xalign(0.0)
        grid.attach(label_widget, 0, row, 1, 1)

        entry = Gtk.Entry()
        entry.set_visibility(visibility)
        entry.set_text(value)
        entry.set_hexpand(True)
        grid.attach(entry, 1, row, 1, 1)
        return entry

    def _create_entry(self, placeholder: str, value: str) -> Gtk.Entry:
        entry = Gtk.Entry()
        entry.set_placeholder_text(placeholder)
        entry.set_text(value)
        entry.set_hexpand(True)
        return entry

    def _format_api_keys(self, mapping: Dict[str, str]) -> str:
        lines = [f"{provider}={key}" for provider, key in sorted(mapping.items())]
        return "\n".join(lines)

    # -- navigation -----------------------------------------------------

    def _go_to_step(self, index: int) -> None:
        self._current_index = max(0, min(index, len(self._steps) - 1))
        step = self._steps[self._current_index]
        self._stack.set_visible_child(step.widget)
        self._update_navigation()

    def _update_navigation(self) -> None:
        if hasattr(self._back_button, "set_sensitive"):
            self._back_button.set_sensitive(self._current_index > 0)
        if hasattr(self._next_button, "set_label"):
            label = "Finish" if self._current_index == len(self._steps) - 1 else "Next"
            self._next_button.set_label(label)

    def _on_back_clicked(self, *_: object) -> None:
        if self._current_index > 0:
            self._go_to_step(self._current_index - 1)

    def _on_next_clicked(self, *_: object) -> None:
        step = self._steps[self._current_index]
        self._set_status("Applying changes…")
        try:
            message = step.apply()
        except BootstrapError as exc:
            self.display_error(exc)
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.display_error(exc)
            self._on_error(exc)
            return

        if self._current_index == len(self._steps) - 1:
            final_message = message or "Setup complete."
            if not self._setup_persisted:
                try:
                    summary = self.controller.build_summary()
                    write_setup_marker(summary)
                except IOError as exc:  # pragma: no cover - defensive
                    self.display_error(exc)
                    self._on_error(exc)
                    return
                else:
                    self._setup_persisted = True
            self._set_status(
                f"{final_message} You can now sign in with the new administrator account."
            )
            self._on_success()
        else:
            self._set_status(message or "Step complete.")
            self._go_to_step(self._current_index + 1)

    # -- apply handlers -------------------------------------------------

    def _apply_database(self) -> str:
        try:
            port = int(self._database_entries["port"].get_text().strip() or 0)
        except ValueError as exc:
            raise ValueError("Port must be a valid integer") from exc

        state = DatabaseState(
            host=self._database_entries["host"].get_text().strip() or "localhost",
            port=port or self.controller.state.database.port,
            database=self._database_entries["database"].get_text().strip() or "atlas",
            user=self._database_entries["user"].get_text().strip() or "atlas",
            password=self._database_entries["password"].get_text(),
        )

        try:
            self.controller.apply_database_settings(
                state,
                privileged_credentials=self._privileged_credentials,
            )
        except BootstrapError as exc:
            credentials = self._prompt_privileged_credentials(
                existing=self._privileged_credentials,
                error=exc,
            )
            if credentials is None:
                raise
            self._privileged_credentials = credentials
            self.controller.apply_database_settings(
                state,
                privileged_credentials=self._privileged_credentials,
            )
        return "Database connection saved."

    def _prompt_privileged_credentials(
        self,
        *,
        existing: Optional[Tuple[Optional[str], Optional[str]]],
        error: BootstrapError,
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        dialog = Gtk.Dialog(transient_for=self, modal=True)
        if hasattr(dialog, "set_title"):
            dialog.set_title("Privileged PostgreSQL credentials required")
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Apply", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        message = Gtk.Label(
            label=(
                "ATLAS needs a privileged PostgreSQL account to provision the conversation store.\n"
                "Provide credentials with sufficient access, or leave both fields blank to skip."
            )
        )
        message.set_wrap(True)
        message.set_xalign(0.0)
        content.append(message)

        error_label = Gtk.Label(label=str(error) or "BootstrapError")
        error_label.set_wrap(True)
        error_label.set_xalign(0.0)
        if hasattr(error_label, "add_css_class"):
            error_label.add_css_class("error-text")
        content.append(error_label)

        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        username_label = Gtk.Label(label="Username")
        username_label.set_xalign(0.0)
        grid.attach(username_label, 0, 0, 1, 1)
        username_entry = Gtk.Entry()
        username_entry.set_hexpand(True)
        username_entry.set_placeholder_text("Leave blank to skip")
        if existing and existing[0]:
            username_entry.set_text(existing[0])
        username_entry.set_activates_default(True)
        grid.attach(username_entry, 1, 0, 1, 1)

        password_label = Gtk.Label(label="Password")
        password_label.set_xalign(0.0)
        grid.attach(password_label, 0, 1, 1, 1)
        password_entry = Gtk.Entry()
        password_entry.set_visibility(False)
        password_entry.set_hexpand(True)
        password_entry.set_placeholder_text("Leave blank to skip")
        if existing and existing[1]:
            password_entry.set_text(existing[1])
        grid.attach(password_entry, 1, 1, 1, 1)

        content.append(grid)
        dialog.set_child(content)

        response: Dict[str, Any] = {
            "response": Gtk.ResponseType.CANCEL,
            "username": "",
            "password": "",
        }

        loop = GLib.MainLoop()

        def _on_response(dlg: Gtk.Dialog, resp: int) -> None:
            response["response"] = resp
            response["username"] = username_entry.get_text().strip()
            response["password"] = password_entry.get_text()
            dlg.destroy()
            if loop.is_running():
                loop.quit()

        dialog.connect("response", _on_response)
        dialog.present()
        loop.run()

        if response["response"] != Gtk.ResponseType.OK:
            return None

        username = response["username"] or None
        password = response["password"] or None
        if username is None:
            return None
        return (username, password)

    def _apply_providers(self) -> str:
        default_provider = self._provider_entries["default_provider"].get_text().strip() or None
        default_model = self._provider_entries["default_model"].get_text().strip() or None

        buffer = self._provider_buffer
        api_keys: Dict[str, str] = {}
        if buffer is not None:
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            text = buffer.get_text(start_iter, end_iter, True)
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError("API key entries must use provider=key format")
                provider, key = line.split("=", 1)
                provider = provider.strip()
                key = key.strip()
                if not provider or not key:
                    raise ValueError("API key entries must include both provider and key")
                api_keys[provider] = key

        state = ProviderState(
            default_provider=default_provider,
            default_model=default_model,
            api_keys=api_keys,
        )

        self.controller.apply_provider_settings(state)
        return "Provider settings saved."

    def _apply_user(self) -> str:
        username = self._user_entries["username"].get_text().strip()
        email = self._user_entries["email"].get_text().strip()
        display_name = self._user_entries["display_name"].get_text().strip()
        password = self._user_entries["password"].get_text()
        confirm = self._user_entries["confirm_password"].get_text()

        if not username:
            raise ValueError("Username is required")
        if not password:
            raise ValueError("Password is required")
        if password != confirm:
            raise ValueError("Passwords do not match")

        state = UserState(
            username=username,
            email=email,
            display_name=display_name,
            password=password,
        )

        self.controller.register_user(state)
        return "Administrator account created."


class SetupWizardController(CoreSetupWizardController):  # type: ignore[misc]
    """Backwards-compatible import shim for legacy callers."""

    pass
