"""Interactive GTK front-end for the ATLAS setup workflow."""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GLib", "2.0")
gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib, Gdk, Gtk

from ATLAS.setup import (
    AdminProfile,
    DatabaseState,
    JobSchedulingState,
    KvStoreState,
    MessageBusState,
    OptionalState,
    ProviderState,
    RetryPolicyState,
    SetupWizardController as CoreSetupWizardController,
    SpeechState,
    UserState,
)
from ATLAS.setup_marker import write_setup_marker
from GTKUI.Utils.logging import GTKUILogHandler
from GTKUI.Utils.styled_window import AtlasWindow
from modules.conversation_store.bootstrap import BootstrapError
from GTKUI.Setup.preflight import PreflightHelper, PreflightCheckResult

logger = logging.getLogger(__name__)

Callback = Callable[[], None]
ErrorCallback = Callable[[BaseException], None]


@dataclass
class WizardStep:
    name: str
    widget: Gtk.Widget
    apply: Callable[[], str]
    subpages: list[Gtk.Widget] | None = None

    def __post_init__(self) -> None:
        if self.subpages is None or not self.subpages:
            self.subpages = [self.widget]
        elif self.widget not in self.subpages:
            self.subpages.insert(0, self.widget)
        if self.subpages:
            self.widget = self.subpages[0]


@dataclass
class PreflightRowWidgets:
    status: Gtk.Label
    message: Gtk.Label
    button: Gtk.Button | None


class SetupWizardLogWindow(AtlasWindow):
    """Lightweight window that streams setup logs to an inspector view."""

    def __init__(
        self,
        *,
        application: Gtk.Application | None = None,
        transient_for: Gtk.Window | None = None,
    ) -> None:
        super().__init__(
            title="ATLAS Setup Logs",
            default_size=(720, 480),
            transient_for=transient_for,
        )

        if application is not None:
            try:
                self.set_application(application)
            except Exception:  # pragma: no cover - GTK stubs in tests
                pass

        header = Gtk.HeaderBar()
        try:
            header.set_show_title_buttons(True)
        except Exception:  # pragma: no cover - compatibility shim
            pass

        title_label = Gtk.Label(label="Setup Activity Log")
        if hasattr(title_label, "add_css_class"):
            title_label.add_css_class("heading")
        try:
            header.set_title_widget(title_label)
        except Exception:  # pragma: no cover - GTK3 fallback
            try:
                header.set_title("Setup Activity Log")
            except Exception:
                pass

        try:
            self.set_titlebar(header)
        except Exception:  # pragma: no cover - GTK3 fallback
            pass

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.set_hexpand(True)
        root.set_vexpand(True)
        self.set_child(root)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        root.append(scroller)

        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_cursor_visible(False)
        self.text_view.set_monospace(True)
        wrap_mode = getattr(Gtk.WrapMode, "WORD_CHAR", getattr(Gtk.WrapMode, "WORD", None))
        if wrap_mode is not None:
            self.text_view.set_wrap_mode(wrap_mode)
        scroller.set_child(self.text_view)

        self.text_buffer = self.text_view.get_buffer()


class SetupWizardWindow(AtlasWindow):
    """A small multi-step wizard bound to :class:`SetupWizardController`."""

    _ADDITIONAL_LOGGER_NAMES: tuple[str, ...] = (
        "ATLAS.setup",
        "ATLAS.setup.cli",
        "modules.conversation_store.bootstrap",
    )
    _DEBUG_ICON_FILENAME = "debug.png"
    _MIN_FORM_CONTENT_WIDTH = 420
    _MAX_FORM_CONTENT_WIDTH = 720

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
        preferred_width, preferred_height = 960, 720
        fallback_width, fallback_height = 800, 600
        margin = 64

        desired_width, desired_height = fallback_width, fallback_height

        display = Gdk.Display.get_default()
        if display is not None:
            monitors = display.get_monitors()
            if monitors is not None and monitors.get_n_items() > 0:
                monitor = monitors.get_item(0)
                if monitor is not None:
                    workarea = monitor.get_workarea()
                    available_width = max(320, workarea.width - margin)
                    available_height = max(320, workarea.height - margin)

                    desired_width = min(preferred_width, available_width)
                    desired_height = min(preferred_height, available_height)

                    if available_width >= fallback_width:
                        desired_width = max(desired_width, fallback_width)
                    if available_height >= fallback_height:
                        desired_height = max(desired_height, fallback_height)

        super().__init__(
            title="ATLAS Setup — Guided Configuration",
            default_size=(desired_width, desired_height),
        )

        header_bar = Gtk.HeaderBar()
        try:
            header_bar.set_show_title_buttons(True)
        except Exception:  # pragma: no cover - compatibility shim
            pass

        if hasattr(header_bar, "add_css_class"):
            try:
                header_bar.add_css_class("setup-wizard-header")
            except Exception:  # pragma: no cover - GTK stubs in tests
                pass

        debug_button = Gtk.Button()
        if hasattr(debug_button, "set_tooltip_text"):
            debug_button.set_tooltip_text("Show setup logs")
        if hasattr(debug_button, "add_css_class"):
            try:
                debug_button.add_css_class("setup-wizard-debug-button")
            except Exception:  # pragma: no cover - GTK stubs
                pass

        icon_widget = self._create_debug_icon_widget()
        if icon_widget is not None:
            if hasattr(debug_button, "set_child"):
                debug_button.set_child(icon_widget)
            elif hasattr(debug_button, "set_image"):
                debug_button.set_image(icon_widget)

        for pack in (getattr(header_bar, "pack_start", None), getattr(header_bar, "add", None)):
            if callable(pack):
                try:
                    pack(debug_button)
                    break
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        self._setup_actions = Gio.SimpleActionGroup()
        try:
            self.insert_action_group("setup", self._setup_actions)
        except Exception:  # pragma: no cover - GTK fallback
            pass

        export_action = Gio.SimpleAction.new("export_config", None)
        export_action.connect("activate", self._on_export_config_action)
        self._setup_actions.add_action(export_action)

        import_action = Gio.SimpleAction.new("import_config", None)
        import_action.connect("activate", self._on_import_config_action)
        self._setup_actions.add_action(import_action)

        config_menu = Gio.Menu()
        config_menu.append("Export config…", "setup.export_config")
        config_menu.append("Import config…", "setup.import_config")

        config_button = Gtk.MenuButton()
        if hasattr(config_button, "set_menu_model"):
            config_button.set_menu_model(config_menu)
        if hasattr(config_button, "set_icon_name"):
            try:
                config_button.set_icon_name("open-menu-symbolic")
            except Exception:  # pragma: no cover - GTK fallback
                pass
        if hasattr(config_button, "set_tooltip_text"):
            config_button.set_tooltip_text("Import or export configuration")
        if hasattr(config_button, "add_css_class"):
            try:
                config_button.add_css_class("flat")
            except Exception:  # pragma: no cover - GTK fallback
                pass
        self._config_menu_button = config_button

        for pack in (getattr(header_bar, "pack_end", None), getattr(header_bar, "add", None)):
            if callable(pack):
                try:
                    pack(config_button)
                    break
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        title_label = Gtk.Label(label="ATLAS Setup — Guided Configuration")
        if hasattr(title_label, "add_css_class"):
            try:
                title_label.add_css_class("heading")
            except Exception:  # pragma: no cover - GTK stubs
                pass
        try:
            header_bar.set_title_widget(title_label)
        except Exception:  # pragma: no cover - GTK3 fallback
            try:
                header_bar.set_title("ATLAS Setup — Guided Configuration")
            except Exception:
                pass

        try:
            self.set_titlebar(header_bar)
        except Exception:  # pragma: no cover - GTK3 fallback
            pass

        self.set_application(application)
        self._on_success = on_success
        self._on_error = on_error

        self.controller = controller or CoreSetupWizardController(
            atlas=atlas,
            request_privileged_password=self._request_sudo_password,
        )

        # NEW: sidebar tracking
        self._completed_steps: set[int] = set()
        self._step_rows: list[Gtk.ListBoxRow] = []
        self._step_list: Gtk.ListBox | None = None
        self._toast_overlay: Gtk.Widget | None = None
        self._toast_history: list[tuple[str, str]] = []
        self._last_config_directory: Path | None = None
        self._config_menu_button: Gtk.Widget | None = None

        root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root_box.set_margin_top(18)
        root_box.set_margin_bottom(18)
        root_box.set_margin_start(18)
        root_box.set_margin_end(18)
        root_box.set_vexpand(True)
        root_box.set_hexpand(True)

        overlay_cls = getattr(Gtk, "ToastOverlay", None)
        overlay: Gtk.Widget | None = None
        if overlay_cls is not None:
            try:
                overlay = overlay_cls()
            except Exception:  # pragma: no cover - GTK fallback
                overlay = None
        if overlay is not None:
            if hasattr(overlay, "set_child"):
                overlay.set_child(root_box)
            if hasattr(overlay, "set_hexpand"):
                overlay.set_hexpand(True)
            if hasattr(overlay, "set_vexpand"):
                overlay.set_vexpand(True)
            self._toast_overlay = overlay
            self.set_child(overlay)
        else:
            self.set_child(root_box)

        # H layout: [Steps Sidebar] [Form Container] [Guidance]
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        content.set_hexpand(True)
        content.set_vexpand(True)
        root_box.append(content)

        # --- Steps sidebar (left) ---
        steps_sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        steps_sidebar.set_hexpand(False)
        steps_sidebar.set_vexpand(True)
        steps_sidebar.set_size_request(220, -1)  # keep form “slid” to the right
        if hasattr(steps_sidebar, "add_css_class"):
            steps_sidebar.add_css_class("setup-wizard-sidebar")
        content.append(steps_sidebar)

        steps_title = Gtk.Label(label="Steps")
        steps_title.set_xalign(0.0)
        if hasattr(steps_title, "add_css_class"):
            steps_title.add_css_class("heading")
        steps_sidebar.append(steps_title)

        self._step_list = Gtk.ListBox()
        self._step_list.set_selection_mode(Gtk.SelectionMode.BROWSE)
        self._step_list.connect("row-activated", self._on_step_row_activated)
        steps_scroller = Gtk.ScrolledWindow()
        steps_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        steps_scroller.set_vexpand(True)
        steps_scroller.set_hexpand(True)
        steps_scroller.set_propagate_natural_height(False)
        steps_sidebar.append(steps_scroller)

        steps_scroller.set_child(self._step_list)

        # --- Center form container (middle) ---
        center_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        center_column.set_hexpand(True)
        center_column.set_vexpand(True)
        center_column.set_halign(Gtk.Align.FILL)
        if hasattr(center_column, "add_css_class"):
            center_column.add_css_class("setup-wizard-main")
        content.append(center_column)

        form_heading = Gtk.Label(label="Active Form")
        form_heading.set_xalign(0.0)
        if hasattr(form_heading, "add_css_class"):
            form_heading.add_css_class("heading")
        center_column.append(form_heading)

        form_scroller = Gtk.ScrolledWindow()
        form_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        form_scroller.set_vexpand(True)
        form_scroller.set_propagate_natural_height(False)
        form_scroller.set_propagate_natural_width(False)
        form_scroller.set_halign(Gtk.Align.FILL)
        form_scroller.set_hexpand(True)
        if hasattr(form_scroller, "set_min_content_width"):
            form_scroller.set_min_content_width(self._MIN_FORM_CONTENT_WIDTH)
        if hasattr(form_scroller, "set_max_content_width"):
            form_scroller.set_max_content_width(self._MAX_FORM_CONTENT_WIDTH)
        center_column.append(form_scroller)

        form_frame = Gtk.Frame()
        form_frame.set_vexpand(True)
        form_frame.set_halign(Gtk.Align.FILL)
        form_scroller.set_child(form_frame)

        self._form_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._form_column.set_vexpand(True)
        self._form_column.set_halign(Gtk.Align.FILL)
        form_frame.set_child(self._form_column)

        # --- Guidance column (right) ---
        guidance_scroller = Gtk.ScrolledWindow()
        guidance_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        guidance_scroller.set_vexpand(True)
        guidance_scroller.set_hexpand(True)
        guidance_scroller.set_propagate_natural_height(False)
        guidance_scroller.set_propagate_natural_width(False)
        content.append(guidance_scroller)

        guidance_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        guidance_column.set_hexpand(True)
        guidance_column.set_vexpand(True)
        if hasattr(guidance_column, "add_css_class"):
            guidance_column.add_css_class("setup-wizard-guidance")
        guidance_scroller.set_child(guidance_column)

        guidance_title = Gtk.Label(label="Instructions")
        guidance_title.set_xalign(0.0)
        if hasattr(guidance_title, "add_css_class"):
            guidance_title.add_css_class("heading")
        guidance_column.append(guidance_title)

        header = Gtk.Label(label="Here's the plan—work through each step to wrap up setup.")
        header.set_wrap(True)
        header.set_xalign(0.0)
        header.set_hexpand(True)
        guidance_column.append(header)

        self._status_label = Gtk.Label()
        self._status_label.set_wrap(True)
        self._status_label.set_xalign(0.0)
        self._status_label.set_hexpand(True)
        guidance_column.append(self._status_label)

        self._instructions_label = Gtk.Label()
        self._instructions_label.set_wrap(True)
        self._instructions_label.set_xalign(0.0)
        self._instructions_label.set_hexpand(True)
        self._instructions_label.set_valign(Gtk.Align.START)
        self._instructions_label.set_visible(False)
        guidance_column.append(self._instructions_label)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        if hasattr(self._stack, "set_hhomogeneous"):
            self._stack.set_hhomogeneous(True)
        if hasattr(self._stack, "set_vhomogeneous"):
            self._stack.set_vhomogeneous(True)
        self._stack.set_hexpand(True)
        self._stack.set_vexpand(True)
        self._form_column.append(self._stack)

        self._stack.connect("notify::visible-child", self._on_stack_visible_child_changed)

        # NOTE: removed the old single-line step indicator bar (sidebar replaces it)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        controls.set_halign(Gtk.Align.FILL)
        controls.set_hexpand(True)
        if hasattr(controls, "add_css_class"):
            controls.add_css_class("setup-wizard-controls")
        root_box.append(controls)

        left_controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left_controls.set_hexpand(True)
        controls.append(left_controls)

        self._step_status_label = Gtk.Label()
        self._step_status_label.set_xalign(0.0)
        self._step_status_label.set_hexpand(True)
        left_controls.append(self._step_status_label)

        self._step_progress_bar = Gtk.ProgressBar()
        self._step_progress_bar.set_hexpand(True)
        left_controls.append(self._step_progress_bar)

        self._back_button = Gtk.Button(label="Back")
        if hasattr(self._back_button, "add_css_class"):
            self._back_button.add_css_class("setup-wizard-nav")
        if hasattr(self._back_button, "set_tooltip_text"):
            self._back_button.set_tooltip_text("Go back (Alt+Left)")
        if hasattr(self._back_button, "set_can_default"):
            self._back_button.set_can_default(False)
        self._back_button.set_halign(Gtk.Align.START)
        left_controls.append(self._back_button)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        controls.append(spacer)

        navigation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        navigation_box.set_halign(Gtk.Align.END)
        navigation_box.set_hexpand(False)
        controls.append(navigation_box)

        self._progress_hint_label = Gtk.Label(label="Progress saved automatically.")
        self._progress_hint_label.set_xalign(1.0)
        self._progress_hint_label.set_halign(Gtk.Align.END)
        self._progress_hint_label.set_margin_bottom(6)
        self._progress_hint_label.set_hexpand(False)
        if hasattr(self._progress_hint_label, "add_css_class"):
            self._progress_hint_label.add_css_class("setup-wizard-progress-hint")
        navigation_box.append(self._progress_hint_label)

        self._next_button = Gtk.Button(label="Next")
        if hasattr(self._next_button, "add_css_class"):
            self._next_button.add_css_class("setup-wizard-nav")
            self._next_button.add_css_class("suggested-action")
        if hasattr(self._next_button, "set_tooltip_text"):
            self._next_button.set_tooltip_text("Next (Alt+Right)")
        if hasattr(self._next_button, "set_can_default"):
            self._next_button.set_can_default(True)
        self._next_button.set_halign(Gtk.Align.END)
        self._next_button.connect("clicked", self._on_next_clicked)
        navigation_box.append(self._next_button)

        self._steps: List[WizardStep] = []
        self._current_index = 0

        self._step_status_label.set_text("")
        self._step_progress_bar.set_fraction(0.0)

        self._instructions_by_widget: Dict[Gtk.Widget, str] = {}
        self._step_containers: List[Gtk.Widget] = []
        self._step_page_stacks: Dict[int, Gtk.Stack] = {}
        self._current_page_indices: Dict[int, int] = {}

        self._setup_type_buttons: Dict[str, Gtk.CheckButton] = {}
        self._setup_type_syncing = False
        self._database_entries: Dict[str, Gtk.Entry] = {}
        self._database_backend_combo: Gtk.ComboBoxText | None = None
        self._database_stack: Gtk.Stack | None = None
        self._provider_entries: Dict[str, Gtk.Entry] = {}
        self._provider_buffer: Optional[Gtk.TextBuffer] = None
        self._provider_mask_buffer: Optional[Gtk.TextBuffer] = None
        self._provider_stack: Gtk.Stack | None = None
        self._provider_show_toggle: Gtk.CheckButton | None = None
        self._user_entries: Dict[str, Gtk.Entry] = {}
        self._kv_widgets: Dict[str, Gtk.Widget] = {}
        self._job_widgets: Dict[str, Gtk.Widget] = {}
        self._message_widgets: Dict[str, Gtk.Widget] = {}
        self._speech_widgets: Dict[str, Gtk.Widget] = {}
        self._optional_widgets: Dict[str, Gtk.Widget] = {}
        self._optional_personal_hint: Gtk.Label | None = None
        self._setup_persisted = False
        self._privileged_credentials: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._entry_pixel_width: Optional[int] = None
        self._database_user_suggestion: str = ""
        self._tenant_id_suggestion: str = ""
        self._log_window: SetupWizardLogWindow | None = None
        self._log_handler: GTKUILogHandler | None = None
        self._log_target_loggers: list[logging.Logger] = []
        self._log_toggle_button: Gtk.Button | None = debug_button
        self._log_button_handler_id: int | None = None
        self._validation_rules: dict[Gtk.Widget, Callable[[], tuple[bool, str | None]]] = {}
        self._validation_signal_ids: dict[Gtk.Widget, int] = {}
        self._validation_base_tooltips: dict[Gtk.Widget, str | None] = {}
        self._validation_base_descriptions: dict[Gtk.Widget, str | None] = {}
        self._preflight_helper = PreflightHelper(
            request_password=self._request_sudo_password
        )
        self._preflight_button: Gtk.Button | None = None
        self._preflight_dialog: Gtk.Dialog | None = None
        self._preflight_rows: dict[str, PreflightRowWidgets] = {}
        self._preflight_results: dict[str, PreflightCheckResult] = {}

        if hasattr(debug_button, "connect"):
            try:
                self._log_button_handler_id = debug_button.connect(
                    "clicked", self._on_log_button_clicked
                )
            except Exception:  # pragma: no cover - GTK fallback
                self._log_button_handler_id = None

        key_controller_cls = getattr(Gtk, "EventControllerKey", None)
        if key_controller_cls is not None:
            try:
                key_controller = key_controller_cls()
            except Exception:  # pragma: no cover - GTK stubs
                key_controller = None
            if key_controller is not None:
                try:
                    key_controller.connect(
                        "key-pressed", self._on_window_key_pressed
                    )
                except Exception:  # pragma: no cover - GTK stubs
                    pass
                else:
                    add_controller = getattr(self, "add_controller", None)
                    if callable(add_controller):
                        try:
                            add_controller(key_controller)
                        except Exception:  # pragma: no cover - GTK3 fallback
                            pass

        self._build_steps()
        self._build_steps_sidebar()  # populate sidebar

        self._refresh_validation_states()

        if error is not None:
            self.display_error(error)
        else:
            self._set_status("Welcome! Let's configure ATLAS together.")
        self._go_to_step(0)

        if hasattr(self, "connect"):
            try:
                self.connect("close-request", self._on_wizard_close_request)
            except (AttributeError, TypeError):
                try:
                    self.connect("destroy", self._on_wizard_destroy)
                except Exception:  # pragma: no cover - GTK stubs
                    pass

    def _request_sudo_password(self) -> str | None:
        privileged_state = self.controller.state.user.privileged_credentials
        stored = (privileged_state.sudo_password or "").strip()
        if stored:
            return stored

        dialog = Gtk.Dialog(transient_for=self, modal=True)
        if hasattr(dialog, "set_title"):
            dialog.set_title("Administrator privileges required")
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Submit", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        message = Gtk.Label(
            label=(
                "ATLAS needs the administrator's sudo password to perform privileged "
                "operations. Enter the password to continue."
            )
        )
        message.set_wrap(True)
        message.set_xalign(0.0)
        content.append(message)

        password_entry = Gtk.Entry()
        password_entry.set_visibility(False)
        password_entry.set_hexpand(True)
        password_entry.set_activates_default(True)
        content.append(password_entry)

        dialog.set_child(content)

        response: Dict[str, Any] = {
            "response": Gtk.ResponseType.CANCEL,
            "password": "",
        }

        loop = GLib.MainLoop()

        def _on_response(dlg: Gtk.Dialog, resp: int) -> None:
            response["response"] = resp
            response["password"] = password_entry.get_text()
            dlg.destroy()
            if loop.is_running():
                loop.quit()

        dialog.connect("response", _on_response)
        dialog.present()
        loop.run()

        if response["response"] != Gtk.ResponseType.OK:
            return None

        password = response["password"].strip()
        if not password:
            return None

        updated_credentials = replace(privileged_state, sudo_password=password)
        self.controller.state.user = replace(
            self.controller.state.user,
            privileged_credentials=updated_credentials,
        )
        return password

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

    def show_success_toast(self, message: str) -> None:
        """Display a transient success toast, falling back to status updates."""

        self._show_toast(message, kind="success")

    def show_error_toast(self, message: str) -> None:
        """Display a transient error toast, falling back to status updates."""

        self._show_toast(message, kind="error")

    def _show_toast(self, message: str, *, kind: str = "info") -> None:
        if not message:
            return
        self._toast_history.append((kind, message))

        overlay = self._toast_overlay
        toast_cls = getattr(Gtk, "Toast", None)
        if overlay is not None and toast_cls is not None:
            try:
                toast = toast_cls.new(message) if hasattr(toast_cls, "new") else toast_cls()
                if hasattr(toast, "set_title"):
                    toast.set_title(message)
                elif hasattr(toast, "set_text"):
                    toast.set_text(message)
                if kind == "error":
                    priority = getattr(Gtk.ToastPriority, "HIGH", None)
                    setter = getattr(toast, "set_priority", None)
                    if callable(setter) and priority is not None:
                        setter(priority)
                adder = getattr(overlay, "add_toast", None)
                if callable(adder):
                    adder(toast)
                    return
            except Exception:  # pragma: no cover - toast best-effort
                logger.debug("Failed to present toast", exc_info=True)

        # Fall back to status label when toasts are unavailable.
        if kind == "error":
            self.display_error(RuntimeError(message))
        else:
            self._set_status(message)

    def _choose_config_path(
        self,
        *,
        title: str,
        action: str,
        suggested_name: str | None = None,
    ) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, action.upper(), None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            return None

        chooser = chooser_cls(title=title, transient_for=self, modal=True, action=action_enum)
        if suggested_name and hasattr(chooser, "set_current_name"):
            try:
                chooser.set_current_name(suggested_name)
            except Exception:
                pass

        if self._last_config_directory and hasattr(chooser, "set_current_folder"):
            try:
                chooser.set_current_folder(str(self._last_config_directory))
            except Exception:
                pass

        file_filter_cls = getattr(Gtk, "FileFilter", None)
        if callable(file_filter_cls):
            try:
                yaml_filter = file_filter_cls()
                if hasattr(yaml_filter, "set_name"):
                    yaml_filter.set_name("YAML files")
                if hasattr(yaml_filter, "add_pattern"):
                    yaml_filter.add_pattern("*.yaml")
                    yaml_filter.add_pattern("*.yml")
                adder = getattr(chooser, "add_filter", None)
                if callable(adder):
                    adder(yaml_filter)
            except Exception:  # pragma: no cover - GTK compatibility fallback
                pass

        response = None
        if hasattr(chooser, "run"):
            try:
                response = chooser.run()
            except Exception:
                response = None
        elif hasattr(chooser, "show"):
            try:
                chooser.show()
                response = getattr(Gtk.ResponseType, "ACCEPT", 1)
            except Exception:
                response = None

        accepted = {
            getattr(Gtk.ResponseType, "ACCEPT", None),
            getattr(Gtk.ResponseType, "OK", None),
            getattr(Gtk.ResponseType, "YES", None),
        }

        filename: Optional[str] = None
        if response in accepted:
            file_obj = getattr(chooser, "get_file", None)
            file_handle = file_obj() if callable(file_obj) else None
            if file_handle is not None and hasattr(file_handle, "get_path"):
                filename = file_handle.get_path()
            else:
                getter = getattr(chooser, "get_filename", None)
                if callable(getter):
                    filename = getter()

        if hasattr(chooser, "destroy"):
            try:
                chooser.destroy()
            except Exception:
                pass

        if not filename:
            return None

        path_obj = Path(filename).expanduser().resolve()
        self._last_config_directory = path_obj.parent
        return str(path_obj)

    def _choose_export_path(self) -> Optional[str]:
        return self._choose_config_path(
            title="Export configuration",
            action="SAVE",
            suggested_name="atlas-config.yaml",
        )

    def _choose_import_path(self) -> Optional[str]:
        return self._choose_config_path(
            title="Import configuration",
            action="OPEN",
        )

    def _on_export_config_action(self, *_: object) -> None:
        path = self._choose_export_path()
        if not path:
            return
        try:
            exported = self.controller.export_config(path)
        except Exception as exc:
            logger.error("Failed to export configuration", exc_info=True)
            self.display_error(exc)
            self.show_error_toast("Failed to export configuration.")
            return

        self._set_status(f"Configuration exported to {exported}.")
        self.show_success_toast(f"Exported configuration to {Path(exported).name}")

    def _on_import_config_action(self, *_: object) -> None:
        path = self._choose_import_path()
        if not path:
            return

        try:
            imported = self.controller.import_config(path)
        except ValueError as exc:
            logger.warning("Invalid configuration import: %s", exc)
            self.display_error(exc)
            self.show_error_toast(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to import configuration", exc_info=True)
            self.display_error(exc)
            self.show_error_toast("Failed to import configuration.")
            return

        self._rebuild_steps_after_config_change()
        self._set_status(f"Configuration imported from {imported}.")
        self.show_success_toast(f"Imported configuration from {Path(imported).name}")

    def _rebuild_steps_after_config_change(self) -> None:
        """Recreate wizard pages so widgets reflect the controller state."""

        self._instructions_by_widget.clear()
        self._completed_steps.clear()
        self._build_steps()
        self._build_steps_sidebar()
        self._refresh_validation_states()
        self._go_to_step(0)

    def _build_steps(self) -> None:
        self._instructions_by_widget.clear()
        try:
            children = list(self._stack.get_children())
        except AttributeError:  # pragma: no cover - GTK3 fallback
            children = []
        for child in children:
            try:
                self._stack.remove(child)
            except Exception:  # pragma: no cover - defensive
                pass

        self._step_containers = []
        self._step_page_stacks.clear()
        self._current_page_indices.clear()
        self._preflight_button = None
        self._preflight_rows.clear()
        self._preflight_results.clear()

        self._setup_type_buttons.clear()
        provider_pages = self._build_provider_pages()

        overview_page = self._build_overview_page()
        setup_type_page = self._build_setup_type_page()
        administrator_intro = self._build_administrator_intro_page()
        administrator_form = self._build_user_page()
        database_intro = self._build_database_intro_page()
        database_form = self._build_database_page()
        job_scheduling_intro = self._build_job_scheduling_intro_page()
        job_scheduling_form = self._build_job_scheduling_page()
        message_bus_intro = self._build_message_bus_intro_page()
        message_bus_form = self._build_message_bus_page()
        kv_intro = self._build_kv_store_intro_page()
        kv_form = self._build_kv_store_page()
        speech_intro = self._build_speech_intro_page()
        speech_form = self._build_speech_page()
        optional_intro = self._build_optional_intro_page()
        optional_form = self._build_optional_page()

        provider_intro = self._build_provider_intro_page()
        provider_pages = [provider_intro, *provider_pages]

        self._steps = [
            WizardStep(
                name="Introduction",
                widget=overview_page,
                subpages=[overview_page],
                apply=lambda: "Review complete.",
            ),
            WizardStep(
                name="Setup Type",
                widget=setup_type_page,
                subpages=[setup_type_page],
                apply=self._apply_setup_type,
            ),
            WizardStep(
                name="Administrator",
                widget=administrator_intro,
                subpages=[administrator_intro, administrator_form],
                apply=self._apply_user,
            ),
            WizardStep(
                name="Database",
                widget=database_intro,
                subpages=[database_intro, database_form],
                apply=self._apply_database,
            ),
            WizardStep(
                name="Job Scheduling",
                widget=job_scheduling_intro,
                subpages=[job_scheduling_intro, job_scheduling_form],
                apply=self._apply_job_scheduling,
            ),
            WizardStep(
                name="Message Bus",
                widget=message_bus_intro,
                subpages=[message_bus_intro, message_bus_form],
                apply=self._apply_message_bus,
            ),
            WizardStep(
                name="Key-Value Store",
                widget=kv_intro,
                subpages=[kv_intro, kv_form],
                apply=self._apply_kv_store,
            ),
            WizardStep(
                name="Providers",
                widget=provider_pages[0],
                subpages=provider_pages,
                apply=self._apply_providers,
            ),
            WizardStep(
                name="Speech",
                widget=speech_intro,
                subpages=[speech_intro, speech_form],
                apply=self._apply_speech,
            ),
            WizardStep(
                name="Organization",
                widget=optional_intro,
                subpages=[optional_intro, optional_form],
                apply=self._apply_optional,
            ),
        ]

        for index, step in enumerate(self._steps):
            container, inner_stack = self._create_step_container(index, step)
            name = f"step-{index}"
            self._step_containers.append(container)
            self._current_page_indices[index] = 0
            self._stack.add_titled(container, name, step.name)
            if inner_stack is not None:
                self._step_page_stacks[index] = inner_stack

        self._update_setup_type_dependent_widgets()

    def _create_step_container(
        self, index: int, step: WizardStep
    ) -> tuple[Gtk.Widget, Gtk.Stack | None]:
        """Wrap each step in a full-width, full-height container.

        This ensures all steps presented in the main stack share the same
        layout envelope, regardless of whether they have one page or multiple
        subpages.
        """
        pages = step.subpages or [step.widget]

        # Outer container always expands to fill the form column
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        outer.set_hexpand(True)
        outer.set_vexpand(True)

        # Single-page step: just add the page directly
        if len(pages) <= 1:
            page = pages[0]
            if hasattr(page, "set_hexpand"):
                page.set_hexpand(True)
            if hasattr(page, "set_vexpand"):
                page.set_vexpand(True)
            outer.append(page)
            return outer, None

        # Multi-page step: keep the existing inner stack behaviour
        inner_stack = Gtk.Stack()
        inner_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        if hasattr(inner_stack, "set_hhomogeneous"):
            inner_stack.set_hhomogeneous(True)
        if hasattr(inner_stack, "set_vhomogeneous"):
            inner_stack.set_vhomogeneous(True)
        inner_stack.set_hexpand(True)
        inner_stack.set_vexpand(True)

        for page_index, page in enumerate(pages):
            if page_index == 0:
                page_title = f"{step.name} — Intro"
            else:
                page_title = f"{step.name} ({page_index + 1})"

            inner_stack.add_titled(
                page,
                f"step-{index}-page-{page_index}",
                page_title,
            )

        inner_stack.set_visible_child(pages[0])
        inner_stack.connect(
            "notify::visible-child",
            self._on_inner_stack_visible_child_changed,
            index,
        )

        outer.append(inner_stack)
        return outer, inner_stack

    def _on_inner_stack_visible_child_changed(
        self, stack: Gtk.Stack, _param_spec: object, step_index: int
    ) -> None:
        step_pages = self._get_step_pages(step_index)
        visible = stack.get_visible_child()
        if visible is None or not step_pages:
            return
        try:
            page_index = step_pages.index(visible)
        except ValueError:
            return
        self._current_page_indices[step_index] = page_index
        if step_index == self._current_index:
            self._update_guidance_for_widget(visible)
            self._update_navigation()
            self._update_step_status()

    def _get_step_pages(self, index: int) -> list[Gtk.Widget]:
        if not (0 <= index < len(self._steps)):
            return []
        pages = self._steps[index].subpages or []
        return list(pages)

    def _format_page_status(self, step: WizardStep, page_index: int, total: int) -> str:
        prefix = ""
        if total > 1:
            if page_index == 0:
                prefix = "Intro — "
            elif page_index == 1 and total == 2:
                prefix = "Configuration — "
        return f"{step.name}: {prefix}Page {page_index + 1} of {total}"

    # -- sidebar -----------------------------------------------------------

    def _build_steps_sidebar(self) -> None:
        """(Re)build the left sidebar list of steps with completion marks."""
        if not self._step_list:
            return

        for row in list(self._step_rows):
            self._step_list.remove(row)
        self._step_rows.clear()

        for idx, step in enumerate(self._steps):
            row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row_box.set_margin_top(6)
            row_box.set_margin_bottom(6)
            row_box.set_margin_start(8)
            row_box.set_margin_end(8)

            mark = Gtk.Label()
            mark.set_text("✓" if idx in self._completed_steps else "•")
            mark.set_width_chars(2)
            row_box.append(mark)

            title = Gtk.Label(label=step.name)
            title.set_xalign(0.0)
            title.set_hexpand(True)
            row_box.append(title)

            row = Gtk.ListBoxRow()
            row.set_activatable(True)
            row.set_child(row_box)
            self._step_list.append(row)
            self._step_rows.append(row)

        self._select_step_row(self._current_index)

    def _register_instructions(self, widget: Gtk.Widget, instructions: str) -> None:
        self._instructions_by_widget[widget] = instructions

    def _wrap_with_instructions(
        self, form: Gtk.Widget, instructions: str, heading: str | None = None
    ) -> Gtk.Widget:
        if heading is None:
            # Let the form fill the available column width instead of
            # hugging the left at minimum size.
            if hasattr(form, "set_halign"):
                form.set_halign(Gtk.Align.FILL)
            if hasattr(form, "set_hexpand"):
                form.set_hexpand(True)
            self._register_instructions(form, instructions)
            return form

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        container.set_halign(Gtk.Align.FILL)
        container.set_hexpand(True)

        heading_label = Gtk.Label(label=heading)
        heading_label.set_wrap(True)
        heading_label.set_xalign(0.0)
        if hasattr(heading_label, "add_css_class"):
            heading_label.add_css_class("heading")
        container.append(heading_label)

        if hasattr(form, "set_halign"):
            form.set_halign(Gtk.Align.FILL)
        if hasattr(form, "set_hexpand"):
            form.set_hexpand(True)
        container.append(form)

        self._register_instructions(container, instructions)
        return container

    def _update_guidance_for_widget(self, widget: Gtk.Widget | None) -> None:
        if widget is None:
            self._instructions_label.set_text("")
            self._instructions_label.set_visible(False)
            return

        instructions = self._instructions_by_widget.get(widget, "")
        self._instructions_label.set_text(instructions)
        self._instructions_label.set_visible(bool(instructions))

    def _build_overview_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)

        heading = Gtk.Label(label="Welcome! Let's get ATLAS ready")
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        box.append(heading)

        summary = Gtk.Label(
            label=(
                "This short walkthrough gathers the essentials so your deployment starts "
                "with sensible defaults. We'll pause along the way to explain what each "
                "choice does."
            )
        )
        summary.set_wrap(True)
        summary.set_xalign(0.0)
        box.append(summary)

        reassurance = Gtk.Label(
            label=(
                "Your answers save automatically—come back to any step from the sidebar "
                "whenever you need to tweak something or resume later."
            )
        )
        reassurance.set_wrap(True)
        reassurance.set_xalign(0.0)
        box.append(reassurance)

        why_callout = self._create_overview_callout(
            "Why this matters",
            [
                "Give ATLAS an owner who can finish setup and invite others.",
                "Connect the services that keep conversations safe and responsive.",
                "Set expectations now so future teammates know what was chosen.",
            ],
        )
        box.append(why_callout)

        needs_callout = self._create_overview_callout(
            "What you'll need",
            [
                "Contact details for the first administrator.",
                "Connection info for your conversation store (PostgreSQL, SQLite, or MongoDB/Atlas) and supporting services.",
                "API keys or credentials for any model providers you plan to use.",
            ],
        )
        box.append(needs_callout)

        preflight_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        preflight_box.set_hexpand(False)
        preflight_label = Gtk.Label(
            label=(
                "Want to double-check your environment first? Run the preflight scan to test"
                " your database backend, Redis, and the project virtualenv."
            )
        )
        preflight_label.set_wrap(True)
        preflight_label.set_xalign(0.0)
        preflight_box.append(preflight_label)

        preflight_button = Gtk.Button(label="Run preflight checks")
        preflight_button.set_halign(Gtk.Align.START)
        preflight_button.connect("clicked", self._on_preflight_button_clicked)
        preflight_box.append(preflight_button)
        box.append(preflight_box)
        self._preflight_button = preflight_button

        cli_label = Gtk.Label(
            label=(
                "Prefer a terminal instead? Run scripts/setup_atlas.py to pick up the "
                "same guided flow from the command line."
            )
        )
        cli_label.set_wrap(True)
        cli_label.set_xalign(0.0)
        box.append(cli_label)

        self._register_instructions(
            box,
            (
                "Glance through the overview and check the two callouts so you know what "
                "we'll ask for before you continue to the administrator details."
            ),
        )
        self._register_instructions(
            why_callout,
            "Use this to align the setup goals with anyone joining you for the rollout.",
        )
        self._register_instructions(
            needs_callout,
            "Gather these items now so the next few forms go quickly.",
        )
        self._register_instructions(
            preflight_label,
            "Use the scan if you're unsure whether supporting services are available before starting the forms.",
        )
        self._register_instructions(
            preflight_button,
            "Run the automated checks any time. You can trigger fixes directly from the results dialog.",
        )
        self._register_instructions(
            cli_label,
            "You can swap to the terminal helper at any point—the wizard keeps your progress in sync.",
        )
        return box

    def _on_preflight_button_clicked(self, button: Gtk.Button) -> None:
        self._start_preflight_checks(button)

    def _start_preflight_checks(self, button: Gtk.Button | None) -> None:
        if button is not None:
            button.set_sensitive(False)
        if self._preflight_dialog is not None:
            try:
                self._preflight_dialog.destroy()
            except Exception:
                pass
            self._preflight_dialog = None
        self._preflight_rows.clear()
        self._preflight_results.clear()
        self._set_status("Running preflight checks…")
        database_state = self._collect_database_state(strict=False)
        self._preflight_helper.configure_database_target(database_state)
        try:
            self._preflight_helper.run_checks(
                on_update=self._on_preflight_update,
                on_complete=self._on_preflight_complete,
            )
        except RuntimeError as exc:
            self._set_status(str(exc))
            if button is not None:
                button.set_sensitive(True)

    def _on_preflight_update(self, result: PreflightCheckResult) -> None:
        self._preflight_results[result.identifier] = result
        summary = (
            f"{result.label} looks good."
            if result.passed
            else f"{result.label} needs attention."
        )
        self._set_status(summary)

    def _on_preflight_complete(self, results: list[PreflightCheckResult]) -> None:
        if self._preflight_button is not None:
            self._preflight_button.set_sensitive(True)
        if not results:
            self._set_status("Preflight checks completed.")
            return

        pending = [result for result in results if not result.passed]
        if pending:
            self._set_status("Preflight checks found items to review.")
        else:
            self._set_status("All preflight checks passed.")
        self._present_preflight_dialog(results)

    def _present_preflight_dialog(self, results: list[PreflightCheckResult]) -> None:
        if self._preflight_dialog is not None:
            try:
                self._preflight_dialog.destroy()
            except Exception:
                pass
        dialog = Gtk.Dialog(transient_for=self, modal=True)
        if hasattr(dialog, "set_title"):
            dialog.set_title("System preflight results")
        dialog.add_button("Close", Gtk.ResponseType.CLOSE)
        dialog.set_default_size(560, 420)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        dialog.set_child(content)

        intro = Gtk.Label(
            label=(
                "Here's what we found. Use the Fix buttons to attempt an automated remedy,"
                " or follow the guidance in the details."
            )
        )
        intro.set_wrap(True)
        intro.set_xalign(0.0)
        content.append(intro)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        content.append(scroller)

        rows_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        rows_box.set_margin_top(6)
        rows_box.set_margin_bottom(6)
        rows_box.set_margin_start(6)
        rows_box.set_margin_end(6)
        scroller.set_child(rows_box)

        self._preflight_rows.clear()
        for result in results:
            row = self._create_preflight_row(result)
            rows_box.append(row)

        dialog.connect("response", self._on_preflight_dialog_response)
        dialog.present()
        self._preflight_dialog = dialog

    def _on_preflight_dialog_response(self, dialog: Gtk.Dialog, _response: int) -> None:
        dialog.destroy()
        if self._preflight_dialog is dialog:
            self._preflight_dialog = None

    def _create_preflight_row(self, result: PreflightCheckResult) -> Gtk.Widget:
        frame = Gtk.Frame()
        frame.set_hexpand(True)
        frame.set_vexpand(False)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)
        frame.set_child(box)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_hexpand(True)
        box.append(header)

        name_label = Gtk.Label(label=result.label)
        name_label.set_xalign(0.0)
        name_label.set_hexpand(True)
        header.append(name_label)

        status_label = Gtk.Label()
        status_label.set_xalign(0.0)
        status_label.set_hexpand(True)
        header.append(status_label)

        fix_button: Gtk.Button | None = None
        if result.fix_label:
            fix_button = Gtk.Button(label=result.fix_label)
            fix_button.set_sensitive(not result.passed)
            fix_button.connect("clicked", self._on_preflight_fix_clicked, result.identifier)
            header.append(fix_button)

        message_label = Gtk.Label()
        message_label.set_wrap(True)
        message_label.set_xalign(0.0)
        box.append(message_label)

        self._preflight_rows[result.identifier] = PreflightRowWidgets(
            status=status_label,
            message=message_label,
            button=fix_button,
        )
        self._update_preflight_row(result)
        return frame

    def _update_preflight_row(self, result: PreflightCheckResult) -> None:
        widgets = self._preflight_rows.get(result.identifier)
        if widgets is None:
            return

        status_text = "Ready" if result.passed else "Needs attention"
        widgets.status.set_text(status_text)
        if hasattr(widgets.status, "remove_css_class"):
            try:
                widgets.status.remove_css_class("error-text")
                widgets.status.remove_css_class("success-text")
            except Exception:
                pass
            target = "success-text" if result.passed else "error-text"
            try:
                widgets.status.add_css_class(target)
            except Exception:
                pass

        widgets.message.set_text(result.message)
        if widgets.button is not None:
            widgets.button.set_sensitive(not result.passed)

    def _on_preflight_fix_clicked(self, button: Gtk.Button, identifier: str) -> None:
        widgets = self._preflight_rows.get(identifier)
        if widgets is None:
            return

        widgets.message.set_text("Attempting fix…")
        button.set_sensitive(False)
        result = self._preflight_results.get(identifier)
        if result is not None:
            self._set_status(f"Applying fix for {result.label}…")

        def _handle_result(updated: PreflightCheckResult) -> None:
            self._preflight_results[identifier] = updated
            self._update_preflight_row(updated)
            if updated.passed:
                self._set_status(f"{updated.label} is ready after the fix.")
            else:
                self._set_status(f"{updated.label} still needs attention.")
                if widgets.button is not None:
                    widgets.button.set_sensitive(True)

        try:
            self._preflight_helper.run_fix(identifier, _handle_result)
        except RuntimeError as exc:
            widgets.message.set_text(str(exc))
            button.set_sensitive(True)

    def _build_setup_type_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)

        heading = Gtk.Label(label="Choose how presets should shape the rest of setup")
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        box.append(heading)

        copy = Gtk.Label(
            label=(
                "Presets apply once to pre-fill the remaining forms. After you tweak a field manually,"
                " picking the same preset again leaves your edits alone."
            )
        )
        copy.set_wrap(True)
        copy.set_xalign(0.0)
        box.append(copy)

        button_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        button_column.set_hexpand(False)

        radio_descriptors = [
            (
                "personal",
                "Personal",
                "Single administrator, defaults favor convenience on one host.",
            ),
            (
                "enterprise",
                "Enterprise",
                "Team rollout with Redis, schedulers, and stricter retention defaults.",
            ),
        ]

        first_button: Gtk.CheckButton | None = None
        for key, title, subtitle in radio_descriptors:
            row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            toggle = Gtk.CheckButton.new_with_label(title)
            if first_button is None:
                first_button = toggle
            else:
                toggle.set_group(first_button)
            toggle.set_hexpand(False)
            toggle.connect("toggled", self._on_setup_type_toggled, key)
            self._setup_type_buttons[key] = toggle
            row.append(toggle)

            if subtitle:
                detail = Gtk.Label(label=subtitle)
                detail.set_wrap(True)
                detail.set_xalign(0.0)
                if hasattr(detail, "add_css_class"):
                    detail.add_css_class("dim-label")
                row.append(detail)

            button_column.append(row)

        box.append(button_column)

        table = Gtk.Grid(column_spacing=12, row_spacing=6)
        table.set_hexpand(True)
        table.set_vexpand(False)

        headers = ["Area", "Personal", "Enterprise"]
        for col, title in enumerate(headers):
            label = Gtk.Label(label=title)
            label.set_wrap(True)
            label.set_xalign(0.0)
            if hasattr(label, "add_css_class"):
                label.add_css_class("heading")
            table.attach(label, col, 0, 1, 1)

        rows = [
            (
                "Message bus",
                "In-memory queue for a single app server.",
                "Redis backend with shared streams.",
            ),
            (
                "Job scheduling",
                "Disabled so nothing else is required.",
                "Enabled with a dedicated database-backed job store.",
            ),
            (
                "Key-value store",
                "Reuse the conversation database.",
                "Separate database-backed cache for scale.",
            ),
            (
                "Retention & policies",
                "No retention limits are pre-set.",
                "30 day retention and 500 message history.",
            ),
            (
                "HTTP server",
                "Auto-start for easy local testing.",
                "Manual start so ops can control ingress.",
            ),
        ]

        for row_index, (area, personal, enterprise) in enumerate(rows, start=1):
            for column, text in enumerate((area, personal, enterprise)):
                label = Gtk.Label(label=text)
                label.set_wrap(True)
                label.set_xalign(0.0)
                table.attach(label, column, row_index, 1, 1)

        box.append(table)

        instructions = (
            "Pick the preset that matches your rollout. You can always override fields later or"
            " switch presets if plans change."
        )

        self._register_instructions(box, instructions)
        self._register_instructions(
            table,
            "Scan this comparison so you know which downstream defaults will update when you choose a preset.",
        )

        self._sync_setup_type_selection()

        return box

    def _sync_setup_type_selection(self) -> None:
        if not self._setup_type_buttons:
            return
        state = getattr(self.controller.state, "setup_type", None)
        mode = (state.mode if state else "") or ""
        normalized = mode.strip().lower()
        if normalized not in {"personal", "enterprise"}:
            normalized = "personal"

        self._setup_type_syncing = True
        try:
            for key, button in self._setup_type_buttons.items():
                if isinstance(button, Gtk.CheckButton):
                    button.set_active(key == normalized)
        finally:
            self._setup_type_syncing = False

    def _get_selected_setup_type(self) -> str | None:
        for key, button in self._setup_type_buttons.items():
            if isinstance(button, Gtk.CheckButton) and button.get_active():
                return key
        return None

    def _on_setup_type_toggled(
        self, button: Gtk.CheckButton, mode: str
    ) -> None:
        if self._setup_type_syncing:
            return
        if not button.get_active():
            return
        self._apply_setup_type_selection(mode, update_status=True)

    def _apply_setup_type_selection(
        self, mode: str | None, *, update_status: bool
    ) -> tuple[str, bool]:
        selected = (mode or self._get_selected_setup_type() or "").strip().lower()
        if selected not in {"personal", "enterprise"}:
            message = "No preset selected. Configure each service manually."
            if update_status:
                self._set_status(message)
            return message, False

        current_state = getattr(self.controller.state, "setup_type", None)
        before_mode = current_state.mode if current_state else ""
        before_applied = current_state.applied if current_state else False

        new_state = self.controller.apply_setup_type(selected)
        after_mode = new_state.mode
        after_applied = new_state.applied
        changed = (before_mode != after_mode) or (before_applied != after_applied)

        if changed:
            self._refresh_setup_type_defaults()

        self._sync_setup_type_selection()
        self._update_setup_type_dependent_widgets()

        if after_applied:
            if changed:
                message = f"{after_mode.title()} preset applied."
            else:
                message = f"{after_mode.title()} preset already applied."
        else:
            message = f"{after_mode.title()} preset saved."

        if update_status:
            self._set_status(message)

        return message, changed

    def _refresh_setup_type_defaults(self) -> None:
        self._sync_job_widgets_from_state()
        self._sync_message_widgets_from_state()
        self._sync_kv_widgets_from_state()
        self._sync_optional_widgets_from_state()

    def _sync_job_widgets_from_state(self) -> None:
        state = self.controller.state.job_scheduling
        toggle = self._job_widgets.get("enabled")
        if isinstance(toggle, Gtk.CheckButton):
            toggle.set_active(state.enabled)

        mappings = {
            "job_store_url": state.job_store_url or "",
            "max_workers": self._optional_to_text(state.max_workers),
            "timezone": state.timezone or "",
            "queue_size": self._optional_to_text(state.queue_size),
            "retry_max_attempts": str(state.retry_policy.max_attempts),
            "retry_backoff_seconds": self._format_float(state.retry_policy.backoff_seconds),
            "retry_jitter_seconds": self._format_float(state.retry_policy.jitter_seconds),
            "retry_backoff_multiplier": self._format_float(
                state.retry_policy.backoff_multiplier
            ),
        }

        for key, value in mappings.items():
            widget = self._job_widgets.get(key)
            if isinstance(widget, Gtk.Entry):
                widget.set_text(value)

    def _sync_message_widgets_from_state(self) -> None:
        state = self.controller.state.message_bus
        backend_widget = self._message_widgets.get("backend")
        active_id = state.backend or "in_memory"
        set_active = getattr(backend_widget, "set_active_id", None)
        if callable(set_active):
            set_active(active_id)

        text_mappings = {
            "redis_url": state.redis_url or "",
            "stream_prefix": state.stream_prefix or "",
        }
        for key, value in text_mappings.items():
            widget = self._message_widgets.get(key)
            if isinstance(widget, Gtk.Entry):
                widget.set_text(value)

    def _sync_kv_widgets_from_state(self) -> None:
        state = self.controller.state.kv_store
        reuse_widget = self._kv_widgets.get("reuse")
        if isinstance(reuse_widget, Gtk.CheckButton):
            reuse_widget.set_active(state.reuse_conversation_store)
        url_widget = self._kv_widgets.get("url")
        if isinstance(url_widget, Gtk.Entry):
            url_widget.set_text(state.url or "")

    def _sync_optional_widgets_from_state(self) -> None:
        state = self.controller.state.optional
        mapping = {
            "tenant_id": state.tenant_id or "",
            "retention_days": self._optional_to_text(state.retention_days),
            "retention_history_limit": self._optional_to_text(
                state.retention_history_limit
            ),
            "scheduler_timezone": state.scheduler_timezone or "",
            "scheduler_queue_size": self._optional_to_text(state.scheduler_queue_size),
        }

        for key, value in mapping.items():
            widget = self._optional_widgets.get(key)
            if isinstance(widget, Gtk.Entry):
                widget.set_text(value)
                if key == "tenant_id" and value:
                    self._tenant_id_suggestion = value

        http_toggle = self._optional_widgets.get("http_auto_start")
        if isinstance(http_toggle, Gtk.CheckButton):
            http_toggle.set_active(state.http_auto_start)

        self._update_setup_type_dependent_widgets()

    def _update_setup_type_dependent_widgets(self) -> None:
        hint = self._optional_personal_hint
        if not isinstance(hint, Gtk.Label):
            return
        mode = getattr(self.controller.state.setup_type, "mode", "")
        hint.set_visible((mode or "").strip().lower() == "personal")

    def _create_overview_callout(self, title: str, bullets: list[str]) -> Gtk.Widget:
        frame = Gtk.Frame()
        frame.set_hexpand(True)
        frame.set_margin_top(6)
        frame.set_margin_bottom(6)
        frame.set_margin_start(6)
        frame.set_margin_end(6)

        if hasattr(frame, "set_label"):
            frame.set_label(title)
        if hasattr(frame, "set_label_align"):
            try:
                frame.set_label_align(0.0, 0.5)
            except Exception:  # pragma: no cover - GTK3 fallback
                pass

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(18)
        box.set_margin_end(18)

        for bullet in bullets:
            label = Gtk.Label(label=f"• {bullet}")
            label.set_wrap(True)
            label.set_xalign(0.0)
            box.append(label)

        if hasattr(frame, "set_child"):
            frame.set_child(box)
        else:  # pragma: no cover - GTK3 fallback
            frame.add(box)

        return frame

    def _create_intro_page(
        self, heading_text: str, bullets: list[str], instructions: str
    ) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)

        heading = Gtk.Label(label=heading_text)
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        box.append(heading)

        for text in bullets:
            bullet = Gtk.Label(label=f"• {text}")
            bullet.set_wrap(True)
            bullet.set_xalign(0.0)
            box.append(bullet)

        self._register_instructions(box, instructions)
        return box

    def _build_administrator_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About the Administrator",
            [
                "Create the first administrator so there's a trusted owner from day one.",
                "We'll reuse their contact and domain details to prefill later steps.",
            ],
            "Grab a strong password and the administrator's contact info, then continue when you're set.",
        )

    def _build_database_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About the Database",
            [
                "ATLAS keeps conversations and configuration in a dedicated database.",
                "Choose between PostgreSQL, SQLite, or MongoDB/Atlas based on your deployment.",
                "We'll verify the backend you select, including connection strings and Atlas SRV URLs.",
            ],
            "Keep the relevant connection details or URI nearby so entering them is quick.",
        )

    def _build_job_scheduling_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About Job Scheduling",
            [
                "Background jobs refresh caches and handle maintenance tasks for ATLAS.",
                "You'll choose how those jobs queue and which backend coordinates them.",
            ],
            "Think about the scheduler service you plan to use so the form choices feel familiar.",
        )

    def _build_message_bus_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About the Message Bus",
            [
                "The message bus keeps ATLAS services talking to each other.",
                "Common choices include NATS, RabbitMQ, or Redis streams—pick what fits your stack.",
            ],
            "Have the broker URL and credentials ready for the next screen.",
        )

    def _build_kv_store_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About the Key-Value Store",
            [
                "A key-value backend keeps short-lived state like rate limits and caches.",
                "Redis works great, but any compatible service with a DSN will do.",
            ],
            "Jot down the hostname, port, and credentials so you can plug them in quickly.",
        )

    def _build_speech_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About Speech Services",
            [
                "Speech synthesis and recognition unlock voice-driven experiences in ATLAS.",
                "We'll ask for provider choices, API keys, and preferred default voices next.",
            ],
            "Pick the providers you want to enable and have their credentials ready to drop in.",
        )

    def _build_optional_intro_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)

        heading = Gtk.Label(label="About the Organization")
        heading.set_wrap(True)
        heading.set_xalign(0.0)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        box.append(heading)

        summary_bullets = [
            "Seed tenancy defaults, retention expectations, and scheduler notes so ATLAS mirrors your policies.",
            "Large teams often standardize retention workers, queue sizing, and namespaces before go-live.",
        ]
        for text in summary_bullets:
            bullet = Gtk.Label(label=f"• {text}")
            bullet.set_wrap(True)
            bullet.set_xalign(0.0)
            box.append(bullet)

        callout_frame = Gtk.Frame()
        callout_frame.set_hexpand(True)
        callout_frame.set_vexpand(False)

        callout_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        callout_box.set_margin_top(6)
        callout_box.set_margin_bottom(6)
        callout_box.set_margin_start(12)
        callout_box.set_margin_end(12)

        runbook_uri: str | None = None
        try:
            runbook_path = Path(__file__).resolve().parents[2] / "docs" / "conversation_retention.md"
            runbook_uri = GLib.filename_to_uri(str(runbook_path), None)
        except Exception:  # pragma: no cover - defensive fallback
            runbook_uri = None

        callout_heading = Gtk.Label(label="Scaling reminders")
        callout_heading.set_wrap(True)
        callout_heading.set_xalign(0.0)
        if hasattr(callout_heading, "add_css_class"):
            callout_heading.add_css_class("heading")
        callout_box.append(callout_heading)

        callout_points = [
            "Tenant defaults start with the administrator domain—adjust if teams need their own space.",
            "Match retention worker cadence with the conversation retention runbook before tightening purges.",
        ]

        callout_points.extend(
            [
                "Document scheduler overrides—queue sizes, time zones, cadence—so workers match production.",
                "Review residency, encryption, and deletion safeguards with stakeholders before scaling up.",
            ]
        )

        for text in callout_points:
            label = Gtk.Label(label=f"• {text}")
            label.set_wrap(True)
            label.set_xalign(0.0)
            callout_box.append(label)

        if runbook_uri:
            link_widget: Gtk.Widget | None = None
            link_button_cls = getattr(Gtk, "LinkButton", None)
            if link_button_cls is not None:
                try:
                    link_widget = link_button_cls.new_with_label(
                        runbook_uri, "Open conversation retention runbook"
                    )
                except Exception:  # pragma: no cover - fallback guard
                    link_widget = None
            if link_widget is not None:
                if hasattr(link_widget, "set_halign"):
                    link_widget.set_halign(Gtk.Align.START)
                callout_box.append(link_widget)
            else:
                fallback_label = Gtk.Label(label=f"Runbook: {runbook_uri}")
                fallback_label.set_wrap(True)
                fallback_label.set_xalign(0.0)
                callout_box.append(fallback_label)

        if hasattr(callout_frame, "set_child"):
            callout_frame.set_child(callout_box)
        else:  # pragma: no cover - GTK3 fallback
            callout_frame.add(callout_box)

        box.append(callout_frame)

        instructions = (
            "• Decide which organizational defaults matter most before moving on.\n"
            "• Jot down any retention or scheduling nuances you want to capture on the next form.\n"
            "• Loop in stakeholders if you need buy-in on safeguards before saving."
        )

        self._register_instructions(box, instructions)
        self._register_instructions(
            callout_frame,
            (
                "Use these reminders to double-check policies and documentation needs before you continue."
            ),
        )
        return box

    def _build_provider_intro_page(self) -> Gtk.Widget:
        return self._create_intro_page(
            "About Providers",
            [
                "Choose the default providers and models new conversations should start with.",
                "You'll tune provider-specific settings on the following pages when more than one is enabled.",
            ],
            "Decide which providers you'll enable so the next forms go quickly.",
        )

    def _select_step_row(self, index: int) -> None:
        if not self._step_list:
            return
        if not (0 <= index < len(self._step_rows)):
            return
        self._step_list.select_row(self._step_rows[index])

    def _build_database_page(self) -> Gtk.Widget:
        state = self.controller.state.database
        self._database_entries.clear()

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        container.set_hexpand(True)
        container.set_vexpand(True)

        backend_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        backend_label = Gtk.Label(label="Backend")
        backend_label.set_xalign(0.0)
        backend_grid.attach(backend_label, 0, 0, 1, 1)

        backend_combo = Gtk.ComboBoxText()
        backend_combo.append("postgresql", "PostgreSQL")
        backend_combo.append("sqlite", "SQLite")
        backend_combo.append("mongodb", "MongoDB / Atlas")
        backend_combo.set_hexpand(False)
        backend_combo.set_halign(Gtk.Align.START)
        backend_combo.set_size_request(self._get_entry_pixel_width(), -1)
        backend_grid.attach(backend_combo, 1, 0, 1, 1)
        container.append(backend_grid)

        pg_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        self._database_entries["postgresql.host"] = self._create_labeled_entry(
            pg_grid, 0, "Host", ""
        )
        self._database_entries["postgresql.port"] = self._create_labeled_entry(
            pg_grid, 1, "Port", ""
        )
        self._database_entries["postgresql.database"] = self._create_labeled_entry(
            pg_grid, 2, "Database", ""
        )
        self._database_entries["postgresql.user"] = self._create_labeled_entry(
            pg_grid, 3, "User", ""
        )
        self._database_entries["postgresql.password"] = self._create_labeled_entry(
            pg_grid, 4, "Password", "", visibility=False
        )

        sqlite_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        self._database_entries["sqlite.path"] = self._create_labeled_entry(
            sqlite_grid,
            0,
            "Database file",
            "",
            placeholder="atlas.sqlite3 or an absolute path",
        )

        mongo_grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        self._database_entries["mongodb.uri"] = self._create_labeled_entry(
            mongo_grid,
            0,
            "Connection string",
            "",
            placeholder="mongodb:// or mongodb+srv://",
        )
        self._database_entries["mongodb.host"] = self._create_labeled_entry(
            mongo_grid, 1, "Host", ""
        )
        self._database_entries["mongodb.port"] = self._create_labeled_entry(
            mongo_grid, 2, "Port", ""
        )
        self._database_entries["mongodb.database"] = self._create_labeled_entry(
            mongo_grid, 3, "Database", ""
        )
        self._database_entries["mongodb.user"] = self._create_labeled_entry(
            mongo_grid, 4, "User", ""
        )
        self._database_entries["mongodb.password"] = self._create_labeled_entry(
            mongo_grid, 5, "Password", "", visibility=False
        )
        self._database_entries["mongodb.options"] = self._create_labeled_entry(
            mongo_grid,
            6,
            "Options",
            "",
            placeholder="retryWrites=true&w=majority",
        )

        stack = Gtk.Stack()
        stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        stack.add_named(pg_grid, "postgresql")
        stack.add_named(sqlite_grid, "sqlite")
        stack.add_named(mongo_grid, "mongodb")
        stack.set_visible_child_name("postgresql")
        container.append(stack)

        backend_combo.connect("changed", self._on_database_backend_changed, stack)

        self._database_backend_combo = backend_combo
        self._database_stack = stack

        # Seed defaults then hydrate from state so switching back retains entries.
        self._set_database_entry_text("postgresql.host", "localhost")
        self._set_database_entry_text("postgresql.port", "5432")
        self._set_database_entry_text("postgresql.database", "atlas")
        self._set_database_entry_text("postgresql.user", "atlas")
        self._set_database_entry_text("postgresql.password", "")

        self._set_database_entry_text("sqlite.path", "atlas.sqlite3")

        self._set_database_entry_text("mongodb.uri", "")
        self._set_database_entry_text("mongodb.host", "localhost")
        self._set_database_entry_text("mongodb.port", "27017")
        self._set_database_entry_text("mongodb.database", "atlas")
        self._set_database_entry_text("mongodb.user", "")
        self._set_database_entry_text("mongodb.password", "")
        self._set_database_entry_text("mongodb.options", "")

        self._populate_database_form(state)
        self._database_user_suggestion = self._get_database_entry_text("postgresql.user")

        instructions = (
            "Pick the backend you want to run and share its connection details."
            " PostgreSQL suits production clusters, SQLite is handy for local demos,"
            " and MongoDB/Atlas works with managed or SRV-based deployments."
        )

        return self._wrap_with_instructions(container, instructions, "Configure Database")

    def _set_database_entry_text(
        self, key: str, value: str | int | None
    ) -> None:
        entry = self._database_entries.get(key)
        if isinstance(entry, Gtk.Entry):
            text = "" if value is None else str(value)
            entry.set_text(text)

    def _get_database_entry_text(self, key: str, *, strip: bool = True) -> str:
        entry = self._database_entries.get(key)
        if not isinstance(entry, Gtk.Entry):
            return ""
        text = entry.get_text()
        return text.strip() if strip else text

    def _populate_database_form(self, state: DatabaseState) -> None:
        backend = (state.backend or "postgresql").strip().lower() or "postgresql"
        if backend not in {"postgresql", "sqlite", "mongodb"}:
            backend = "postgresql"
        if self._database_backend_combo is not None:
            try:
                self._database_backend_combo.set_active_id(backend)
            except Exception:
                pass

        if backend == "postgresql":
            self._set_database_entry_text("postgresql.host", state.host or "localhost")
            self._set_database_entry_text("postgresql.port", state.port or 5432)
            self._set_database_entry_text("postgresql.database", state.database or "atlas")
            self._set_database_entry_text("postgresql.user", state.user or "atlas")
            self._set_database_entry_text("postgresql.password", state.password or "")
        elif backend == "sqlite":
            database = state.database or state.dsn or "atlas.sqlite3"
            self._set_database_entry_text("sqlite.path", database)
        elif backend == "mongodb":
            self._set_database_entry_text("mongodb.uri", state.dsn or "")
            self._set_database_entry_text("mongodb.host", state.host or "localhost")
            self._set_database_entry_text("mongodb.port", state.port or 27017)
            self._set_database_entry_text("mongodb.database", state.database or "atlas")
            self._set_database_entry_text("mongodb.user", state.user or "")
            self._set_database_entry_text("mongodb.password", state.password or "")
            self._set_database_entry_text("mongodb.options", state.options or "")

        if self._database_stack is not None:
            try:
                self._database_stack.set_visible_child_name(backend)
            except Exception:
                pass
        self._database_user_suggestion = self._get_database_entry_text("postgresql.user")

    def _on_database_backend_changed(
        self, combo: Gtk.ComboBoxText, stack: Gtk.Stack
    ) -> None:
        backend = combo.get_active_id() or "postgresql"
        if backend not in {"postgresql", "sqlite", "mongodb"}:
            backend = "postgresql"
        try:
            stack.set_visible_child_name(backend)
        except Exception:
            pass

    def _collect_database_state(self, *, strict: bool = False) -> DatabaseState:
        current = self.controller.state.database
        backend = current.backend or "postgresql"
        if self._database_backend_combo is not None:
            active = self._database_backend_combo.get_active_id()
            if active:
                backend = active
        normalized = (backend or "postgresql").strip().lower() or "postgresql"

        if normalized == "sqlite":
            path_text = self._get_database_entry_text("sqlite.path")
            if not path_text:
                if current.backend == "sqlite" and current.database:
                    path_text = current.database
                else:
                    path_text = "atlas.sqlite3"
            return DatabaseState(
                backend="sqlite",
                host="",
                port=0,
                database=path_text,
                user="",
                password="",
                dsn="",
                options="",
            )

        if normalized == "mongodb":
            uri = self._get_database_entry_text("mongodb.uri")
            host = self._get_database_entry_text("mongodb.host") or (
                current.host if current.backend == "mongodb" and current.host else "localhost"
            )
            default_port = (
                current.port if current.backend == "mongodb" and current.port else 27017
            )
            port_text = self._get_database_entry_text("mongodb.port")
            port = default_port
            if port_text:
                try:
                    port = int(port_text)
                except ValueError as exc:
                    if strict:
                        raise ValueError("MongoDB port must be a valid integer") from exc
                    port = default_port
            database = self._get_database_entry_text("mongodb.database") or (
                current.database if current.backend == "mongodb" and current.database else "atlas"
            )
            user = self._get_database_entry_text("mongodb.user")
            password = self._get_database_entry_text("mongodb.password", strip=False)
            options = self._get_database_entry_text("mongodb.options")
            return DatabaseState(
                backend="mongodb",
                host=host or "localhost",
                port=port,
                database=database or "atlas",
                user=user,
                password=password,
                dsn=uri,
                options=options,
            )

        host = self._get_database_entry_text("postgresql.host") or (
            current.host if current.backend == "postgresql" and current.host else "localhost"
        )
        default_port = (
            current.port if current.backend == "postgresql" and current.port else 5432
        )
        port_text = self._get_database_entry_text("postgresql.port")
        port = default_port
        if port_text:
            try:
                port = int(port_text)
            except ValueError as exc:
                if strict:
                    raise ValueError("PostgreSQL port must be a valid integer") from exc
                port = default_port
        database_name = self._get_database_entry_text("postgresql.database") or (
            current.database if current.backend == "postgresql" and current.database else "atlas"
        )
        user = self._get_database_entry_text("postgresql.user") or (
            current.user if current.backend == "postgresql" and current.user else "atlas"
        )
        password = self._get_database_entry_text("postgresql.password", strip=False)
        return DatabaseState(
            backend="postgresql",
            host=host or "localhost",
            port=port,
            database=database_name or "atlas",
            user=user or "atlas",
            password=password,
            dsn="",
            options="",
        )

    def _build_provider_pages(self) -> list[Gtk.Widget]:
        state = self.controller.state.providers

        defaults_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        defaults_box.set_hexpand(True)
        defaults_box.set_vexpand(True)

        self._provider_entries["default_provider"] = self._create_entry(
            "Default provider", state.default_provider or ""
        )
        self._provider_entries["default_model"] = self._create_entry(
            "Default model", state.default_model or ""
        )

        defaults_box.append(self._provider_entries["default_provider"])
        defaults_box.append(self._provider_entries["default_model"])

        defaults_instructions = (
            "Pick the default provider and model. New workspaces inherit these unless they choose otherwise."
        )
        defaults_form = self._wrap_with_instructions(
            defaults_box, defaults_instructions, "Configure defaults"
        )

        keys_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        keys_box.set_hexpand(True)
        keys_box.set_vexpand(True)

        helper_label = Gtk.Label(
            label=(
                "Enter each provider and key on its own line using provider=key. "
                "Keys are stored in your local configuration and only sent to the provider when needed."
            )
        )
        helper_label.set_wrap(True)
        helper_label.set_xalign(0.0)
        keys_box.append(helper_label)

        show_toggle = Gtk.CheckButton(label="Show keys")
        if hasattr(show_toggle, "set_tooltip_text"):
            show_toggle.set_tooltip_text(
                "Temporarily reveal saved API keys to review or edit them."
            )
        keys_box.append(show_toggle)

        stack = Gtk.Stack()
        stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        stack.set_transition_duration(150)
        stack.set_hexpand(True)
        stack.set_vexpand(True)

        visible_view = Gtk.TextView()
        visible_view.set_monospace(True)
        visible_view.set_hexpand(True)
        visible_view.set_vexpand(True)
        wrap_mode = getattr(Gtk.WrapMode, "CHAR", getattr(Gtk.WrapMode, "WORD_CHAR", None))
        if wrap_mode is not None:
            visible_view.set_wrap_mode(wrap_mode)
        visible_buffer = visible_view.get_buffer()
        visible_buffer.set_text(self._format_api_keys(state.api_keys))

        masked_view = Gtk.TextView()
        masked_view.set_monospace(True)
        masked_view.set_editable(False)
        masked_view.set_cursor_visible(False)
        if hasattr(masked_view, "set_focusable"):
            masked_view.set_focusable(False)
        elif hasattr(masked_view, "set_can_focus"):
            masked_view.set_can_focus(False)
        if wrap_mode is not None:
            masked_view.set_wrap_mode(wrap_mode)

        visible_scroller = Gtk.ScrolledWindow()
        visible_scroller.set_hexpand(True)
        visible_scroller.set_vexpand(True)
        visible_scroller.set_child(visible_view)

        masked_scroller = Gtk.ScrolledWindow()
        masked_scroller.set_hexpand(True)
        masked_scroller.set_vexpand(True)
        masked_scroller.set_child(masked_view)

        stack.add_named(masked_scroller, "masked")
        stack.add_named(visible_scroller, "visible")
        stack.set_visible_child_name("masked")

        keys_box.append(stack)

        self._provider_buffer = visible_buffer
        self._provider_mask_buffer = masked_view.get_buffer()
        self._provider_stack = stack
        self._provider_show_toggle = show_toggle

        visible_buffer.connect("changed", self._on_provider_buffer_changed)
        show_toggle.connect("toggled", self._on_provider_show_toggled)

        self._on_provider_buffer_changed(visible_buffer)

        keys_instructions = (
            "Add API keys for any providers you plan to use. We only display the full values while the toggle is enabled."
        )
        keys_form = self._wrap_with_instructions(
            keys_box, keys_instructions, "API keys"
        )

        return [defaults_form, keys_form]

    def _build_kv_store_page(self) -> Gtk.Widget:
        state = self.controller.state.kv_store
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_hexpand(True)
        box.set_vexpand(True)

        reuse_toggle = Gtk.CheckButton(label="Reuse conversation store for key-value storage")
        reuse_toggle.set_active(state.reuse_conversation_store)
        self._kv_widgets["reuse"] = reuse_toggle
        box.append(reuse_toggle)

        url_entry = Gtk.Entry()
        url_entry.set_placeholder_text("Key-value store DSN (leave blank to skip)")
        url_entry.set_text(state.url or "")
        url_entry.set_hexpand(True)
        self._kv_widgets["url"] = url_entry
        box.append(url_entry)

        instructions = (
            "Reuse the conversation store for key-value data or share a dedicated DSN if you prefer a separate service."
        )

        return self._wrap_with_instructions(box, instructions, "Configure Key-Value Store")

    def _build_job_scheduling_page(self) -> Gtk.Widget:
        state = self.controller.state.job_scheduling
        retry = state.retry_policy
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        enable_toggle = Gtk.CheckButton(label="Enable background job scheduling")
        enable_toggle.set_active(state.enabled)
        enable_toggle.set_hexpand(False)
        self._job_widgets["enabled"] = enable_toggle
        grid.attach(enable_toggle, 0, 0, 2, 1)

        row = 1
        self._job_widgets["job_store_url"] = self._create_labeled_entry(
            grid, row, "Job store DSN", state.job_store_url or ""
        )
        row += 1
        self._job_widgets["max_workers"] = self._create_labeled_entry(
            grid, row, "Max workers", self._optional_to_text(state.max_workers)
        )
        row += 1
        self._job_widgets["timezone"] = self._create_labeled_entry(
            grid, row, "Scheduler timezone", state.timezone or ""
        )
        row += 1
        self._job_widgets["queue_size"] = self._create_labeled_entry(
            grid, row, "Queue size", self._optional_to_text(state.queue_size)
        )
        row += 1

        retry_label = Gtk.Label(label="Retry policy")
        retry_label.set_xalign(0.0)
        if hasattr(retry_label, "add_css_class"):
            retry_label.add_css_class("heading")
        grid.attach(retry_label, 0, row, 2, 1)
        row += 1

        self._job_widgets["retry_max_attempts"] = self._create_labeled_entry(
            grid, row, "Max attempts", str(retry.max_attempts)
        )
        retry_max_entry = self._job_widgets["retry_max_attempts"]
        if isinstance(retry_max_entry, Gtk.Entry):
            self._register_required_number(
                retry_max_entry,
                "Max attempts",
                parse=lambda value: int(value),
                number_label="whole number",
            )
        row += 1
        self._job_widgets["retry_backoff_seconds"] = self._create_labeled_entry(
            grid, row, "Backoff seconds", self._format_float(retry.backoff_seconds)
        )
        backoff_entry = self._job_widgets["retry_backoff_seconds"]
        if isinstance(backoff_entry, Gtk.Entry):
            self._register_required_number(
                backoff_entry,
                "Backoff seconds",
                parse=lambda value: float(value),
                number_label="number",
            )
        row += 1
        self._job_widgets["retry_jitter_seconds"] = self._create_labeled_entry(
            grid, row, "Jitter seconds", self._format_float(retry.jitter_seconds)
        )
        jitter_entry = self._job_widgets["retry_jitter_seconds"]
        if isinstance(jitter_entry, Gtk.Entry):
            self._register_required_number(
                jitter_entry,
                "Jitter seconds",
                parse=lambda value: float(value),
                number_label="number",
            )
        row += 1
        self._job_widgets["retry_backoff_multiplier"] = self._create_labeled_entry(
            grid, row, "Backoff multiplier", self._format_float(retry.backoff_multiplier)
        )
        multiplier_entry = self._job_widgets["retry_backoff_multiplier"]
        if isinstance(multiplier_entry, Gtk.Entry):
            self._register_required_number(
                multiplier_entry,
                "Backoff multiplier",
                parse=lambda value: float(value),
                number_label="number",
            )

        instructions = (
            "• Turn scheduling on when you want background jobs to run.\n"
            "• Share DSNs and limits that match how you plan to operate workers.\n"
            "• Capture retry details so teammates know what to expect."
        )

        form = self._wrap_with_instructions(
            grid, instructions, "Configure Job Scheduling"
        )
        return form

    def _build_message_bus_page(self) -> Gtk.Widget:
        state = self.controller.state.message_bus
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        backend_label = Gtk.Label(label="Backend")
        backend_label.set_xalign(0.0)
        grid.attach(backend_label, 0, 0, 1, 1)

        backend_combo = Gtk.ComboBoxText()
        backend_combo.append("in_memory", "In-memory")
        backend_combo.append("redis", "Redis")
        backend_combo.set_active_id(state.backend or "in_memory")
        backend_combo.set_hexpand(False)
        backend_combo.set_halign(Gtk.Align.START)
        backend_combo.set_size_request(self._get_entry_pixel_width(), -1)
        grid.attach(backend_combo, 1, 0, 1, 1)
        self._message_widgets["backend"] = backend_combo

        self._message_widgets["redis_url"] = self._create_labeled_entry(
            grid, 1, "Redis URL", state.redis_url or ""
        )
        self._message_widgets["stream_prefix"] = self._create_labeled_entry(
            grid, 2, "Stream prefix", state.stream_prefix or ""
        )

        instructions = (
            "Choose the message bus backend. Redis keeps multiple workers in sync, while in-memory suits single instances."
        )

        return self._wrap_with_instructions(grid, instructions, "Configure Message Bus")

    def _build_speech_page(self) -> Gtk.Widget:
        state = self.controller.state.speech
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        tts_toggle = Gtk.CheckButton(label="Enable text-to-speech")
        tts_toggle.set_active(state.tts_enabled)
        self._speech_widgets["tts_enabled"] = tts_toggle
        grid.attach(tts_toggle, 0, 0, 2, 1)

        stt_toggle = Gtk.CheckButton(label="Enable speech-to-text")
        stt_toggle.set_active(state.stt_enabled)
        self._speech_widgets["stt_enabled"] = stt_toggle
        grid.attach(stt_toggle, 0, 1, 2, 1)

        self._speech_widgets["default_tts"] = self._create_labeled_entry(
            grid, 2, "Default TTS provider", state.default_tts_provider or ""
        )
        self._speech_widgets["default_stt"] = self._create_labeled_entry(
            grid, 3, "Default STT provider", state.default_stt_provider or ""
        )
        self._speech_widgets["elevenlabs_key"] = self._create_labeled_entry(
            grid, 4, "ElevenLabs API key", state.elevenlabs_key or ""
        )
        self._speech_widgets["openai_key"] = self._create_labeled_entry(
            grid, 5, "OpenAI speech API key", state.openai_key or ""
        )
        self._speech_widgets["google_credentials"] = self._create_labeled_entry(
            grid, 6, "Google speech credentials path", state.google_credentials or ""
        )

        instructions = (
            "Toggle the speech features you need and drop in credentials for the providers you'll rely on."
        )

        return self._wrap_with_instructions(grid, instructions, "Configure Speech Services")

    def _build_optional_page(self) -> Gtk.Widget:
        state = self.controller.state.optional
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        row = 0

        personal_hint = Gtk.Label(label="Most people can keep the defaults.")
        personal_hint.set_wrap(True)
        personal_hint.set_xalign(0.0)
        personal_hint.set_visible(False)
        grid.attach(personal_hint, 0, row, 2, 1)
        self._optional_personal_hint = personal_hint

        row += 1

        self._optional_widgets["tenant_id"] = self._create_labeled_entry(
            grid, row, "Tenant ID", state.tenant_id or ""
        )
        tenant_entry = self._optional_widgets["tenant_id"]
        if isinstance(tenant_entry, Gtk.Entry):
            self._tenant_id_suggestion = tenant_entry.get_text().strip()
        row += 1
        self._optional_widgets["retention_days"] = self._create_labeled_entry(
            grid, row, "Conversation retention days", self._optional_to_text(state.retention_days)
        )
        row += 1
        self._optional_widgets["retention_history_limit"] = self._create_labeled_entry(
            grid,
            row,
            "Conversation history limit",
            self._optional_to_text(state.retention_history_limit),
        )
        row += 1
        self._optional_widgets["scheduler_timezone"] = self._create_labeled_entry(
            grid, row, "Scheduler timezone", state.scheduler_timezone or ""
        )
        row += 1
        self._optional_widgets["scheduler_queue_size"] = self._create_labeled_entry(
            grid,
            row,
            "Scheduler queue size",
            self._optional_to_text(state.scheduler_queue_size),
        )
        row += 1

        http_toggle = Gtk.CheckButton(label="Auto-start HTTP server")
        http_toggle.set_active(state.http_auto_start)
        self._optional_widgets["http_auto_start"] = http_toggle
        grid.attach(http_toggle, 0, row, 2, 1)
        row += 1

        callout_frame = Gtk.Frame()
        callout_frame.set_hexpand(True)
        callout_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        callout_box.set_margin_top(6)
        callout_box.set_margin_bottom(6)
        callout_box.set_margin_start(12)
        callout_box.set_margin_end(12)

        runbook_uri: str | None = None
        try:
            runbook_path = Path(__file__).resolve().parents[2] / "docs" / "conversation_retention.md"
            runbook_uri = GLib.filename_to_uri(str(runbook_path), None)
        except Exception:  # pragma: no cover - defensive fallback
            runbook_uri = None

        info_heading = Gtk.Label(label="Tips for larger teams")
        info_heading.set_wrap(True)
        info_heading.set_xalign(0.0)
        if hasattr(info_heading, "add_css_class"):
            info_heading.add_css_class("heading")
        callout_box.append(info_heading)

        tenancy_info = Gtk.Label(
            label=(
                "Tenant defaults start with the administrator domain—tweak them if teams need their own space."
            )
        )
        tenancy_info.set_wrap(True)
        tenancy_info.set_xalign(0.0)
        callout_box.append(tenancy_info)

        retention_info = Gtk.Label(
            label=(
                "Match retention windows with the cadence documented in docs/conversation_retention.md so purge jobs stay predictable."
            )
        )
        retention_info.set_wrap(True)
        retention_info.set_xalign(0.0)
        callout_box.append(retention_info)

        if runbook_uri:
            link_widget: Gtk.Widget | None = None
            link_button_cls = getattr(Gtk, "LinkButton", None)
            if link_button_cls is not None:
                try:
                    link_widget = link_button_cls.new_with_label(
                        runbook_uri, "Open conversation retention runbook"
                    )
                except Exception:  # pragma: no cover - fallback guard
                    link_widget = None
            if link_widget is not None:
                if hasattr(link_widget, "set_halign"):
                    link_widget.set_halign(Gtk.Align.START)
                callout_box.append(link_widget)
            else:
                fallback_label = Gtk.Label(label=f"Runbook: {runbook_uri}")
                fallback_label.set_wrap(True)
                fallback_label.set_xalign(0.0)
                callout_box.append(fallback_label)

        scheduler_info = Gtk.Label(
            label=(
                "Note any scheduler overrides—queue sizes, time zones—so operators keep peak workloads in check."
            )
        )
        scheduler_info.set_wrap(True)
        scheduler_info.set_xalign(0.0)
        callout_box.append(scheduler_info)

        safeguards_info = Gtk.Label(
            label=(
                "Record shared safeguards like audit logging, residency requirements, and regular retention reviews."
            )
        )
        safeguards_info.set_wrap(True)
        safeguards_info.set_xalign(0.0)
        callout_box.append(safeguards_info)

        if hasattr(callout_frame, "set_child"):
            callout_frame.set_child(callout_box)
        else:  # pragma: no cover - GTK3 fallback
            callout_frame.add(callout_box)

        grid.attach(callout_frame, 0, row, 2, 1)

        instructions = (
            "• Set tenant defaults and retention expectations that fit your rollout.\n"
            "• Share scheduler tweaks so background jobs line up with your policies.\n"
            "• Decide whether ATLAS should auto-start its HTTP server for you."
        )

        form = self._wrap_with_instructions(
            grid, instructions, "Configure Organization"
        )
        self._register_instructions(
            callout_frame,
            (
                "Capture any notes that will help future teammates understand your organizational defaults."
            ),
        )
        return form

    def _build_user_page(self) -> Gtk.Widget:
        state = self.controller.state.user
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)
        grid.set_hexpand(True)
        grid.set_vexpand(True)

        row = 0
        self._user_entries["full_name"] = self._create_labeled_entry(
            grid, row, "Full name", state.full_name
        )
        row += 1
        self._user_entries["username"] = self._create_labeled_entry(grid, row, "Username", state.username)
        row += 1
        self._user_entries["email"] = self._create_labeled_entry(grid, row, "Email", state.email)
        row += 1
        self._user_entries["domain"] = self._create_labeled_entry(
            grid,
            row,
            "Domain",
            state.domain,
            placeholder="example.com",
        )
        row += 1
        dob_entry = self._create_labeled_entry(
            grid,
            row,
            "Date of birth",
            state.date_of_birth,
            placeholder="YYYY-MM-DD",
        )
        # ``Gtk.InputPurpose.DATE`` was added in GTK 4. Fall back to a broadly
        # compatible purpose so the widget still works under GTK 3 where the
        # enum value may not exist.
        if hasattr(Gtk.InputPurpose, "DATE"):
            dob_entry.set_input_purpose(Gtk.InputPurpose.DATE)
        elif hasattr(Gtk.InputPurpose, "FREE_FORM"):
            dob_entry.set_input_purpose(Gtk.InputPurpose.FREE_FORM)
        self._user_entries["date_of_birth"] = dob_entry
        row += 1
        self._user_entries["password"] = self._create_labeled_entry(
            grid, row, "Password", state.password, visibility=False
        )
        password_entry = self._user_entries["password"]
        self._register_required_text(password_entry, "Password", strip=False)
        row += 1
        self._user_entries["confirm_password"] = self._create_labeled_entry(
            grid, row, "Confirm password", state.password, visibility=False
        )
        confirm_password_entry = self._user_entries["confirm_password"]
        self._register_password_confirmation(password_entry, confirm_password_entry)
        row += 1

        privileged_label = Gtk.Label(label="System privileged credentials")
        privileged_label.set_xalign(0.0)
        if hasattr(privileged_label, "add_css_class"):
            privileged_label.add_css_class("heading")
        grid.attach(privileged_label, 0, row, 2, 1)
        row += 1

        privileged_state = state.privileged_credentials
        self._user_entries["sudo_username"] = self._create_labeled_entry(
            grid, row, "Sudo username", privileged_state.sudo_username
        )
        row += 1
        self._user_entries["sudo_password"] = self._create_labeled_entry(
            grid,
            row,
            "Sudo password",
            privileged_state.sudo_password,
            visibility=False,
        )
        sudo_password_entry = self._user_entries["sudo_password"]
        row += 1
        self._user_entries["confirm_sudo_password"] = self._create_labeled_entry(
            grid,
            row,
            "Confirm sudo password",
            privileged_state.sudo_password,
            visibility=False,
        )
        confirm_sudo_entry = self._user_entries["confirm_sudo_password"]

        def _sudo_validator() -> tuple[bool, str | None]:
            sudo_password = sudo_password_entry.get_text()
            confirm_sudo = confirm_sudo_entry.get_text()
            if not sudo_password and not confirm_sudo:
                return True, None
            if sudo_password and not confirm_sudo:
                return False, "Confirm sudo password is required"
            if sudo_password != confirm_sudo:
                return False, "Sudo passwords must match"
            return True, None

        self._register_validation(confirm_sudo_entry, _sudo_validator)
        self._register_linked_validation_trigger(sudo_password_entry, confirm_sudo_entry)

        instructions = (
            "Share the administrator's contact details, optional domain, and any privileged credentials so we can stage them for later steps."
        )

        form = self._wrap_with_instructions(
            grid, instructions, "Configure Administrator"
        )
        username_entry = self._user_entries["username"]
        self._register_required_text(username_entry, "Username")
        self._sync_user_entries_from_state()
        return form

    def _sync_user_entries_from_state(self) -> None:
        state = self.controller.state.user
        privileged_state = state.privileged_credentials
        mapping = {
            "full_name": state.full_name,
            "username": state.username,
            "email": state.email,
            "domain": state.domain,
            "date_of_birth": state.date_of_birth,
            "password": state.password,
            "confirm_password": state.password,
            "sudo_username": privileged_state.sudo_username,
            "sudo_password": privileged_state.sudo_password,
            "confirm_sudo_password": privileged_state.sudo_password,
        }
        for key, value in mapping.items():
            entry = self._user_entries.get(key)
            if isinstance(entry, Gtk.Entry):
                entry.set_text(value or "")

    def _create_labeled_entry(
        self,
        grid: Gtk.Grid,
        row: int,
        label: str,
        value: str,
        *,
        placeholder: str | None = None,
        visibility: bool = True,
    ) -> Gtk.Entry:
        label_widget = Gtk.Label(label=label)
        label_widget.set_xalign(0.0)
        grid.attach(label_widget, 0, row, 1, 1)

        entry = Gtk.Entry()
        entry.set_visibility(visibility)
        entry.set_text(value)
        entry.set_hexpand(False)
        entry.set_width_chars(25)
        entry.set_max_width_chars(25)
        if placeholder:
            entry.set_placeholder_text(placeholder)
        grid.attach(entry, 1, row, 1, 1)
        return entry

    def _get_entry_pixel_width(self) -> int:
        if self._entry_pixel_width is None:
            entry = Gtk.Entry()
            entry.set_hexpand(False)
            entry.set_width_chars(25)
            entry.set_max_width_chars(25)
            minimum, natural, _, _ = entry.measure(Gtk.Orientation.HORIZONTAL, -1)
            width = max(natural, minimum)
            self._entry_pixel_width = width or 250
        return self._entry_pixel_width

    def _create_entry(self, placeholder: str, value: str) -> Gtk.Entry:
        entry = Gtk.Entry()
        entry.set_placeholder_text(placeholder)
        entry.set_text(value)
        entry.set_hexpand(False)
        entry.set_width_chars(25)
        entry.set_max_width_chars(25)
        return entry

    def _register_validation(
        self,
        widget: Gtk.Widget,
        validator: Callable[[], tuple[bool, str | None]],
        *,
        signal: str = "changed",
    ) -> None:
        if widget in self._validation_rules:
            self._validation_rules[widget] = validator
        else:
            self._validation_rules[widget] = validator
            getter = getattr(widget, "get_tooltip_text", None)
            if callable(getter):
                try:
                    current = getter()
                except Exception:  # pragma: no cover - GTK stubs
                    current = None
            else:
                current = None
            self._validation_base_tooltips.setdefault(widget, current)

            desc_getter = getattr(widget, "get_accessible_description", None)
            if callable(desc_getter):
                try:
                    description = desc_getter()
                except Exception:  # pragma: no cover - GTK stubs
                    description = None
            else:
                description = None
            self._validation_base_descriptions.setdefault(widget, description)

        connector = getattr(widget, "connect", None)
        handler_id: int | None = None
        if callable(connector):
            try:
                handler_id = connector(signal, lambda *_args, w=widget: self._run_validation(w))
            except Exception:  # pragma: no cover - GTK stubs
                handler_id = None
        if handler_id is not None:
            self._validation_signal_ids[widget] = handler_id
        self._run_validation(widget)

    def _register_linked_validation_trigger(
        self, source: Gtk.Widget, target: Gtk.Widget
    ) -> None:
        connector = getattr(source, "connect", None)
        if callable(connector):
            try:
                connector("changed", lambda *_args, w=target: self._run_validation(w))
            except Exception:  # pragma: no cover - GTK stubs
                pass

    def _register_required_text(
        self, entry: Gtk.Entry, field: str, *, strip: bool = True
    ) -> None:
        def _validator() -> tuple[bool, str | None]:
            text = entry.get_text()
            if strip:
                text = text.strip()
            if not text:
                return False, f"{field} is required"
            return True, None

        self._register_validation(entry, _validator)

    def _register_password_confirmation(
        self, password_entry: Gtk.Entry, confirm_entry: Gtk.Entry
    ) -> None:
        def _validator() -> tuple[bool, str | None]:
            password = password_entry.get_text()
            confirm = confirm_entry.get_text()
            if not confirm:
                return False, "Confirm password is required"
            if password != confirm:
                return False, "Passwords must match"
            return True, None

        self._register_validation(confirm_entry, _validator)
        self._register_linked_validation_trigger(password_entry, confirm_entry)

    def _register_required_number(
        self,
        entry: Gtk.Entry,
        field: str,
        *,
        parse: Callable[[str], object],
        number_label: str,
    ) -> None:
        def _validator() -> tuple[bool, str | None]:
            text = entry.get_text().strip()
            if not text:
                return False, f"{field} is required"
            try:
                parse(text)
            except ValueError:
                return False, f"{field} must be a {number_label}"
            return True, None

        self._register_validation(entry, _validator)

    def _run_validation(self, widget: Gtk.Widget) -> None:
        validator = self._validation_rules.get(widget)
        if validator is None:
            return
        try:
            valid, message = validator()
        except Exception:  # pragma: no cover - defensive
            valid, message = False, "Invalid value"
        self._apply_validation_feedback(widget, valid, message)

    def _apply_validation_feedback(
        self, widget: Gtk.Widget, is_valid: bool, message: str | None
    ) -> None:
        css_class = "validation-error"
        add_class = getattr(widget, "add_css_class", None)
        remove_class = getattr(widget, "remove_css_class", None)
        set_tooltip = getattr(widget, "set_tooltip_text", None)
        set_description = getattr(widget, "set_accessible_description", None)

        if is_valid:
            if callable(remove_class):
                try:
                    remove_class(css_class)
                except Exception:  # pragma: no cover - GTK stubs
                    pass
            base_tooltip = self._validation_base_tooltips.get(widget)
            if callable(set_tooltip):
                try:
                    set_tooltip(base_tooltip)
                except Exception:  # pragma: no cover - GTK stubs
                    pass
            base_description = self._validation_base_descriptions.get(widget) or ""
            if callable(set_description):
                try:
                    set_description(base_description)
                except Exception:  # pragma: no cover - GTK stubs
                    pass
        else:
            if callable(add_class):
                try:
                    add_class(css_class)
                except Exception:  # pragma: no cover - GTK stubs
                    pass
            if callable(set_tooltip):
                try:
                    set_tooltip(message or "")
                except Exception:  # pragma: no cover - GTK stubs
                    pass
            if callable(set_description):
                try:
                    set_description(message or "")
                except Exception:  # pragma: no cover - GTK stubs
                    pass

    def _refresh_validation_states(self) -> None:
        for widget in list(self._validation_rules):
            self._run_validation(widget)

    def _on_provider_buffer_changed(self, buffer: Gtk.TextBuffer) -> None:
        mask_buffer = self._provider_mask_buffer
        if mask_buffer is None:
            return

        start_iter = buffer.get_start_iter()
        end_iter = buffer.get_end_iter()
        text = buffer.get_text(start_iter, end_iter, True)

        masked_text = self._mask_api_keys_text(text)
        mask_buffer.set_text(masked_text)

    def _on_provider_show_toggled(self, toggle: Gtk.CheckButton) -> None:
        stack = self._provider_stack
        if stack is None:
            return

        show_plain = toggle.get_active()
        if not show_plain:
            buffer = self._provider_buffer
            if buffer is not None:
                self._on_provider_buffer_changed(buffer)
            stack.set_visible_child_name("masked")
            return

        stack.set_visible_child_name("visible")

    def _mask_api_keys_text(self, text: str) -> str:
        if not text:
            return ""

        masked_parts: list[str] = []
        for segment in text.splitlines(keepends=True):
            newline = ""
            content = segment
            if segment.endswith("\r\n"):
                newline = "\r\n"
                content = segment[:-2]
            elif segment.endswith("\n") or segment.endswith("\r"):
                newline = segment[-1]
                content = segment[:-1]

            if "=" in content:
                prefix, suffix = content.split("=", 1)
                masked_suffix = "".join("•" if not ch.isspace() else ch for ch in suffix)
                masked_content = f"{prefix}={masked_suffix}"
            else:
                masked_content = "".join("•" if not ch.isspace() else ch for ch in content)

            masked_parts.append(masked_content + newline)

        return "".join(masked_parts)

    def _format_api_keys(self, mapping: Dict[str, str]) -> str:
        lines = [f"{provider}={key}" for provider, key in sorted(mapping.items())]
        return "\n".join(lines)

    def _optional_to_text(self, value: Optional[int]) -> str:
        return "" if value is None else str(value)

    def _format_float(self, value: float) -> str:
        return f"{value}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)

    def _parse_required_int(self, entry: Gtk.Entry, field: str) -> int:
        text = entry.get_text().strip()
        if not text:
            raise ValueError(f"{field} is required")
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer") from exc

    def _parse_optional_int(self, entry: Gtk.Entry, field: str) -> Optional[int]:
        text = entry.get_text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer") from exc

    def _parse_required_float(self, entry: Gtk.Entry, field: str) -> float:
        text = entry.get_text().strip()
        if not text:
            raise ValueError(f"{field} is required")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be a number") from exc

    # -- navigation -----------------------------------------------------

    def _get_current_page_index(self, step_index: int | None = None) -> int:
        if step_index is None:
            step_index = self._current_index
        return self._current_page_indices.get(step_index, 0)

    def _get_total_pages(self, step_index: int | None = None) -> int:
        if step_index is None:
            step_index = self._current_index
        pages = self._get_step_pages(step_index)
        return len(pages)

    def _get_current_page_widget(self) -> Gtk.Widget | None:
        pages = self._get_step_pages(self._current_index)
        if not pages:
            return None
        index = self._get_current_page_index(self._current_index)
        index = max(0, min(index, len(pages) - 1))
        return pages[index]

    def _show_subpage(self, step_index: int, page_index: int) -> None:
        pages = self._get_step_pages(step_index)
        if not pages:
            return
        page_index = max(0, min(page_index, len(pages) - 1))
        self._current_page_indices[step_index] = page_index
        stack = self._step_page_stacks.get(step_index)
        if stack is not None:
            try:
                stack.set_visible_child(pages[page_index])
            except Exception:  # pragma: no cover - defensive
                pass
        if step_index == self._current_index:
            self._update_guidance_for_widget(pages[page_index])
            self._update_navigation()
            self._update_step_status()

    def _advance_subpage(self) -> bool:
        total = self._get_total_pages()
        current = self._get_current_page_index()
        if total <= 1 or current >= total - 1:
            return False
        new_index = current + 1
        self._show_subpage(self._current_index, new_index)
        step = self._steps[self._current_index]
        self._set_status(
            self._format_page_status(step, new_index, total)
        )
        return True

    def _retreat_subpage(self) -> bool:
        total = self._get_total_pages()
        current = self._get_current_page_index()
        if total <= 1 or current == 0:
            return False
        new_index = current - 1
        self._show_subpage(self._current_index, new_index)
        step = self._steps[self._current_index]
        self._set_status(
            self._format_page_status(step, new_index, total)
        )
        return True

    def _go_to_step(self, index: int) -> None:
        if not self._steps:
            return
        self._current_index = max(0, min(index, len(self._steps) - 1))
        if not (0 <= self._current_index < len(self._step_containers)):
            return
        container = self._step_containers[self._current_index]
        self._stack.set_visible_child(container)
        current_page = self._get_current_page_index(self._current_index)
        self._show_subpage(self._current_index, current_page)
        step = self._steps[self._current_index]
        self._update_step_indicator(step.name)
        total_pages = self._get_total_pages()
        if total_pages:
            self._set_status(
                self._format_page_status(step, current_page, total_pages)
            )
        self._select_step_row(self._current_index)

    def _on_stack_visible_child_changed(
        self, stack: Gtk.Stack, _param_spec: object
    ) -> None:
        if not self._steps:
            return

        child = stack.get_visible_child()
        if child is None:
            return

        try:
            index = self._step_containers.index(child)
        except ValueError:
            return

        if index == self._current_index:
            return

        self._current_index = index
        current_page = self._get_current_page_index(index)
        self._show_subpage(index, current_page)
        self._select_step_row(self._current_index)

    def _update_step_indicator(self, name: str) -> None:
        # Deprecated: the sidebar now acts as the step indicator.
        pass

    def _update_navigation(self) -> None:
        total_pages = self._get_total_pages()
        current_page = self._get_current_page_index()
        has_previous_step = self._current_index > 0
        has_next_step = self._current_index < len(self._steps) - 1
        has_previous_page = current_page > 0
        has_next_page = total_pages > 0 and current_page < total_pages - 1

        if hasattr(self._back_button, "set_sensitive"):
            self._back_button.set_sensitive(has_previous_step or has_previous_page)

        if hasattr(self._next_button, "set_sensitive"):
            self._next_button.set_sensitive(has_next_page or has_next_step)

        if hasattr(self._next_button, "set_label"):
            if has_next_page:
                label = "Next page"
            elif not has_next_step:
                label = "Finish"
            else:
                label = "Next"
            self._next_button.set_label(label)

        back_tooltip = "Go back (Alt+Left)"
        back_description = "Use Alt+Left to move to the previous page or step"
        if has_previous_page:
            back_tooltip = "Previous page (Alt+Left)"
            back_description = "Use Alt+Left to return to the previous page"
        elif has_previous_step:
            back_tooltip = "Previous step (Alt+Left)"
            back_description = "Use Alt+Left to return to the previous step"

        next_tooltip = "Next (Alt+Right)"
        next_description = "Use Alt+Right to move forward"
        if has_next_page:
            next_tooltip = "Next page (Alt+Right)"
            next_description = "Use Alt+Right to continue to the next page"
        elif not has_next_step:
            next_tooltip = "Finish setup (Alt+Right)"
            next_description = "Use Alt+Right to complete setup"

        for widget, tooltip, description in (
            (self._back_button, back_tooltip, back_description),
            (self._next_button, next_tooltip, next_description),
        ):
            if hasattr(widget, "set_tooltip_text"):
                try:
                    widget.set_tooltip_text(tooltip)
                except Exception:  # pragma: no cover - GTK stubs
                    pass
            if hasattr(widget, "set_accessible_description"):
                try:
                    widget.set_accessible_description(description)
                except Exception:  # pragma: no cover - GTK stubs
                    pass

    def _update_step_status(self) -> None:
        total_steps = len(self._steps)
        if total_steps == 0:
            self._step_status_label.set_text("")
            self._step_progress_bar.set_fraction(0.0)
            return

        current_step_number = self._current_index + 1
        step = self._steps[self._current_index]
        total_pages = self._get_total_pages()
        current_page = self._get_current_page_index()
        status = f"{step.name}: Step {current_step_number} of {total_steps}"
        if total_pages > 1:
            if current_page == 0:
                page_label = "Intro"
            elif current_page == 1 and total_pages == 2:
                page_label = "Configuration"
            else:
                page_label = f"Page {current_page + 1}"
            status = (
                f"{status} — {page_label} (Page {current_page + 1} of {total_pages})"
            )
        self._step_status_label.set_text(status)
        self._step_progress_bar.set_fraction(current_step_number / total_steps)

    def _on_window_key_pressed(
        self, _controller: Gtk.EventControllerKey | None, keyval: int, _keycode: int, state: int
    ) -> bool:
        alt_mask = getattr(Gdk.ModifierType, "ALT_MASK", getattr(Gdk.ModifierType, "MOD1_MASK", 0))
        if not state & alt_mask:
            return False

        handled = False
        if keyval in {
            getattr(Gdk, "KEY_Left", 0),
            getattr(Gdk, "KEY_KP_Left", 0),
        }:
            self._on_back_clicked(None)
            handled = True
        elif keyval in {
            getattr(Gdk, "KEY_Right", 0),
            getattr(Gdk, "KEY_KP_Right", 0),
        }:
            self._on_next_clicked(None)
            handled = True

        return handled

    def _on_back_clicked(self, *_: object) -> None:
        if self._retreat_subpage():
            return
        if self._current_index > 0:
            self._go_to_step(self._current_index - 1)

    def _on_step_row_activated(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        """Jump directly to a step when its row is activated in the sidebar."""
        try:
            index = self._step_rows.index(row)
        except ValueError:
            return
        self._go_to_step(index)
        if self._step_list:
            self._step_list.select_row(row)

    def _on_next_clicked(self, *_: object) -> None:
        if not self._steps:
            return
        if self._advance_subpage():
            return
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

        # Mark current step as completed and refresh sidebar visuals
        self._completed_steps.add(self._current_index)
        self._build_steps_sidebar()
        if self._step_list and 0 <= self._current_index < len(self._step_rows):
            self._step_list.select_row(self._step_rows[self._current_index])

        if self._current_index == len(self._steps) - 1:
            try:
                registration = self.controller.register_user()
            except Exception as exc:  # pragma: no cover - defensive
                self.display_error(exc)
                self._on_error(exc)
                return
            admin_message = "Administrator account created."
            final_message = message or "Setup complete."
            if admin_message:
                final_message = f"{final_message} {admin_message}".strip()
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
        state = self._collect_database_state(strict=True)

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
        self._populate_database_form(self.controller.state.database)
        return "Database connection saved."

    def _create_privileged_credentials_dialog(
        self,
        *,
        existing: Optional[Tuple[Optional[str], Optional[str]]],
        error: BootstrapError,
    ) -> Tuple[Gtk.Dialog, Gtk.Entry, Gtk.Entry, Gtk.Button, Gtk.Button]:
        dialog = Gtk.Dialog(transient_for=self, modal=True)
        if hasattr(dialog, "set_title"):
            dialog.set_title("Privileged PostgreSQL credentials required")
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
        if hasattr(password_entry, "set_activates_default"):
            password_entry.set_activates_default(True)
        grid.attach(password_entry, 1, 1, 1, 1)

        content.append(grid)

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        action_box.set_halign(Gtk.Align.END)
        cancel_button = Gtk.Button(label="Cancel")

        def _emit_cancel(_: Gtk.Widget) -> None:
            dialog.response(Gtk.ResponseType.CANCEL)

        cancel_button.connect("clicked", _emit_cancel)
        action_box.append(cancel_button)

        apply_button = Gtk.Button(label="Apply")

        def _emit_apply(_: Gtk.Widget) -> None:
            dialog.response(Gtk.ResponseType.OK)

        apply_button.connect("clicked", _emit_apply)
        try:
            apply_button.connect("activate", _emit_apply)
        except TypeError:
            # Some Gtk versions may not expose the "activate" signal on Gtk.Button.
            pass
        if hasattr(apply_button, "set_receives_default"):
            apply_button.set_receives_default(True)
        if hasattr(dialog, "set_default_widget"):
            dialog.set_default_widget(apply_button)
        if hasattr(dialog, "set_default_response"):
            dialog.set_default_response(Gtk.ResponseType.OK)
        action_box.append(apply_button)

        content.append(action_box)
        dialog.set_child(content)

        return dialog, username_entry, password_entry, apply_button, cancel_button

    def _prompt_privileged_credentials(
        self,
        *,
        existing: Optional[Tuple[Optional[str], Optional[str]]],
        error: BootstrapError,
    ) -> Optional[Tuple[Optional[str], Optional[str]]]:
        (
            dialog,
            username_entry,
            password_entry,
            _apply_button,
            _cancel_button,
        ) = self._create_privileged_credentials_dialog(existing=existing, error=error)

        if hasattr(username_entry, "grab_focus"):
            username_entry.grab_focus()

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

    def _apply_job_scheduling(self) -> str:
        enabled_widget = self._job_widgets.get("enabled")
        if not isinstance(enabled_widget, Gtk.CheckButton):  # pragma: no cover - defensive
            raise RuntimeError("Job scheduling toggle is missing")
        enabled = enabled_widget.get_active()

        job_store_entry = self._job_widgets.get("job_store_url")
        max_workers_entry = self._job_widgets.get("max_workers")
        timezone_entry = self._job_widgets.get("timezone")
        queue_size_entry = self._job_widgets.get("queue_size")
        retry_max_entry = self._job_widgets.get("retry_max_attempts")
        backoff_entry = self._job_widgets.get("retry_backoff_seconds")
        jitter_entry = self._job_widgets.get("retry_jitter_seconds")
        multiplier_entry = self._job_widgets.get("retry_backoff_multiplier")

        if not all(
            isinstance(widget, Gtk.Entry)
            for widget in (
                job_store_entry,
                max_workers_entry,
                timezone_entry,
                queue_size_entry,
                retry_max_entry,
                backoff_entry,
                jitter_entry,
                multiplier_entry,
            )
        ):
            raise RuntimeError("Job scheduling inputs are not configured correctly")

        assert isinstance(job_store_entry, Gtk.Entry)
        job_store_url = job_store_entry.get_text().strip() or None
        assert isinstance(max_workers_entry, Gtk.Entry)
        max_workers = self._parse_optional_int(max_workers_entry, "Max workers")
        assert isinstance(timezone_entry, Gtk.Entry)
        timezone = timezone_entry.get_text().strip() or None
        assert isinstance(queue_size_entry, Gtk.Entry)
        queue_size = self._parse_optional_int(queue_size_entry, "Queue size")

        assert isinstance(retry_max_entry, Gtk.Entry)
        max_attempts = self._parse_required_int(retry_max_entry, "Max attempts")
        assert isinstance(backoff_entry, Gtk.Entry)
        backoff_seconds = self._parse_required_float(backoff_entry, "Backoff seconds")
        assert isinstance(jitter_entry, Gtk.Entry)
        jitter_seconds = self._parse_required_float(jitter_entry, "Jitter seconds")
        assert isinstance(multiplier_entry, Gtk.Entry)
        backoff_multiplier = self._parse_required_float(multiplier_entry, "Backoff multiplier")

        if not enabled:
            job_store_url = None
            max_workers = None
            timezone = None
            queue_size = None

        state = JobSchedulingState(
            enabled=enabled,
            job_store_url=job_store_url,
            max_workers=max_workers,
            retry_policy=RetryPolicyState(
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                jitter_seconds=jitter_seconds,
                backoff_multiplier=backoff_multiplier,
            ),
            timezone=timezone,
            queue_size=queue_size,
        )

        self.controller.apply_job_scheduling(state)
        return "Job scheduling settings saved."

    def _apply_message_bus(self) -> str:
        backend_widget = self._message_widgets.get("backend")
        redis_entry = self._message_widgets.get("redis_url")
        stream_entry = self._message_widgets.get("stream_prefix")

        if not isinstance(backend_widget, Gtk.ComboBoxText) or not isinstance(
            redis_entry, Gtk.Entry
        ) or not isinstance(stream_entry, Gtk.Entry):
            raise RuntimeError("Message bus widgets are not configured correctly")

        backend = backend_widget.get_active_id() or ""
        backend = backend.strip().lower()
        if backend not in {"in_memory", "redis"}:
            raise ValueError("Backend must be 'in_memory' or 'redis'")

        redis_url = redis_entry.get_text().strip() or None
        stream_prefix = stream_entry.get_text().strip() or None

        if backend != "redis":
            backend = "in_memory"
            redis_url = None
            stream_prefix = None

        state = MessageBusState(
            backend=backend,
            redis_url=redis_url,
            stream_prefix=stream_prefix,
        )

        self.controller.apply_message_bus(state)
        return "Message bus settings saved."

    def _apply_kv_store(self) -> str:
        reuse_widget = self._kv_widgets.get("reuse")
        url_widget = self._kv_widgets.get("url")
        if not isinstance(reuse_widget, Gtk.CheckButton) or not isinstance(url_widget, Gtk.Entry):
            raise RuntimeError("Key-value store widgets are not configured correctly")

        reuse = reuse_widget.get_active()
        url = url_widget.get_text().strip() or None
        if reuse:
            url = None

        state = KvStoreState(
            reuse_conversation_store=reuse,
            url=url,
        )

        self.controller.apply_kv_store_settings(state)
        return "Key-value store settings saved."

    def _apply_speech(self) -> str:
        tts_widget = self._speech_widgets.get("tts_enabled")
        stt_widget = self._speech_widgets.get("stt_enabled")
        default_tts_entry = self._speech_widgets.get("default_tts")
        default_stt_entry = self._speech_widgets.get("default_stt")
        elevenlabs_entry = self._speech_widgets.get("elevenlabs_key")
        openai_entry = self._speech_widgets.get("openai_key")
        google_entry = self._speech_widgets.get("google_credentials")

        if not all(
            isinstance(widget, Gtk.Entry)
            for widget in (
                default_tts_entry,
                default_stt_entry,
                elevenlabs_entry,
                openai_entry,
                google_entry,
            )
        ) or not isinstance(tts_widget, Gtk.CheckButton) or not isinstance(
            stt_widget, Gtk.CheckButton
        ):
            raise RuntimeError("Speech widgets are not configured correctly")

        assert isinstance(tts_widget, Gtk.CheckButton)
        assert isinstance(stt_widget, Gtk.CheckButton)
        assert isinstance(default_tts_entry, Gtk.Entry)
        assert isinstance(default_stt_entry, Gtk.Entry)
        assert isinstance(elevenlabs_entry, Gtk.Entry)
        assert isinstance(openai_entry, Gtk.Entry)
        assert isinstance(google_entry, Gtk.Entry)

        state = SpeechState(
            tts_enabled=tts_widget.get_active(),
            stt_enabled=stt_widget.get_active(),
            default_tts_provider=default_tts_entry.get_text().strip() or None,
            default_stt_provider=default_stt_entry.get_text().strip() or None,
            elevenlabs_key=elevenlabs_entry.get_text().strip() or None,
            openai_key=openai_entry.get_text().strip() or None,
            google_credentials=google_entry.get_text().strip() or None,
        )

        self.controller.apply_speech_settings(state)
        return "Speech settings saved."

    def _apply_optional(self) -> str:
        tenant_entry = self._optional_widgets.get("tenant_id")
        retention_days_entry = self._optional_widgets.get("retention_days")
        retention_history_entry = self._optional_widgets.get("retention_history_limit")
        scheduler_timezone_entry = self._optional_widgets.get("scheduler_timezone")
        scheduler_queue_entry = self._optional_widgets.get("scheduler_queue_size")
        http_toggle = self._optional_widgets.get("http_auto_start")

        if not all(
            isinstance(widget, Gtk.Entry)
            for widget in (
                tenant_entry,
                retention_days_entry,
                retention_history_entry,
                scheduler_timezone_entry,
                scheduler_queue_entry,
            )
        ) or not isinstance(http_toggle, Gtk.CheckButton):
            raise RuntimeError("Organization settings widgets are not configured correctly")

        assert isinstance(tenant_entry, Gtk.Entry)
        assert isinstance(retention_days_entry, Gtk.Entry)
        assert isinstance(retention_history_entry, Gtk.Entry)
        assert isinstance(scheduler_timezone_entry, Gtk.Entry)
        assert isinstance(scheduler_queue_entry, Gtk.Entry)
        assert isinstance(http_toggle, Gtk.CheckButton)

        retention_days = self._parse_optional_int(retention_days_entry, "Conversation retention days")
        retention_history = self._parse_optional_int(
            retention_history_entry, "Conversation history limit"
        )
        scheduler_queue_size = self._parse_optional_int(
            scheduler_queue_entry, "Scheduler queue size"
        )

        state = OptionalState(
            tenant_id=tenant_entry.get_text().strip() or None,
            retention_days=retention_days,
            retention_history_limit=retention_history,
            scheduler_timezone=scheduler_timezone_entry.get_text().strip() or None,
            scheduler_queue_size=scheduler_queue_size,
            http_auto_start=http_toggle.get_active(),
        )

        self.controller.apply_optional_settings(state)
        return "Organization settings saved."

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

    def _apply_setup_type(self) -> str:
        message, _ = self._apply_setup_type_selection(None, update_status=True)
        return message

    def _apply_user(self) -> str:
        username = self._user_entries["username"].get_text().strip()
        email = self._user_entries["email"].get_text().strip()
        full_name = self._user_entries["full_name"].get_text().strip()
        domain_entry = self._user_entries["domain"].get_text().strip()
        date_of_birth_text = self._user_entries["date_of_birth"].get_text().strip()
        password = self._user_entries["password"].get_text()
        confirm = self._user_entries["confirm_password"].get_text()
        sudo_username = self._user_entries["sudo_username"].get_text().strip()
        sudo_password = self._user_entries["sudo_password"].get_text()
        confirm_sudo = self._user_entries["confirm_sudo_password"].get_text()

        if not username:
            raise ValueError("Username is required")
        if not password:
            raise ValueError("Password is required")
        if password != confirm:
            raise ValueError("Passwords do not match")
        if sudo_password != confirm_sudo:
            raise ValueError("Sudo passwords do not match")

        normalized_domain = domain_entry.strip()
        if normalized_domain.startswith("@"):
            normalized_domain = normalized_domain[1:]
        normalized_domain = normalized_domain.lower()

        dob_value = date_of_birth_text
        if date_of_birth_text:
            try:
                parsed = datetime.strptime(date_of_birth_text, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError("Date of birth must use YYYY-MM-DD format") from exc
            dob_value = parsed.strftime("%Y-%m-%d")

        display_name = full_name or username

        privileged_username, privileged_password = (self._privileged_credentials or (None, None))

        profile = AdminProfile(
            username=username,
            email=email,
            password=password,
            display_name=display_name,
            full_name=full_name,
            domain=normalized_domain,
            date_of_birth=dob_value,
            sudo_username=sudo_username,
            sudo_password=sudo_password,
            privileged_db_username=privileged_username or "",
            privileged_db_password=privileged_password or "",
        )

        self.controller.set_user_profile(profile)
        self._privileged_credentials = self.controller.get_privileged_credentials()
        self._sync_user_entries_from_state()
        self._refresh_downstream_defaults()
        return "Administrator profile saved."

    def _refresh_downstream_defaults(self) -> None:
        user_state = self.controller.state.user
        db_entry = self._database_entries.get("postgresql.user")
        suggested_db_user = user_state.username.strip() if user_state.username else ""
        if isinstance(db_entry, Gtk.Entry):
            current = db_entry.get_text().strip()
            if suggested_db_user and (not current or current == self._database_user_suggestion):
                db_entry.set_text(suggested_db_user)
            if suggested_db_user:
                self._database_user_suggestion = suggested_db_user
        tenant_entry = self._optional_widgets.get("tenant_id")
        suggested_tenant = user_state.domain.strip() if user_state.domain else ""
        if isinstance(tenant_entry, Gtk.Entry):
            current = tenant_entry.get_text().strip()
            if suggested_tenant and (not current or current == self._tenant_id_suggestion):
                tenant_entry.set_text(suggested_tenant)
            if suggested_tenant:
                self._tenant_id_suggestion = suggested_tenant

    def _create_debug_icon_widget(self) -> Gtk.Widget | None:
        icon_path = self._resolve_icon_path(self._DEBUG_ICON_FILENAME)
        image: Gtk.Image | None = None
        if icon_path is not None:
            try:
                image = Gtk.Image.new_from_file(icon_path)
            except Exception:  # pragma: no cover - GTK runtime fallback
                image = None

        if image is None:
            for icon_name in ("applications-system-symbolic", "applications-system"):
                factory = getattr(Gtk.Image, "new_from_icon_name", None)
                if callable(factory):
                    try:
                        image = factory(icon_name)
                    except TypeError:
                        try:
                            image = factory(icon_name, Gtk.IconSize.BUTTON)
                        except Exception:
                            image = None
                    except Exception:
                        image = None
                if image is not None:
                    break

        if image is None:
            return None

        pixel_setter = getattr(image, "set_pixel_size", None)
        if callable(pixel_setter):
            try:
                pixel_setter(24)
            except Exception:  # pragma: no cover - GTK stubs in tests
                pass
        return image

    def _resolve_icon_path(self, filename: str) -> str | None:
        try:
            root_dir = Path(__file__).resolve().parents[2]
        except Exception:  # pragma: no cover - path resolution fallback
            return None

        icon_path = root_dir / "Icons" / filename
        if icon_path.exists():
            self._register_icon_directory(icon_path.parent)
            return str(icon_path)
        return None

    def _register_icon_directory(self, directory: Path) -> None:
        try:
            display = Gdk.Display.get_default()
        except Exception:  # pragma: no cover - display unavailable in tests
            return

        if display is None:
            return

        theme_getter = getattr(Gtk.IconTheme, "get_for_display", None)
        if callable(theme_getter):
            try:
                theme = theme_getter(display)
            except Exception:  # pragma: no cover - GTK fallback
                theme = Gtk.IconTheme.get_default()
        else:
            theme = Gtk.IconTheme.get_default()

        if theme is None:
            return

        add_path = getattr(theme, "add_search_path", None)
        if callable(add_path):
            try:
                add_path(str(directory))
            except Exception:  # pragma: no cover - GTK fallback
                pass

    def _on_log_button_clicked(self, *_: object) -> None:
        window = self._log_window
        is_visible = False
        if window is not None:
            getter = getattr(window, "get_visible", None)
            if callable(getter):
                try:
                    is_visible = bool(getter())
                except Exception:
                    is_visible = False

        if window is None or not is_visible:
            self._ensure_log_window()
        else:
            self._close_log_window()

    def _set_log_button_active(self, active: bool) -> None:
        button = self._log_toggle_button
        if button is None:
            return

        add_css = getattr(button, "add_css_class", None)
        remove_css = getattr(button, "remove_css_class", None)

        if callable(add_css) and callable(remove_css):
            try:
                if active:
                    add_css("setup-wizard-debug-button-active")
                else:
                    remove_css("setup-wizard-debug-button-active")
            except Exception:  # pragma: no cover - GTK fallback
                return
        else:
            context = getattr(button, "get_style_context", None)
            if context is None:
                return
            add_class = getattr(context, "add_class", None)
            remove_class = getattr(context, "remove_class", None)
            if callable(add_class) and callable(remove_class):
                try:
                    if active:
                        add_class("setup-wizard-debug-button-active")
                    else:
                        remove_class("setup-wizard-debug-button-active")
                except Exception:
                    return

    def _disconnect_log_button(self) -> None:
        button = self._log_toggle_button
        handler_id = self._log_button_handler_id
        if button is not None and handler_id is not None:
            disconnect = getattr(button, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect(handler_id)
                except Exception:  # pragma: no cover - GTK fallback
                    pass
        self._set_log_button_active(False)
        self._log_button_handler_id = None
        self._log_toggle_button = None

    # -- log window helpers -------------------------------------------------

    def _ensure_log_window(self) -> SetupWizardLogWindow:
        """Create and present the setup log window if it is not already visible."""

        window = self._log_window
        if window is None:
            application = None
            if hasattr(self, "get_application"):
                try:
                    application = self.get_application()
                except Exception:  # pragma: no cover - defensive
                    application = None

            window = SetupWizardLogWindow(application=application, transient_for=self)
            if hasattr(window, "connect"):
                try:
                    window.connect("close-request", self._on_log_window_close_request)
                except (AttributeError, TypeError):
                    try:
                        window.connect("destroy", self._on_log_window_destroy)
                    except Exception:  # pragma: no cover - GTK stubs
                        pass

            handler = GTKUILogHandler(window.text_buffer, text_view=window.text_view)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)

            self._log_handler = handler
            self._log_target_loggers = []

            for logger in self._collect_setup_loggers():
                logger.addHandler(handler)
                effective_level = logger.getEffectiveLevel()
                if logger.level == logging.NOTSET or effective_level > logging.INFO:
                    logger.setLevel(logging.INFO)
                self._log_target_loggers.append(logger)

            self._log_window = window

        if hasattr(window, "present"):
            window.present()
        self._set_log_button_active(True)
        return window

    def _close_log_window(self) -> None:
        """Close the setup log window and detach logging handlers."""

        window = self._log_window
        self._detach_log_handler()
        self._log_window = None
        if window is None:
            return
        for attr in ("close", "destroy"):
            closer = getattr(window, attr, None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
                break
        self._set_log_button_active(False)

    def _detach_log_handler(self) -> None:
        handler = self._log_handler
        if handler is None:
            return
        for logger in list(self._log_target_loggers):
            try:
                logger.removeHandler(handler)
            except Exception:  # pragma: no cover - defensive cleanup
                continue
        self._log_target_loggers.clear()
        try:
            handler.close()
        finally:
            self._log_handler = None

    def _collect_setup_loggers(self) -> list[logging.Logger]:
        names = set(self._ADDITIONAL_LOGGER_NAMES)
        controller_module = self.controller.__class__.__module__
        names.add(controller_module)
        if "." in controller_module:
            names.add(controller_module.rsplit(".", 1)[0])
        return [logging.getLogger(name) for name in sorted(names)]

    def _on_log_window_close_request(self, *_: object) -> bool:
        self._detach_log_handler()
        self._log_window = None
        self._set_log_button_active(False)
        return False

    def _on_log_window_destroy(self, *_: object) -> None:
        self._detach_log_handler()
        self._log_window = None
        self._set_log_button_active(False)

    def _on_wizard_close_request(self, *_: object) -> bool:
        self._close_log_window()
        self._disconnect_log_button()
        return False

    def _on_wizard_destroy(self, *_: object) -> None:
        self._close_log_window()
        self._disconnect_log_button()

    def close(self) -> None:  # type: ignore[override]
        self._close_log_window()
        self._disconnect_log_button()
        try:
            super().close()
        except AttributeError:  # pragma: no cover - GTK3 fallback
            destroy = getattr(super(), "destroy", None)
            if callable(destroy):
                destroy()


class SetupWizardController(CoreSetupWizardController):  # type: ignore[misc]
    """Backwards-compatible import shim for legacy callers."""

    pass
