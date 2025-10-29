"""Interactive GTK front-end for the ATLAS setup workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gdk, Gtk

from ATLAS.setup import (
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
from GTKUI.Utils.styled_window import AtlasWindow
from modules.conversation_store.bootstrap import BootstrapError

Callback = Callable[[], None]
ErrorCallback = Callable[[BaseException], None]


@dataclass
class WizardStep:
    name: str
    widget: Gtk.Widget
    apply: Callable[[], str]


class SetupWizardWindow(AtlasWindow):
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
            title="ATLAS Setup Utility",
            default_size=(desired_width, desired_height),
        )
        self.set_application(application)
        self._on_success = on_success
        self._on_error = on_error

        self.controller = controller or CoreSetupWizardController(atlas=atlas)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        root.set_margin_top(18)
        root.set_margin_bottom(18)
        root.set_margin_start(18)
        root.set_margin_end(18)
        root.set_vexpand(True)
        root.set_hexpand(True)
        self.set_child(root)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)
        scroller.set_hexpand(True)
        scroller.set_propagate_natural_height(False)
        root.append(scroller)

        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        content.set_hexpand(True)
        content.set_vexpand(True)
        scroller.set_child(content)

        self._form_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._form_column.set_hexpand(True)
        self._form_column.set_vexpand(True)
        content.append(self._form_column)

        guidance_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        guidance_column.set_hexpand(True)
        guidance_column.set_vexpand(True)
        content.append(guidance_column)

        header = Gtk.Label(label="Complete the following steps to finish configuring ATLAS.")
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
        self._instructions_label.set_vexpand(True)
        self._instructions_label.set_visible(False)
        guidance_column.append(self._instructions_label)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._stack.set_vexpand(True)
        self._form_column.append(self._stack)

        self._switcher = Gtk.StackSwitcher()
        self._switcher.set_stack(self._stack)
        self._stack.connect("notify::visible-child", self._on_stack_visible_child_changed)
        self._form_column.append(self._switcher)

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

        self._instructions_by_widget: Dict[Gtk.Widget, str] = {}

        self._database_entries: Dict[str, Gtk.Entry] = {}
        self._provider_entries: Dict[str, Gtk.Entry] = {}
        self._provider_buffer: Optional[Gtk.TextBuffer] = None
        self._user_entries: Dict[str, Gtk.Entry] = {}
        self._kv_widgets: Dict[str, Gtk.Widget] = {}
        self._job_widgets: Dict[str, Gtk.Widget] = {}
        self._message_widgets: Dict[str, Gtk.Widget] = {}
        self._speech_widgets: Dict[str, Gtk.Widget] = {}
        self._optional_widgets: Dict[str, Gtk.Widget] = {}
        self._setup_persisted = False
        self._privileged_credentials: Optional[Tuple[Optional[str], Optional[str]]] = None
        self._entry_pixel_width: Optional[int] = None

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
                name="Job Scheduling",
                widget=self._build_job_scheduling_page(),
                apply=self._apply_job_scheduling,
            ),
            WizardStep(
                name="Message Bus",
                widget=self._build_message_bus_page(),
                apply=self._apply_message_bus,
            ),
            WizardStep(
                name="Key-Value Store",
                widget=self._build_kv_store_page(),
                apply=self._apply_kv_store,
            ),
            WizardStep(
                name="Providers",
                widget=self._build_provider_page(),
                apply=self._apply_providers,
            ),
            WizardStep(
                name="Speech",
                widget=self._build_speech_page(),
                apply=self._apply_speech,
            ),
            WizardStep(
                name="Optional",
                widget=self._build_optional_page(),
                apply=self._apply_optional,
            ),
            WizardStep(
                name="Administrator",
                widget=self._build_user_page(),
                apply=self._apply_user,
            ),
        ]

        for step in self._steps:
            self._stack.add_titled(step.widget, step.name.lower(), step.name)

    def _register_instructions(self, widget: Gtk.Widget, instructions: str) -> None:
        self._instructions_by_widget[widget] = instructions

    def _wrap_with_instructions(self, form: Gtk.Widget, instructions: str) -> Gtk.Widget:
        form.set_halign(Gtk.Align.START)
        form.set_hexpand(False)
        self._register_instructions(form, instructions)
        return form

    def _update_guidance_for_widget(self, widget: Gtk.Widget | None) -> None:
        if widget is None:
            self._instructions_label.set_text("")
            self._instructions_label.set_visible(False)
            return

        instructions = self._instructions_by_widget.get(widget, "")
        self._instructions_label.set_text(instructions)
        self._instructions_label.set_visible(bool(instructions))

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

        instructions = (
            "Enter the PostgreSQL connection details. If the conversation store "
            "already exists, the wizard will reuse it."
        )

        return self._wrap_with_instructions(grid, instructions)

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

        instructions = (
            "Set the default provider and model used for conversations, then supply any API"
            " keys the assistant needs. Add keys one per line in provider=key format."
        )

        self._register_instructions(box, instructions)
        return box

    def _build_kv_store_page(self) -> Gtk.Widget:
        state = self.controller.state.kv_store
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

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
            "Reuse the existing conversation store for key-value storage, or provide a dedicated"
            " DSN when you need an external database."
        )

        self._register_instructions(box, instructions)
        return box

    def _build_job_scheduling_page(self) -> Gtk.Widget:
        state = self.controller.state.job_scheduling
        retry = state.retry_policy
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        enable_toggle = Gtk.CheckButton(label="Enable durable job scheduling")
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
        row += 1
        self._job_widgets["retry_backoff_seconds"] = self._create_labeled_entry(
            grid, row, "Backoff seconds", self._format_float(retry.backoff_seconds)
        )
        row += 1
        self._job_widgets["retry_jitter_seconds"] = self._create_labeled_entry(
            grid, row, "Jitter seconds", self._format_float(retry.jitter_seconds)
        )
        row += 1
        self._job_widgets["retry_backoff_multiplier"] = self._create_labeled_entry(
            grid, row, "Backoff multiplier", self._format_float(retry.backoff_multiplier)
        )

        instructions = (
            "Enable durable job scheduling and configure the worker pool, timezone,"
            " and retry policy used for queued jobs."
        )

        self._register_instructions(grid, instructions)
        return grid

    def _build_message_bus_page(self) -> Gtk.Widget:
        state = self.controller.state.message_bus
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

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
            "Choose the message bus backend used for inter-process communication."
            " Select Redis to coordinate multiple workers or keep the in-memory backend for"
            " single-instance deployments."
        )

        self._register_instructions(grid, instructions)
        return grid

    def _build_speech_page(self) -> Gtk.Widget:
        state = self.controller.state.speech
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

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
            "Enable text-to-speech or speech-to-text features and provide credentials for the"
            " services you plan to use."
        )

        self._register_instructions(grid, instructions)
        return grid

    def _build_optional_page(self) -> Gtk.Widget:
        state = self.controller.state.optional
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        self._optional_widgets["tenant_id"] = self._create_labeled_entry(
            grid, 0, "Tenant ID", state.tenant_id or ""
        )
        self._optional_widgets["retention_days"] = self._create_labeled_entry(
            grid, 1, "Conversation retention days", self._optional_to_text(state.retention_days)
        )
        self._optional_widgets["retention_history_limit"] = self._create_labeled_entry(
            grid,
            2,
            "Conversation history limit",
            self._optional_to_text(state.retention_history_limit),
        )
        self._optional_widgets["scheduler_timezone"] = self._create_labeled_entry(
            grid, 3, "Scheduler timezone", state.scheduler_timezone or ""
        )
        self._optional_widgets["scheduler_queue_size"] = self._create_labeled_entry(
            grid,
            4,
            "Scheduler queue size",
            self._optional_to_text(state.scheduler_queue_size),
        )

        http_toggle = Gtk.CheckButton(label="Auto-start HTTP server")
        http_toggle.set_active(state.http_auto_start)
        self._optional_widgets["http_auto_start"] = http_toggle
        grid.attach(http_toggle, 0, 5, 2, 1)

        instructions = (
            "Configure optional tuning parameters such as tenant scoping, retention policies,"
            " and scheduler defaults."
        )

        self._register_instructions(grid, instructions)
        return grid

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

        instructions = "Create the administrator account that will be used to sign in after setup."

        return self._wrap_with_instructions(grid, instructions)

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
        entry.set_hexpand(False)
        entry.set_width_chars(25)
        entry.set_max_width_chars(25)
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

    def _go_to_step(self, index: int) -> None:
        self._current_index = max(0, min(index, len(self._steps) - 1))
        step = self._steps[self._current_index]
        self._stack.set_visible_child(step.widget)
        self._update_navigation()
        self._update_guidance_for_widget(step.widget)

    def _on_stack_visible_child_changed(
        self, stack: Gtk.Stack, _param_spec: object
    ) -> None:
        if not self._steps:
            return

        child = stack.get_visible_child()
        if child is None:
            return

        for index, step in enumerate(self._steps):
            if step.widget is child:
                break
        else:
            return

        if index == self._current_index:
            return

        self._current_index = index
        self._update_navigation()
        self._update_guidance_for_widget(child)

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
            raise RuntimeError("Optional settings widgets are not configured correctly")

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
        return "Optional settings saved."

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
