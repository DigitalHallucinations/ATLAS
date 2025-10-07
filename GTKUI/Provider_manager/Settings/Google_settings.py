from __future__ import annotations

"""GTK dialog for configuring Google Gemini provider defaults."""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css, create_box

logger = logging.getLogger(__name__)


class GoogleSettingsWindow(AtlasWindow):
    """Collect Google Gemini defaults such as model selection and safety filters."""

    _SAFETY_CATEGORIES: Tuple[Tuple[str, str], ...] = (
        ("Harassment / Abuse", "HARM_CATEGORY_HARASSMENT"),
        ("Hate Speech", "HARM_CATEGORY_HATE_SPEECH"),
        ("Sexually Explicit", "HARM_CATEGORY_SEXUALLY_EXPLICIT"),
        ("Dangerous Content", "HARM_CATEGORY_DANGEROUS_CONTENT"),
        ("Civic Integrity", "HARM_CATEGORY_CIVIC_INTEGRITY"),
        ("Derogatory", "HARM_CATEGORY_DEROGATORY"),
        ("Toxicity", "HARM_CATEGORY_TOXICITY"),
        ("Violence", "HARM_CATEGORY_VIOLENCE"),
        ("Sexual Content", "HARM_CATEGORY_SEXUAL"),
        ("Medical Advice", "HARM_CATEGORY_MEDICAL"),
        ("Self-Harm / Dangerous Acts", "HARM_CATEGORY_DANGEROUS"),
        ("Unspecified / Other", "HARM_CATEGORY_UNSPECIFIED"),
    )

    _SAFETY_CATEGORY_ALIASES: Dict[str, str] = {
        "HARM_CATEGORY_HARASSMENT_ABUSE": "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUAL_CONTENT": "HARM_CATEGORY_SEXUAL",
        "HARM_CATEGORY_SELF_HARM": "HARM_CATEGORY_DANGEROUS",
        "HARM_CATEGORY_DANGEROUS_ACTS": "HARM_CATEGORY_DANGEROUS",
    }

    _SAFETY_THRESHOLDS: Tuple[Tuple[str, str], ...] = (
        ("Allow all content", "BLOCK_NONE"),
        ("Block only high risk", "BLOCK_ONLY_HIGH"),
        ("Block medium and above", "BLOCK_MEDIUM_AND_ABOVE"),
        ("Block low and above", "BLOCK_LOW_AND_ABOVE"),
    )

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(
            title="Google Settings",
            modal=True,
            transient_for=parent_window,
            default_size=(420, 520),
        )
        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window

        self._api_key_visible = False
        self._defaults: Dict[str, Any] = {
            "model": "",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "candidate_count": 1,
            "max_output_tokens": 32000,
            "stop_sequences": [],
            "safety_settings": [],
            "response_mime_type": "",
            "system_instruction": "",
            "stream": True,
            "function_calling": True,
            "function_call_mode": "auto",
            "allowed_function_names": [],
            "cached_allowed_function_names": [],
            "response_schema": {},
            "seed": None,
            "response_logprobs": False,
        }
        self._allowed_function_cache: List[str] = []
        self._safety_controls: Dict[str, Tuple[Gtk.CheckButton, Gtk.ComboBoxText]] = {}
        self._available_models: List[str] = []
        self._function_call_mode_options: List[Tuple[str, str]] = [
            ("auto", "Automatic"),
            ("any", "Allow any function"),
            ("none", "Disable function calls"),
            ("require", "Require allowed functions"),
        ]
        self._function_call_mode_label_map = {
            label: value for value, label in self._function_call_mode_options
        }
        self._function_call_mode_index_map = {
            value: index for index, (value, _label) in enumerate(self._function_call_mode_options)
        }

        notebook = Gtk.Notebook()
        notebook.get_style_context().add_class("sidebar-notebook")
        notebook.set_hexpand(True)
        notebook.set_vexpand(True)
        self.notebook = notebook

        credentials_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=12,
        )
        notebook.append_page(credentials_box, Gtk.Label(label="Credentials"))

        credentials_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        credentials_box.append(credentials_grid)

        row = 0
        api_label = Gtk.Label(label="Google API Key:")
        api_label.set_xalign(0.0)
        credentials_grid.attach(api_label, 0, row, 1, 1)

        api_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        credentials_grid.attach(api_box, 1, row, 1, 1)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_hexpand(True)
        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(False)
        if hasattr(self.api_key_entry, "set_invisible_char"):
            self.api_key_entry.set_invisible_char("•")
        self.api_key_entry.set_placeholder_text("Enter your Google API key")
        self.api_key_entry.set_tooltip_text(
            "Provide the Google Generative Language API key. It will be stored securely."
        )
        api_box.append(self.api_key_entry)

        self.api_key_toggle = Gtk.Button(label="Show")
        self.api_key_toggle.set_tooltip_text("Toggle visibility of the API key field.")
        self.api_key_toggle.connect("clicked", self._on_api_key_toggle_clicked)
        api_box.append(self.api_key_toggle)

        self.save_key_button = Gtk.Button(label="Save Key")
        self.save_key_button.set_tooltip_text("Persist the API key and refresh provider access.")
        self.save_key_button.connect("clicked", self.on_save_api_key_clicked)
        api_box.append(self.save_key_button)

        row += 1
        self.api_key_status_label = Gtk.Label(label="")
        self.api_key_status_label.set_xalign(0.0)
        self.api_key_status_label.set_hexpand(True)
        credentials_grid.attach(self.api_key_status_label, 0, row, 2, 1)

        general_scroller = Gtk.ScrolledWindow()
        general_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        general_scroller.set_hexpand(True)
        general_scroller.set_vexpand(True)
        notebook.append_page(general_scroller, Gtk.Label(label="General"))

        general_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=12,
        )
        general_scroller.set_child(general_box)

        general_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        general_box.append(general_grid)

        general_row = 0
        model_label = Gtk.Label(label="Default model:")
        model_label.set_xalign(0.0)
        general_grid.attach(model_label, 0, general_row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        self.model_combo.set_tooltip_text("Pick the default Gemini model used for completions.")
        general_grid.attach(self.model_combo, 1, general_row, 1, 1)

        general_row += 1
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_xalign(0.0)
        general_grid.attach(temp_label, 0, general_row, 1, 1)

        self.temperature_adjustment = Gtk.Adjustment(
            lower=0.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.temperature_spin = Gtk.SpinButton(
            adjustment=self.temperature_adjustment, digits=2
        )
        self.temperature_spin.set_increments(0.05, 0.1)
        self.temperature_spin.set_tooltip_text(
            "Higher values introduce more randomness. Range 0.0 – 2.0."
        )
        general_grid.attach(self.temperature_spin, 1, general_row, 1, 1)

        general_row += 1
        top_p_label = Gtk.Label(label="Top-p:")
        top_p_label.set_xalign(0.0)
        general_grid.attach(top_p_label, 0, general_row, 1, 1)

        self.top_p_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.05, value=1.0
        )
        self.top_p_spin = Gtk.SpinButton(adjustment=self.top_p_adjustment, digits=2)
        self.top_p_spin.set_increments(0.01, 0.05)
        self.top_p_spin.set_tooltip_text(
            "Nucleus sampling parameter. Lower values focus on the most likely tokens."
        )
        general_grid.attach(self.top_p_spin, 1, general_row, 1, 1)

        general_row += 1
        top_k_label = Gtk.Label(label="Top-k override:")
        top_k_label.set_xalign(0.0)
        general_grid.attach(top_k_label, 0, general_row, 1, 1)

        top_k_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        general_grid.attach(top_k_box, 1, general_row, 1, 1)

        self.top_k_toggle = Gtk.CheckButton(label="Enable")
        self.top_k_toggle.set_tooltip_text(
            "Toggle to limit sampling to the top-k most likely tokens."
        )
        self.top_k_toggle.connect("toggled", self._on_top_k_toggled)
        top_k_box.append(self.top_k_toggle)

        self.top_k_adjustment = Gtk.Adjustment(
            lower=1, upper=512, step_increment=1, page_increment=10, value=40
        )
        self.top_k_spin = Gtk.SpinButton(adjustment=self.top_k_adjustment, digits=0)
        self.top_k_spin.set_tooltip_text("Number of highest probability tokens to sample from.")
        self.top_k_spin.set_sensitive(False)
        top_k_box.append(self.top_k_spin)

        general_row += 1
        candidate_label = Gtk.Label(label="Candidate count:")
        candidate_label.set_xalign(0.0)
        general_grid.attach(candidate_label, 0, general_row, 1, 1)

        self.candidate_adjustment = Gtk.Adjustment(
            lower=1, upper=8, step_increment=1, page_increment=1, value=1
        )
        self.candidate_spin = Gtk.SpinButton(adjustment=self.candidate_adjustment, digits=0)
        self.candidate_spin.set_tooltip_text(
            "Number of candidates to request per prompt. Higher values increase cost."
        )
        general_grid.attach(self.candidate_spin, 1, general_row, 1, 1)

        general_row += 1
        seed_label = Gtk.Label(label="Deterministic seed:")
        seed_label.set_xalign(0.0)
        general_grid.attach(seed_label, 0, general_row, 1, 1)

        seed_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        general_grid.attach(seed_box, 1, general_row, 1, 1)

        self.seed_toggle = Gtk.CheckButton(label="Enable")
        self.seed_toggle.set_tooltip_text(
            "Provide a fixed random seed to reproduce Gemini outputs."
        )
        self.seed_toggle.connect("toggled", self._on_seed_toggled)
        seed_box.append(self.seed_toggle)

        self.seed_adjustment = Gtk.Adjustment(
            lower=0,
            upper=2_147_483_647,
            step_increment=1,
            page_increment=1000,
            value=0,
        )
        self.seed_spin = Gtk.SpinButton(adjustment=self.seed_adjustment, digits=0)
        self.seed_spin.set_sensitive(False)
        self.seed_spin.set_tooltip_text(
            "Use zero or leave disabled for non-deterministic behaviour."
        )
        seed_box.append(self.seed_spin)

        general_row += 1
        logprob_label = Gtk.Label(label="Log probabilities:")
        logprob_label.set_xalign(0.0)
        general_grid.attach(logprob_label, 0, general_row, 1, 1)

        self.response_logprobs_toggle = Gtk.CheckButton(
            label="Include token log probabilities"
        )
        self.response_logprobs_toggle.set_tooltip_text(
            "Request per-token log probabilities when generating responses."
        )
        general_grid.attach(self.response_logprobs_toggle, 1, general_row, 1, 1)

        general_row += 1
        self.stream_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.stream_toggle.set_halign(Gtk.Align.START)
        self.stream_toggle.set_tooltip_text(
            "Toggle to stream token updates during Gemini completions."
        )
        general_grid.attach(self.stream_toggle, 0, general_row, 2, 1)

        general_row += 1
        self.function_call_toggle = Gtk.CheckButton(label="Enable tool calling")
        self.function_call_toggle.set_halign(Gtk.Align.START)
        self.function_call_toggle.set_tooltip_text(
            "Allow Gemini to invoke persona tools and function calls by default."
        )
        self.function_call_toggle.connect("toggled", self._on_function_call_toggled)
        general_grid.attach(self.function_call_toggle, 0, general_row, 2, 1)

        general_row += 1
        function_mode_label = Gtk.Label(label="Function call mode:")
        function_mode_label.set_xalign(0.0)
        general_grid.attach(function_mode_label, 0, general_row, 1, 1)

        self.function_call_mode_combo = Gtk.ComboBoxText()
        for _value, label in self._function_call_mode_options:
            self.function_call_mode_combo.append_text(label)
        self.function_call_mode_combo.set_tooltip_text(
            "Select how Gemini should decide when to call tools."
        )
        general_grid.attach(self.function_call_mode_combo, 1, general_row, 1, 1)

        general_row += 1
        allowed_functions_label = Gtk.Label(label="Allowed function names:")
        allowed_functions_label.set_xalign(0.0)
        general_grid.attach(allowed_functions_label, 0, general_row, 1, 1)

        self.allowed_functions_entry = Gtk.Entry()
        self.allowed_functions_entry.set_hexpand(True)
        self.allowed_functions_entry.set_placeholder_text("Comma-separated whitelist of functions")
        self.allowed_functions_entry.set_tooltip_text(
            "Optional whitelist restricting Gemini function calls to specific tools."
        )
        general_grid.attach(self.allowed_functions_entry, 1, general_row, 1, 1)

        general_row += 1
        max_output_label = Gtk.Label(label="Max output tokens:")
        max_output_label.set_xalign(0.0)
        general_grid.attach(max_output_label, 0, general_row, 1, 1)

        self.max_output_tokens_adjustment = Gtk.Adjustment(
            lower=0,
            upper=128000,
            step_increment=128,
            page_increment=1024,
            value=32000,
        )
        self.max_output_tokens_spin = Gtk.SpinButton(
            adjustment=self.max_output_tokens_adjustment,
            digits=0,
        )
        self.max_output_tokens_spin.set_hexpand(True)
        self.max_output_tokens_spin.set_tooltip_text(
            "Optional limit for response tokens. Set to 0 to remove the cap."
        )
        general_grid.attach(self.max_output_tokens_spin, 1, general_row, 1, 1)

        general_row += 1
        stop_label = Gtk.Label(label="Stop sequences:")
        stop_label.set_xalign(0.0)
        general_grid.attach(stop_label, 0, general_row, 1, 1)

        self.stop_sequences_entry = Gtk.Entry()
        self.stop_sequences_entry.set_hexpand(True)
        self.stop_sequences_entry.set_placeholder_text("Comma-separated tokens to stop generation")
        self.stop_sequences_entry.set_tooltip_text(
            "Provide comma separated stop strings. Generation halts when a match is produced."
        )
        general_grid.attach(self.stop_sequences_entry, 1, general_row, 1, 1)

        general_row += 1
        mime_label = Gtk.Label(label="Response MIME type:")
        mime_label.set_xalign(0.0)
        general_grid.attach(mime_label, 0, general_row, 1, 1)

        self.response_mime_entry = Gtk.Entry()
        self.response_mime_entry.set_hexpand(True)
        self.response_mime_entry.set_placeholder_text("e.g. text/plain or application/json")
        self.response_mime_entry.set_tooltip_text(
            "Optional MIME type for responses when using multimodal Gemini features."
        )
        general_grid.attach(self.response_mime_entry, 1, general_row, 1, 1)

        general_row += 1
        system_label = Gtk.Label(label="System instruction:")
        system_label.set_xalign(0.0)
        general_grid.attach(system_label, 0, general_row, 1, 1)

        system_scroller = Gtk.ScrolledWindow()
        system_scroller.set_hexpand(True)
        system_scroller.set_vexpand(True)
        system_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.system_instruction_view = Gtk.TextView()
        self.system_instruction_view.set_hexpand(True)
        self.system_instruction_view.set_vexpand(True)
        if hasattr(self.system_instruction_view, "set_wrap_mode") and hasattr(Gtk, "WrapMode"):
            try:
                self.system_instruction_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            except Exception:
                pass
        system_scroller.set_child(self.system_instruction_view)
        general_grid.attach(system_scroller, 1, general_row, 1, 1)

        general_row += 1
        schema_label = Gtk.Label(label="Response schema (JSON):")
        schema_label.set_xalign(0.0)
        general_grid.attach(schema_label, 0, general_row, 1, 1)

        schema_scroller = Gtk.ScrolledWindow()
        schema_scroller.set_hexpand(True)
        schema_scroller.set_vexpand(True)
        schema_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.response_schema_view = Gtk.TextView()
        self.response_schema_view.set_hexpand(True)
        self.response_schema_view.set_vexpand(True)
        self.response_schema_view.set_tooltip_text(
            "Optional JSON schema enforcing Gemini responses. Leave blank to disable."
        )
        if hasattr(self.response_schema_view, "set_wrap_mode") and hasattr(Gtk, "WrapMode"):
            try:
                self.response_schema_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            except Exception:
                pass
        schema_scroller.set_child(self.response_schema_view)
        general_grid.attach(schema_scroller, 1, general_row, 1, 1)

        safety_scroller = Gtk.ScrolledWindow()
        safety_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        safety_scroller.set_hexpand(True)
        safety_scroller.set_vexpand(True)
        notebook.append_page(safety_scroller, Gtk.Label(label="Safety"))

        safety_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=12,
        )
        safety_scroller.set_child(safety_box)

        safety_frame = Gtk.Frame(label="Safety filters")
        safety_frame.set_tooltip_text(
            "Configure safety filters to block responses from categories such as "
            "Harassment, Hate Speech, Derogatory, Self-Harm, and more."
        )
        safety_box.append(safety_frame)

        safety_filters_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            margin=12,
        )
        safety_frame.set_child(safety_filters_box)

        for friendly, category in self._SAFETY_CATEGORIES:
            row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            toggle = Gtk.CheckButton(label=friendly)
            toggle.set_tooltip_text(
                "Enable blocking for this harm category using the selected threshold."
            )
            toggle.connect("toggled", self._on_safety_toggle_toggled, category)
            row_box.append(toggle)

            combo = Gtk.ComboBoxText()
            combo.set_hexpand(True)
            for caption, value in self._SAFETY_THRESHOLDS:
                combo.append_text(caption)
            combo.set_active(2)  # BLOCK_MEDIUM_AND_ABOVE
            combo.set_sensitive(False)
            combo.set_tooltip_text("Select the severity threshold to block for this category.")
            row_box.append(combo)

            safety_filters_box.append(row_box)
            self._safety_controls[category] = (toggle, combo)

        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        action_box.set_halign(Gtk.Align.END)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.set_tooltip_text("Close without saving changes.")
        cancel_button.connect("clicked", lambda *_args: self.close())
        action_box.append(cancel_button)

        save_button = Gtk.Button(label="Save Settings")
        save_button.set_tooltip_text("Persist defaults and refresh the Google provider.")
        save_button.connect("clicked", self.on_save_clicked)
        action_box.append(save_button)

        def _assign_child(widget: Gtk.Widget) -> None:
            setter = getattr(self, "set_child", None)
            if callable(setter):
                setter(widget)
            else:
                try:
                    setattr(self, "child", widget)
                except Exception:
                    pass

        set_action_widget = getattr(notebook, "set_action_widget", None)
        pack_type = getattr(Gtk, "PackType", None)
        if callable(set_action_widget) and pack_type is not None and hasattr(pack_type, "END"):
            try:
                set_action_widget(action_box, pack_type.END)
                _assign_child(notebook)
            except Exception:
                fallback_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=0, margin=0)
                fallback_box.append(notebook)
                fallback_box.append(action_box)
                _assign_child(fallback_box)
        else:
            fallback_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=0, margin=0)
            fallback_box.append(notebook)
            fallback_box.append(action_box)
            _assign_child(fallback_box)

        self._load_models()
        self._load_settings()
        self._refresh_api_key_status()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        models: Sequence[str] = []
        fetcher = getattr(self.ATLAS, "get_models_for_provider", None)
        if callable(fetcher):
            try:
                models = fetcher("Google") or []
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Unable to load Google models: %s", exc, exc_info=True)

        self.model_combo.remove_all()
        self._available_models = []
        seen: List[str] = []
        for name in models:
            if isinstance(name, str) and name.strip():
                cleaned = name.strip()
                if cleaned not in seen:
                    self.model_combo.append_text(cleaned)
                    seen.append(cleaned)
        self._available_models = list(seen)

        if seen:
            self.model_combo.set_active(0)
        else:
            self.model_combo.append_text("gemini-1.5-pro-latest")
            self.model_combo.set_active(0)
            self._available_models = ["gemini-1.5-pro-latest"]

    def _load_settings(self) -> None:
        settings: Dict[str, Any] = {}
        getter = getattr(self.ATLAS, "get_google_llm_settings", None)
        if callable(getter):
            try:
                settings = getter() or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Unable to read Google defaults: %s", exc, exc_info=True)

        if isinstance(settings, dict):
            self._defaults.update(settings)

        model = str(self._defaults.get("model") or "").strip()
        if model:
            self._ensure_model_visible(model)

        temperature = self._defaults.get("temperature", 0.0)
        if isinstance(temperature, (int, float)):
            self.temperature_spin.set_value(float(temperature))

        top_p = self._defaults.get("top_p", 1.0)
        if isinstance(top_p, (int, float)):
            self.top_p_spin.set_value(float(top_p))

        top_k = self._defaults.get("top_k")
        if isinstance(top_k, (int, float)) and top_k > 0:
            self.top_k_toggle.set_active(True)
            self.top_k_spin.set_value(int(top_k))
        else:
            self.top_k_toggle.set_active(False)
            self.top_k_spin.set_value(40)

        candidate_count = self._defaults.get("candidate_count", 1)
        if isinstance(candidate_count, (int, float)) and candidate_count > 0:
            self.candidate_spin.set_value(int(candidate_count))

        seed_value = self._defaults.get("seed")
        if isinstance(seed_value, (int, float)) and seed_value >= 0:
            self.seed_toggle.set_active(True)
            self.seed_spin.set_value(int(seed_value))
        else:
            self.seed_toggle.set_active(False)
            self.seed_spin.set_value(0)

        self.response_logprobs_toggle.set_active(
            bool(self._defaults.get("response_logprobs", False))
        )

        self.stream_toggle.set_active(bool(self._defaults.get("stream", True)))
        self.function_call_toggle.set_active(
            bool(self._defaults.get("function_calling", True))
        )

        mode_value = str(self._defaults.get("function_call_mode") or "auto").strip().lower()
        mode_index = self._function_call_mode_index_map.get(mode_value, 0)
        try:
            self.function_call_mode_combo.set_active(mode_index)
        except Exception:
            pass

        allowed_names_value = self._defaults.get("allowed_function_names", [])
        cached_names_value = self._defaults.get("cached_allowed_function_names", [])
        if (
            (not allowed_names_value)
            and isinstance(cached_names_value, Sequence)
            and cached_names_value
        ):
            allowed_names_value = cached_names_value
        if isinstance(allowed_names_value, str):
            allowed_tokens = [
                token.strip()
                for token in allowed_names_value.split(",")
                if token.strip()
            ]
        elif isinstance(allowed_names_value, Sequence):
            allowed_tokens = [
                str(token).strip()
                for token in allowed_names_value
                if isinstance(token, str) and str(token).strip()
            ]
        else:
            allowed_tokens = []
        self._allowed_function_cache = list(allowed_tokens)
        try:
            self.allowed_functions_entry.set_text(", ".join(allowed_tokens))
        except Exception:
            pass

        self._update_tool_controls_sensitive()

        max_output_tokens = self._defaults.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            self.max_output_tokens_spin.set_value(int(max_output_tokens))
        else:
            self.max_output_tokens_spin.set_value(0)

        stop_sequences = self._defaults.get("stop_sequences", [])
        if isinstance(stop_sequences, str):
            tokens = [token.strip() for token in stop_sequences.split(",") if token.strip()]
        elif isinstance(stop_sequences, Sequence):
            tokens = [str(token).strip() for token in stop_sequences if str(token).strip()]
        else:
            tokens = []
        self.stop_sequences_entry.set_text(", ".join(tokens))

        response_mime_type = str(self._defaults.get("response_mime_type") or "").strip()
        self.response_mime_entry.set_text(response_mime_type)

        system_instruction = str(self._defaults.get("system_instruction") or "")
        buffer = self.system_instruction_view.get_buffer()
        if buffer is not None:
            try:
                buffer.set_text(system_instruction)
            except Exception:
                pass

        schema_buffer = self.response_schema_view.get_buffer()
        if schema_buffer is not None:
            schema_value = self._defaults.get("response_schema")
            if isinstance(schema_value, str):
                text_value = schema_value.strip()
            elif isinstance(schema_value, dict) and schema_value:
                try:
                    text_value = json.dumps(schema_value, indent=2, sort_keys=True)
                except (TypeError, ValueError):
                    text_value = ""
            else:
                text_value = ""
            try:
                schema_buffer.set_text(text_value)
            except Exception:
                pass

        safety_settings = self._defaults.get("safety_settings", [])
        if isinstance(safety_settings, Sequence):
            for category, controls in self._safety_controls.items():
                toggle, combo = controls
                entry = self._find_safety_entry(safety_settings, category)
                if entry:
                    toggle.set_active(True)
                    threshold = entry.get("threshold") or entry.get("harmBlockThreshold")
                    self._select_safety_threshold(combo, str(threshold or ""))
                else:
                    toggle.set_active(False)

    def _ensure_model_visible(self, model: str) -> None:
        if not model:
            return
        if model not in self._available_models:
            self.model_combo.append_text(model)
            self._available_models.append(model)
        try:
            index = self._available_models.index(model)
        except ValueError:
            index = 0
        try:
            self.model_combo.set_active(index)
        except Exception:
            pass

    def _find_safety_entry(self, entries: Sequence[Any], category: str) -> Optional[Dict[str, str]]:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            current = (
                entry.get("category")
                or entry.get("harmCategory")
                or entry.get("name")
            )
            canonical = self._canonicalize_safety_category(current)
            if canonical != category:
                continue
            threshold = (
                entry.get("threshold")
                or entry.get("thresholdValue")
                or entry.get("harmBlockThreshold")
            )
            if not threshold:
                continue
            return {"category": category, "threshold": str(threshold)}
        return None

    @classmethod
    def _canonicalize_safety_category(cls, category: Optional[Any]) -> Optional[str]:
        if category is None:
            return None
        cleaned = str(category).strip()
        if not cleaned:
            return None
        normalized = cleaned.upper()
        canonical = cls._SAFETY_CATEGORY_ALIASES.get(normalized, normalized)
        return canonical or None

    # ------------------------------------------------------------------
    # API key helpers
    # ------------------------------------------------------------------

    def _refresh_api_key_status(self) -> None:
        status_text = "API key status unavailable."
        placeholder = "Enter your Google API key"
        tooltip = "Provide the Google Generative Language API key."

        atlas = getattr(self, "ATLAS", None)
        if atlas is not None and hasattr(atlas, "get_provider_api_key_status"):
            try:
                payload = atlas.get_provider_api_key_status("Google") or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Unable to load Google API key status: %s", exc, exc_info=True)
            else:
                has_key = bool(payload.get("has_key"))
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                hint = ""
                if isinstance(metadata, dict):
                    hint = metadata.get("hint") or ""
                if has_key:
                    suffix = f" ({hint})" if hint else ""
                    status_text = f"A Google API key is stored.{suffix}"
                    if hint:
                        placeholder = f"Saved key: {hint}"
                        tooltip = (
                            "A Google key is stored. Enter a new value to replace the existing credential."
                        )
                    else:
                        tooltip = "A Google key is stored. Enter a new value to replace it."
                else:
                    status_text = "No Google API key detected."

        self.api_key_entry.set_placeholder_text(placeholder)
        self.api_key_entry.set_tooltip_text(tooltip)
        self.api_key_status_label.set_label(status_text)

    def _on_api_key_toggle_clicked(self, _button: Gtk.Button) -> None:
        self._api_key_visible = not self._api_key_visible
        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(self._api_key_visible)
        label = "Hide" if self._api_key_visible else "Show"
        if hasattr(self.api_key_toggle, "set_label"):
            self.api_key_toggle.set_label(label)

    def _set_save_key_sensitive(self, enabled: bool) -> None:
        try:
            self.save_key_button.set_sensitive(enabled)
        except Exception:  # pragma: no cover - GTK stubs
            self.save_key_button.sensitive = enabled

    def on_save_api_key_clicked(self, *_args) -> None:
        key = self.api_key_entry.get_text().strip()
        if not key:
            self._show_message("Validation", "API key cannot be empty.", Gtk.MessageType.WARNING)
            return

        updater = getattr(self.ATLAS, "update_provider_api_key_in_background", None)
        if not callable(updater):
            self._show_message(
                "Unavailable",
                "This build does not support saving provider credentials programmatically.",
                Gtk.MessageType.ERROR,
            )
            return

        self._set_save_key_sensitive(False)

        def handle_success(result: Dict[str, Any]) -> None:
            GLib.idle_add(self._handle_api_key_result, result)

        def handle_error(exc: Exception) -> None:
            logger.error("Failed to save Google API key: %s", exc, exc_info=True)
            GLib.idle_add(
                self._handle_api_key_result,
                {"success": False, "error": str(exc)},
            )

        try:
            updater(
                "Google",
                key,
                on_success=handle_success,
                on_error=handle_error,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to schedule Google API key update: %s", exc, exc_info=True)
            self._set_save_key_sensitive(True)
            self._show_message("Error", str(exc), Gtk.MessageType.ERROR)

    def _handle_api_key_result(self, result: Dict[str, Any]) -> bool:
        self._set_save_key_sensitive(True)
        if isinstance(result, dict) and result.get("success"):
            message = result.get("message") or "Google API key saved."
            self.api_key_entry.set_text("")
            self._refresh_api_key_status()
            self._show_message("Success", message, Gtk.MessageType.INFO)
        else:
            detail = "Failed to save API key."
            if isinstance(result, dict):
                detail = result.get("error") or result.get("message") or detail
            self._show_message("Error", detail, Gtk.MessageType.ERROR)
        return False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_function_call_toggled(self, toggle: Gtk.CheckButton) -> None:
        enabled = toggle.get_active()
        if not enabled:
            cached_names = self._parse_allowed_function_names()
            if cached_names:
                self._allowed_function_cache = list(cached_names)
        else:
            if not self._allowed_function_cache:
                self._allowed_function_cache = self._parse_allowed_function_names()
            try:
                current_text = self.allowed_functions_entry.get_text()
            except Exception:
                current_text = ""
            if (current_text or "").strip() == "" and self._allowed_function_cache:
                try:
                    self.allowed_functions_entry.set_text(
                        ", ".join(self._allowed_function_cache)
                    )
                except Exception:
                    pass
        self._update_tool_controls_sensitive(enabled)

    def _on_top_k_toggled(self, toggle: Gtk.CheckButton) -> None:
        active = toggle.get_active()
        self.top_k_spin.set_sensitive(active)

    def _on_seed_toggled(self, toggle: Gtk.CheckButton) -> None:
        self.seed_spin.set_sensitive(toggle.get_active())

    def _on_safety_toggle_toggled(self, toggle: Gtk.CheckButton, category: str) -> None:
        combo = self._safety_controls.get(category, (None, None))[1]
        if combo is None:
            return
        combo.set_sensitive(toggle.get_active())

    def on_save_clicked(self, *_args) -> None:
        model = self.model_combo.get_active_text()
        if not model:
            self._show_message("Validation", "Select a default model before saving.", Gtk.MessageType.WARNING)
            return

        if self.top_k_toggle.get_active():
            top_k_value = self.top_k_spin.get_value_as_int()
            if top_k_value < 1:
                self._show_message(
                    "Validation",
                    "Top-k must be a positive integer when enabled.",
                    Gtk.MessageType.WARNING,
                )
                return
            top_k_payload: Optional[int | str] = top_k_value
        else:
            top_k_payload = ""

        functions_enabled = self.function_call_toggle.get_active()
        parsed_allowed_names = self._parse_allowed_function_names()
        if functions_enabled:
            self._allowed_function_cache = list(parsed_allowed_names)
        else:
            if parsed_allowed_names:
                self._allowed_function_cache = list(parsed_allowed_names)

        payload_allowed_names = (
            list(self._allowed_function_cache) if functions_enabled else []
        )

        if self.seed_toggle.get_active():
            seed_payload: int | str = self.seed_spin.get_value_as_int()
        else:
            seed_payload = ""

        payload = {
            "model": model,
            "temperature": round(self.temperature_spin.get_value(), 2),
            "top_p": round(self.top_p_spin.get_value(), 2),
            "top_k": top_k_payload,
            "candidate_count": self.candidate_spin.get_value_as_int(),
            "max_output_tokens": (
                self.max_output_tokens_spin.get_value_as_int()
                if self.max_output_tokens_spin.get_value_as_int() > 0
                else ""
            ),
            "stop_sequences": self._parse_stop_sequences(),
            "safety_settings": self._collect_safety_settings(),
            "response_mime_type": self._sanitize_response_mime_type(),
            "system_instruction": self._sanitize_system_instruction(),
            "stream": self.stream_toggle.get_active(),
            "function_calling": functions_enabled,
            "function_call_mode": self._get_selected_function_call_mode(),
            "allowed_function_names": payload_allowed_names,
            "cached_allowed_function_names": list(self._allowed_function_cache),
            "response_schema": None,
            "seed": seed_payload,
            "response_logprobs": self.response_logprobs_toggle.get_active(),
        }

        try:
            payload["response_schema"] = self._collect_response_schema()
        except ValueError as exc:
            self._show_message("Validation", str(exc), Gtk.MessageType.WARNING)
            return

        if not payload["function_calling"]:
            payload["function_call_mode"] = "none"

        schema = payload["response_schema"]
        if schema not in (None, "", {}):
            current_mime = payload.get("response_mime_type") or ""
            normalized_mime = current_mime.strip().lower()
            if normalized_mime and normalized_mime != "application/json":
                self._show_message(
                    "Validation",
                    (
                        "Response MIME type must be 'application/json' when a response "
                        "schema is provided."
                    ),
                    Gtk.MessageType.WARNING,
                )
                return
            payload["response_mime_type"] = "application/json"

        setter = getattr(self.ATLAS, "set_google_llm_settings", None)
        if not callable(setter):
            self._show_message(
                "Unavailable",
                "Provider manager does not expose Google settings persistence.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            result = setter(**payload)
        except Exception as exc:
            logger.error("Error saving Google settings: %s", exc, exc_info=True)
            self._show_message("Error", str(exc), Gtk.MessageType.ERROR)
            return

        if isinstance(result, dict) and result.get("success"):
            message = result.get("message") or "Google settings saved."
            self._show_message("Success", message, Gtk.MessageType.INFO)
            self._trigger_provider_refresh()
            self.close()
        else:
            detail = "Unable to save Google settings."
            if isinstance(result, dict):
                detail = result.get("error") or result.get("message") or detail
            self._show_message("Error", detail, Gtk.MessageType.ERROR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_tool_controls_sensitive(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            enabled = self.function_call_toggle.get_active()
        try:
            self.function_call_mode_combo.set_sensitive(bool(enabled))
        except Exception:
            pass
        try:
            self.allowed_functions_entry.set_sensitive(bool(enabled))
        except Exception:
            pass

    def _parse_stop_sequences(self) -> List[str]:
        text = self.stop_sequences_entry.get_text()
        if not text:
            return []
        tokens = []
        for chunk in text.replace("\n", ",").split(","):
            cleaned = chunk.strip()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def _get_selected_function_call_mode(self) -> str:
        try:
            label = self.function_call_mode_combo.get_active_text() or ""
        except Exception:
            label = ""
        return self._function_call_mode_label_map.get(label, "auto")

    def _parse_allowed_function_names(self) -> List[str]:
        try:
            text = self.allowed_functions_entry.get_text()
        except Exception:
            text = ""
        if not text:
            return []
        names: List[str] = []
        for chunk in text.replace("\n", ",").split(","):
            cleaned = chunk.strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
        return names

    def _collect_safety_settings(self) -> List[Dict[str, str]]:
        settings: List[Dict[str, str]] = []
        for category, controls in self._safety_controls.items():
            toggle, combo = controls
            if not toggle.get_active():
                continue
            index = self._get_combo_active_index(combo)
            if index < 0:
                index = 2
            threshold = self._SAFETY_THRESHOLDS[index][1]
            canonical = self._canonicalize_safety_category(category)
            if not canonical:
                continue
            settings.append({"category": canonical, "threshold": threshold})
        return settings

    def _sanitize_response_mime_type(self) -> str:
        text = ""
        try:
            text = self.response_mime_entry.get_text()
        except Exception:
            return ""
        return text.strip()

    def _sanitize_system_instruction(self) -> str:
        buffer = self.system_instruction_view.get_buffer()
        if buffer is None:
            return ""
        try:
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            text = buffer.get_text(start_iter, end_iter, True)
        except Exception:
            try:
                text = buffer.get_text(None, None, True)
            except Exception:
                text = ""
        return text.strip()

    def _collect_response_schema(self) -> Any:
        buffer = self.response_schema_view.get_buffer()
        if buffer is None:
            return ""
        try:
            start_iter = buffer.get_start_iter()
            end_iter = buffer.get_end_iter()
            text = buffer.get_text(start_iter, end_iter, True)
        except Exception:
            try:
                text = buffer.get_text(None, None, True)
            except Exception:
                text = ""
        cleaned = text.strip()
        if not cleaned:
            return ""
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response schema must be valid JSON: {exc.msg}.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Response schema must be a JSON object.")
        return parsed

    def _select_safety_threshold(self, combo: Gtk.ComboBoxText, threshold: str) -> None:
        normalized = threshold.strip().upper()
        for idx, (_label, value) in enumerate(self._SAFETY_THRESHOLDS):
            if value == normalized:
                combo.set_active(idx)
                break

    def _get_combo_active_index(self, combo: Gtk.ComboBoxText) -> int:
        getter = getattr(combo, "get_active", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if isinstance(value, int):
                return value
        for attr in ("active", "_active"):
            value = getattr(combo, attr, None)
            if isinstance(value, int):
                return value
        text_getter = getattr(combo, "get_active_text", None)
        if callable(text_getter):
            try:
                caption = text_getter()
            except Exception:
                caption = None
            if caption:
                for idx, (label, _value) in enumerate(self._SAFETY_THRESHOLDS):
                    if label == caption:
                        return idx
        return -1

    def _trigger_provider_refresh(self) -> None:
        refresher = getattr(self.ATLAS, "run_in_background", None)
        if not callable(refresher):
            return

        def handle_error(exc: Exception) -> None:
            logger.warning("Google provider refresh failed: %s", exc, exc_info=True)

        try:
            refresher(
                lambda: self.ATLAS.refresh_current_provider("Google"),
                on_error=handle_error,
                thread_name="refresh-google-provider",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to schedule Google provider refresh: %s", exc, exc_info=True)

    def _show_message(self, title: str, message: str, message_type: Gtk.MessageType) -> None:
        try:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=message_type,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
        except Exception:  # pragma: no cover - fallback for stub GTK
            logger.warning("GTK MessageDialog unavailable; logging message: %s", message)
            return

        self._style_dialog(dialog)

        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:  # pragma: no cover - GTK3 compatibility
            dialog.props.secondary_text = message

        dialog.connect("response", lambda dlg, _resp: dlg.destroy())
        dialog.present()

    def _style_dialog(self, dialog: Gtk.Widget) -> None:
        try:
            apply_css()
        except Exception:  # pragma: no cover - defensive styling guard
            logger.debug("Unable to apply CSS styling to Google dialog.")

        get_style_context = getattr(dialog, "get_style_context", None)
        if callable(get_style_context):
            try:
                style_context = get_style_context()
            except Exception:  # pragma: no cover - stub fallback
                style_context = None
            if style_context is not None:
                add_class = getattr(style_context, "add_class", None)
                if callable(add_class):
                    for css_class in ("chat-page", "sidebar"):
                        try:
                            add_class(css_class)
                        except Exception:  # pragma: no cover - ignore styling issues
                            continue
