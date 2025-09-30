from __future__ import annotations

"""GTK dialog for configuring Google Gemini provider defaults."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box

logger = logging.getLogger(__name__)


class GoogleSettingsWindow(Gtk.Window):
    """Collect Google Gemini defaults such as model selection and safety filters."""

    _SAFETY_CATEGORIES: Tuple[Tuple[str, str], ...] = (
        ("Harassment / Abuse", "HARM_CATEGORY_HARASSMENT_ABUSE"),
        ("Hate Speech", "HARM_CATEGORY_HATE_SPEECH"),
        ("Sexually Explicit", "HARM_CATEGORY_SEXUALLY_EXPLICIT"),
        ("Dangerous Content", "HARM_CATEGORY_DANGEROUS_CONTENT"),
    )

    _SAFETY_THRESHOLDS: Tuple[Tuple[str, str], ...] = (
        ("Allow all content", "BLOCK_NONE"),
        ("Block only high risk", "BLOCK_ONLY_HIGH"),
        ("Block medium and above", "BLOCK_MEDIUM_AND_ABOVE"),
        ("Block low and above", "BLOCK_LOW_AND_ABOVE"),
    )

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(title="Google Settings")
        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window
        if parent_window is not None:
            self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_default_size(420, 520)

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
        }
        self._safety_controls: Dict[str, Tuple[Gtk.CheckButton, Gtk.ComboBoxText]] = {}
        self._available_models: List[str] = []

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        self.set_child(scroller)

        main_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        scroller.set_child(main_box)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        main_box.append(grid)

        row = 0
        api_label = Gtk.Label(label="Google API Key:")
        api_label.set_xalign(0.0)
        grid.attach(api_label, 0, row, 1, 1)

        api_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        grid.attach(api_box, 1, row, 1, 1)

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
        grid.attach(self.api_key_status_label, 0, row, 2, 1)

        row += 1
        model_label = Gtk.Label(label="Default model:")
        model_label.set_xalign(0.0)
        grid.attach(model_label, 0, row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        self.model_combo.set_tooltip_text("Pick the default Gemini model used for completions.")
        grid.attach(self.model_combo, 1, row, 1, 1)

        row += 1
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_xalign(0.0)
        grid.attach(temp_label, 0, row, 1, 1)

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
        grid.attach(self.temperature_spin, 1, row, 1, 1)

        row += 1
        top_p_label = Gtk.Label(label="Top-p:")
        top_p_label.set_xalign(0.0)
        grid.attach(top_p_label, 0, row, 1, 1)

        self.top_p_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.05, value=1.0
        )
        self.top_p_spin = Gtk.SpinButton(adjustment=self.top_p_adjustment, digits=2)
        self.top_p_spin.set_increments(0.01, 0.05)
        self.top_p_spin.set_tooltip_text(
            "Nucleus sampling parameter. Lower values focus on the most likely tokens."
        )
        grid.attach(self.top_p_spin, 1, row, 1, 1)

        row += 1
        top_k_label = Gtk.Label(label="Top-k override:")
        top_k_label.set_xalign(0.0)
        grid.attach(top_k_label, 0, row, 1, 1)

        top_k_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        grid.attach(top_k_box, 1, row, 1, 1)

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

        row += 1
        candidate_label = Gtk.Label(label="Candidate count:")
        candidate_label.set_xalign(0.0)
        grid.attach(candidate_label, 0, row, 1, 1)

        self.candidate_adjustment = Gtk.Adjustment(
            lower=1, upper=8, step_increment=1, page_increment=1, value=1
        )
        self.candidate_spin = Gtk.SpinButton(adjustment=self.candidate_adjustment, digits=0)
        self.candidate_spin.set_tooltip_text(
            "Number of candidates to request per prompt. Higher values increase cost."
        )
        grid.attach(self.candidate_spin, 1, row, 1, 1)

        row += 1
        self.stream_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.stream_toggle.set_halign(Gtk.Align.START)
        self.stream_toggle.set_tooltip_text(
            "Toggle to stream token updates during Gemini completions."
        )
        grid.attach(self.stream_toggle, 0, row, 2, 1)

        row += 1
        max_output_label = Gtk.Label(label="Max output tokens:")
        max_output_label.set_xalign(0.0)
        grid.attach(max_output_label, 0, row, 1, 1)

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
        grid.attach(self.max_output_tokens_spin, 1, row, 1, 1)

        row += 1
        stop_label = Gtk.Label(label="Stop sequences:")
        stop_label.set_xalign(0.0)
        grid.attach(stop_label, 0, row, 1, 1)

        self.stop_sequences_entry = Gtk.Entry()
        self.stop_sequences_entry.set_hexpand(True)
        self.stop_sequences_entry.set_placeholder_text("Comma-separated tokens to stop generation")
        self.stop_sequences_entry.set_tooltip_text(
            "Provide comma separated stop strings. Generation halts when a match is produced."
        )
        grid.attach(self.stop_sequences_entry, 1, row, 1, 1)

        row += 1
        mime_label = Gtk.Label(label="Response MIME type:")
        mime_label.set_xalign(0.0)
        grid.attach(mime_label, 0, row, 1, 1)

        self.response_mime_entry = Gtk.Entry()
        self.response_mime_entry.set_hexpand(True)
        self.response_mime_entry.set_placeholder_text("e.g. text/plain or application/json")
        self.response_mime_entry.set_tooltip_text(
            "Optional MIME type for responses when using multimodal Gemini features."
        )
        grid.attach(self.response_mime_entry, 1, row, 1, 1)

        row += 1
        system_label = Gtk.Label(label="System instruction:")
        system_label.set_xalign(0.0)
        grid.attach(system_label, 0, row, 1, 1)

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
        grid.attach(system_scroller, 1, row, 1, 1)

        row += 1
        safety_frame = Gtk.Frame(label="Safety filters")
        safety_frame.set_tooltip_text(
            "Configure safety filters to block responses from specific harm categories."
        )
        main_box.append(safety_frame)

        safety_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        safety_frame.set_child(safety_box)

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

            safety_box.append(row_box)
            self._safety_controls[category] = (toggle, combo)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        button_box.set_halign(Gtk.Align.END)
        main_box.append(button_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.set_tooltip_text("Close without saving changes.")
        cancel_button.connect("clicked", lambda *_args: self.close())
        button_box.append(cancel_button)

        save_button = Gtk.Button(label="Save Settings")
        save_button.set_tooltip_text("Persist defaults and refresh the Google provider.")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

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

        self.stream_toggle.set_active(bool(self._defaults.get("stream", True)))

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
            current = entry.get("category") or entry.get("harmCategory")
            if current == category:
                threshold = entry.get("threshold") or entry.get("thresholdValue") or entry.get("harmBlockThreshold")
                if not threshold:
                    continue
                return {"category": category, "threshold": str(threshold)}
        return None

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

    def _on_top_k_toggled(self, toggle: Gtk.CheckButton) -> None:
        active = toggle.get_active()
        self.top_k_spin.set_sensitive(active)

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
        }

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

    def _collect_safety_settings(self) -> List[Dict[str, str]]:
        settings: List[Dict[str, str]] = []
        for category, controls in self._safety_controls.items():
            toggle, combo = controls
            if not toggle.get_active():
                continue
            index = combo.get_active()
            if index < 0:
                index = 2
            threshold = self._SAFETY_THRESHOLDS[index][1]
            settings.append({"category": category, "threshold": threshold})
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

    def _select_safety_threshold(self, combo: Gtk.ComboBoxText, threshold: str) -> None:
        normalized = threshold.strip().upper()
        for idx, (_label, value) in enumerate(self._SAFETY_THRESHOLDS):
            if value == normalized:
                combo.set_active(idx)
                break

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

        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:  # pragma: no cover - GTK3 compatibility
            dialog.props.secondary_text = message

        dialog.connect("response", lambda dlg, _resp: dlg.destroy())
        dialog.present()
