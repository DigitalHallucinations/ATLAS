# GTKUI/Persona_manager/persona_management.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .General_Tab.general_tab import GeneralTab
from .Persona_Type_Tab.persona_type_tab import PersonaTypeTab
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css
from modules.logging.audit import PersonaAuditEntry, get_persona_audit_logger


class PersonaManagement:
    """
    Manages the persona selection and settings functionality.
    Displays a window listing available personas and allows switching or
    opening the settings for each persona.
    """

    def __init__(self, ATLAS, parent_window):
        """
        Initializes the PersonaManagement.

        Args:
            ATLAS (ATLAS): The main ATLAS instance.
            parent_window (Gtk.Window): The parent window for the persona menu.
        """
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.persona_window = None
        self._persona_page = None
        self.settings_window = None
        self._persona_message_handler = self._handle_persona_message
        self.ATLAS.register_message_dispatcher(self._persona_message_handler)
        self._current_editor_state = None
        self.tool_rows: Dict[str, Dict[str, Any]] = {}
        self._tool_order: List[str] = []
        self.tool_list_box: Optional[Gtk.ListBox] = None
        self.history_list_box: Optional[Gtk.ListBox] = None
        self._history_persona_name: Optional[str] = None
        self._history_page_size: int = 20
        self._history_offset: int = 0
        self._history_total: int = 0
        self._history_load_more_button: Optional[Gtk.Button] = None
        self._history_placeholder: Optional[Gtk.Widget] = None
        self._last_bundle_directory: Optional[Path] = None

    # --------------------------- Helpers ---------------------------

    def _abs_icon(self, rel_path: str) -> str:
        """Resolve absolute path to an icon from project root."""
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base, rel_path)

    def _style_dialog(self, dialog: Gtk.Dialog) -> None:
        """Ensure dialogs inherit the app styling and dark theme classes."""
        apply_css()
        context = dialog.get_style_context()
        context.add_class("chat-page")
        context.add_class("sidebar")

    def _handle_persona_message(self, role: str, message: str) -> None:
        """Surface persona manager messages via a modal dialog."""
        if not message:
            return

        role_normalized = (role or "").lower()
        if role_normalized == "error":
            message_type = Gtk.MessageType.ERROR
            title = "Error"
        elif role_normalized == "warning":
            message_type = Gtk.MessageType.WARNING
            title = "Warning"
        else:
            message_type = Gtk.MessageType.INFO
            title = "System"

        def _present_dialog():
            dialog = Gtk.MessageDialog(
                transient_for=self.parent_window,
                modal=True,
                message_type=message_type,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
            self._style_dialog(dialog)
            dialog.format_secondary_text(message)
            dialog.connect("response", lambda d, *_: d.destroy())
            dialog.present()
            return False

        GLib.idle_add(_present_dialog)

    def _make_icon_widget(self, rel_path: str, fallback_icon_name: str = "emblem-system-symbolic", size: int = 16) -> Gtk.Widget:
        """
        Try to create a Gtk.Picture from a file path; fall back to a themed icon name.
        Returns a Gtk.Picture (file) or Gtk.Image (themed) widget.
        """
        try:
            path = self._abs_icon(rel_path)
            texture = Gdk.Texture.new_from_filename(path)
            pic = Gtk.Picture.new_for_paintable(texture)
            pic.set_size_request(size, size)
            pic.set_content_fit(Gtk.ContentFit.CONTAIN)
            return pic
        except Exception:
            img = Gtk.Image.new_from_icon_name(fallback_icon_name)
            img.set_pixel_size(size)
            return img

    def _choose_file_path(
        self,
        *,
        title: str,
        action: str,
        suggested_name: Optional[str] = None,
    ) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, action.upper(), None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            return None

        chooser = chooser_cls(title=title, transient_for=self.parent_window, modal=True, action=action_enum)
        if suggested_name and hasattr(chooser, "set_current_name"):
            try:
                chooser.set_current_name(suggested_name)
            except Exception:
                pass

        if self._last_bundle_directory and hasattr(chooser, "set_current_folder"):
            try:
                chooser.set_current_folder(str(self._last_bundle_directory))
            except Exception:
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

        accepted_responses = {
            getattr(Gtk.ResponseType, "ACCEPT", None),
            getattr(Gtk.ResponseType, "OK", None),
            getattr(Gtk.ResponseType, "YES", None),
        }

        filename: Optional[str] = None
        if response in accepted_responses:
            file_obj = getattr(chooser, "get_file", lambda: None)()
            if file_obj is not None and hasattr(file_obj, "get_path"):
                filename = file_obj.get_path()
            elif hasattr(chooser, "get_filename"):
                filename = chooser.get_filename()

        if hasattr(chooser, "destroy"):
            try:
                chooser.destroy()
            except Exception:
                pass

        if filename:
            path_obj = Path(filename).expanduser().resolve()
            self._last_bundle_directory = path_obj.parent
            return str(path_obj)

        return None

    def _prompt_signing_key(self, title: str) -> Optional[str]:
        dialog_cls = getattr(Gtk, "Dialog", None)
        entry_cls = getattr(Gtk, "Entry", None)
        if dialog_cls is None or entry_cls is None:
            return None

        dialog = dialog_cls(title=title, transient_for=self.parent_window, modal=True)
        self._style_dialog(dialog)

        content_area = getattr(dialog, "get_content_area", lambda: None)()
        if content_area is None and hasattr(dialog, "set_child"):
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            dialog.set_child(box)
            content_area = box

        entry = entry_cls()
        if hasattr(entry, "set_visibility"):
            entry.set_visibility(False)

        label = Gtk.Label(label="Signing key:")
        if content_area is not None and hasattr(content_area, "append"):
            content_area.append(label)
            content_area.append(entry)

        if hasattr(dialog, "add_button"):
            dialog.add_button("Cancel", getattr(Gtk.ResponseType, "CANCEL", 0))
            dialog.add_button("OK", getattr(Gtk.ResponseType, "OK", 1))

        response = None
        if hasattr(dialog, "run"):
            try:
                response = dialog.run()
            except Exception:
                response = None
        else:
            response = getattr(Gtk.ResponseType, "OK", 1)

        key_value = entry.get_text().strip() if hasattr(entry, "get_text") else ""

        if hasattr(dialog, "destroy"):
            try:
                dialog.destroy()
            except Exception:
                pass

        if response == getattr(Gtk.ResponseType, "OK", 1) and key_value:
            return key_value
        return None

    def _refresh_persona_catalog(self) -> None:
        manager = getattr(self.ATLAS, "persona_manager", None)
        loader = getattr(manager, "load_persona_names", None)
        base_path = getattr(manager, "persona_base_path", None)

        if callable(loader) and base_path is not None:
            try:
                names = loader(base_path)
            except Exception:
                names = None
            else:
                setattr(manager, "persona_names", names)

        self._persona_page = None

    def _on_export_persona_clicked(self, _button):
        if not self._current_editor_state:
            self.ATLAS.show_persona_message("error", "No persona loaded for export.")
            return

        persona_name = self._current_editor_state.get("original_name") or ""
        persona_name = str(persona_name).strip()
        if not persona_name:
            self.ATLAS.show_persona_message("error", "Persona name is required for export.")
            return

        path = self._choose_file_path(
            title=f"Export {persona_name}",
            action="SAVE",
            suggested_name=f"{persona_name}.atlasbundle",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for export")
        if not signing_key:
            self.ATLAS.show_persona_message("warning", "Export cancelled: signing key required.")
            return

        try:
            response = self.ATLAS.export_persona_bundle(persona_name, signing_key=signing_key)
        except Exception:
            self.ATLAS.show_persona_message("error", "Failed to export persona bundle.")
            return

        if not response.get("success"):
            self.ATLAS.show_persona_message("error", response.get("error") or "Persona export failed.")
            return

        bundle_bytes = response.get("bundle_bytes")
        if not isinstance(bundle_bytes, (bytes, bytearray)):
            self.ATLAS.show_persona_message("error", "Persona export did not return bundle data.")
            return

        try:
            Path(path).write_bytes(bundle_bytes)
        except OSError:
            self.ATLAS.show_persona_message("error", f"Failed to write bundle to {path}.")
            return

        warnings = response.get("warnings") or []
        message = f"Exported persona '{persona_name}' to {path}."
        if warnings:
            message += " " + " ".join(warnings)
        self.ATLAS.show_persona_message("system", message)

    def _on_import_persona_clicked(self, _button):
        path = self._choose_file_path(
            title="Import Persona Bundle",
            action="OPEN",
        )
        if not path:
            return

        signing_key = self._prompt_signing_key("Enter signing key for import")
        if not signing_key:
            self.ATLAS.show_persona_message("warning", "Import cancelled: signing key required.")
            return

        try:
            bundle_bytes = Path(path).read_bytes()
        except OSError:
            self.ATLAS.show_persona_message("error", f"Failed to read bundle from {path}.")
            return

        try:
            response = self.ATLAS.import_persona_bundle(
                bundle_bytes=bundle_bytes,
                signing_key=signing_key,
                rationale="Imported via GTK UI",
            )
        except Exception:
            self.ATLAS.show_persona_message("error", "Failed to import persona bundle.")
            return

        if not response.get("success"):
            self.ATLAS.show_persona_message("error", response.get("error") or "Persona import failed.")
            return

        persona = response.get("persona") or {}
        persona_name = persona.get("name") or "persona"
        warnings = response.get("warnings") or []
        message = f"Imported persona '{persona_name}' from {path}."
        if warnings:
            message += " " + " ".join(warnings)
        self.ATLAS.show_persona_message("system", message)

        self._refresh_persona_catalog()
        if self.persona_window:
            try:
                self.persona_window.close()
            except Exception:
                pass
        self.show_persona_menu()

    # --------------------------- Menu ---------------------------

    def _build_persona_list_widget(self) -> Gtk.Widget:
        """Create the persona selection list suitable for embedding."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        outer.set_margin_top(10)
        outer.set_margin_bottom(10)
        outer.set_margin_start(10)
        outer.set_margin_end(10)
        outer.set_valign(Gtk.Align.START)
        scroll.set_child(outer)

        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.set_tooltip_text("Click a row to select the persona; use the gear to edit its settings.")
        list_box.set_valign(Gtk.Align.START)
        outer.append(list_box)

        persona_names = self.ATLAS.get_persona_names() or []

        for persona_name in persona_names:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            select_btn = Gtk.Button()
            select_btn.add_css_class("flat")
            select_btn.add_css_class("sidebar")
            select_btn.set_can_focus(True)
            select_btn.set_hexpand(True)
            select_btn.set_halign(Gtk.Align.FILL)
            select_btn.set_tooltip_text(f"Select persona: {persona_name}")

            name_lbl = Gtk.Label(label=persona_name)
            name_lbl.set_xalign(0.0)
            name_lbl.set_yalign(0.5)
            name_lbl.get_style_context().add_class("provider-label")
            name_lbl.set_halign(Gtk.Align.START)
            name_lbl.set_hexpand(True)
            select_btn.set_child(name_lbl)

            select_btn.connect("clicked", lambda _b, pname=persona_name: self.select_persona(pname))

            settings_btn = Gtk.Button()
            settings_btn.add_css_class("flat")
            settings_btn.set_can_focus(True)
            settings_btn.set_tooltip_text(f"Open settings for {persona_name}")
            role = getattr(Gtk.AccessibleRole, "BUTTON", None)
            if role is not None:
                settings_btn.set_accessible_role(role)

            gear = self._make_icon_widget("Icons/settings.png", fallback_icon_name="emblem-system-symbolic", size=16)
            settings_btn.set_child(gear)
            settings_btn.connect("clicked", lambda _b, pname=persona_name: self.open_persona_settings(pname))

            row.append(select_btn)
            row.append(settings_btn)

            lrow = Gtk.ListBoxRow()
            lrow.set_child(row)
            list_box.append(lrow)

        hint = Gtk.Label(label="Tip: double-click a row to select.")
        hint.set_tooltip_text("You can also use arrow keys to move and Enter to activate.")
        hint.set_margin_top(6)
        outer.append(hint)

        import_btn = Gtk.Button(label="Import persona…")
        import_btn.set_tooltip_text("Import a persona bundle from disk.")
        import_btn.connect("clicked", self._on_import_persona_clicked)
        outer.append(import_btn)

        def on_row_activated(_lb, lbrow):
            box = lbrow.get_child()
            if isinstance(box, Gtk.Box):
                child = box.get_first_child()
                if isinstance(child, Gtk.Button):
                    child.emit("clicked")

        list_box.connect("row-activated", on_row_activated)

        return scroll

    def get_embeddable_widget(self) -> Gtk.Widget:
        if self._persona_page is None:
            self._persona_page = self._build_persona_list_widget()
        return self._persona_page

    def show_persona_menu(self):
        """
        Displays the "Select Persona" window. This window lists all available
        personas with a label and a settings icon next to each name. Styling is
        provided via :class:`AtlasWindow` so the picker matches the chat UI.
        """
        self.persona_window = AtlasWindow(
            title="Select Persona",
            default_size=(220, 600),
            modal=True,
            transient_for=self.parent_window,
        )
        self.persona_window.set_tooltip_text("Choose a persona or open its settings.")

        # Container inside a scrolled window so long persona lists remain usable
        self.persona_window.set_child(self._build_persona_list_widget())
        self.persona_window.present()

    def select_persona(self, persona):
        """
        Loads the specified persona.

        Args:
            persona (str): The name of the persona to select.
        """
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.get_current_persona_prompt()}")

    # --------------------------- Settings ---------------------------

    def open_persona_settings(self, persona_name):
        """
        Opens the persona settings window for the specified persona.

        Args:
            persona_name (str): The name of the persona.
        """
        if self.persona_window:
            self.persona_window.close()

        state = self.ATLAS.get_persona_editor_state(persona_name)
        if state is None:
            self.ATLAS.show_persona_message(
                "error",
                f"Unable to load persona '{persona_name}' for editing.",
            )
            return

        general_state = state.get('general', {})
        persona_name = general_state.get('name') or state.get('original_name') or "Persona"

        if self.settings_window:
            self.settings_window.close()

        self.settings_window = AtlasWindow(
            title=f"Settings for {persona_name}",
            default_size=(560, 820),
            modal=True,
            transient_for=self.parent_window,
        )
        self.show_persona_settings(state, self.settings_window)

    def show_persona_settings(self, persona_state, settings_window):
        """Populate and display the persona settings window."""
        self._current_editor_state = persona_state
        self.tool_rows = {}
        self._tool_order = []
        self.tool_list_box = None
        settings_window.set_tooltip_text("Configure the persona's details, provider/model, and speech options.")

        # Create a vertical box container for the settings.
        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        main_vbox.set_margin_top(10)
        main_vbox.set_margin_bottom(10)
        main_vbox.set_margin_start(10)
        main_vbox.set_margin_end(10)
        settings_window.set_child(main_vbox)

        # Create a Gtk.Stack and Gtk.StackSwitcher for tabbed settings.
        stack = Gtk.Stack()
        stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        stack.set_transition_duration(200)
        # GTK4 quirk: Stack uses hhomogeneous / vhomogeneous (not 'homogeneous')
        if hasattr(stack, "set_hhomogeneous"):
            stack.set_hhomogeneous(True)
        if hasattr(stack, "set_vhomogeneous"):
            stack.set_vhomogeneous(True)

        stack_switcher = Gtk.StackSwitcher()
        stack_switcher.set_stack(stack)
        stack_switcher.set_tooltip_text("Switch between settings tabs.")
        main_vbox.append(stack_switcher)
        main_vbox.append(stack)

        # General Tab (with scrollable box)
        self.general_tab = GeneralTab(persona_state, self.ATLAS)
        general_box = self.general_tab.get_widget()
        scrolled_general_tab = Gtk.ScrolledWindow()
        scrolled_general_tab.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_general_tab.set_child(general_box)
        stack.add_titled(scrolled_general_tab, "general", "General")
        scrolled_general_tab.set_tooltip_text("Edit persona name, meaning, and prompt content.")

        # Persona Type Tab (with scrollable box)
        self.persona_type_tab = PersonaTypeTab(persona_state, self.general_tab)
        type_box = self.persona_type_tab.get_widget()
        scrolled_persona_type = Gtk.ScrolledWindow()
        scrolled_persona_type.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_persona_type.set_child(type_box)
        scrolled_persona_type.set_tooltip_text("Enable persona roles and configure their specific options.")
        stack.add_titled(scrolled_persona_type, "persona_type", "Persona Type")

        # Provider and Model Tab
        provider_model_box = self.create_provider_model_tab(persona_state)
        provider_model_box.set_tooltip_text("Select which provider/model this persona should use.")
        stack.add_titled(provider_model_box, "provider_model", "Provider & Model")

        # Speech Provider and Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona_state)
        speech_voice_box.set_tooltip_text("Select speech provider and voice defaults for this persona.")
        stack.add_titled(speech_voice_box, "speech_voice", "Speech & Voice")

        tools_box = self.create_tools_tab(persona_state)
        stack.add_titled(tools_box, "tools", "Tools")

        history_box = self.create_history_tab(persona_state)
        history_box.set_tooltip_text("Review persona change history for audit purposes.")
        stack.add_titled(history_box, "history", "History")

        # Save Button at the bottom
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        export_button = Gtk.Button(label="Export…")
        export_button.set_tooltip_text("Export this persona to a signed bundle.")
        export_button.connect("clicked", self._on_export_persona_clicked)
        actions_box.append(export_button)

        save_button = Gtk.Button(label="Save")
        save_button.set_tooltip_text("Save all changes to this persona.")
        save_button.connect("clicked", lambda _widget: self.save_persona_settings(settings_window))
        actions_box.append(save_button)

        main_vbox.append(actions_box)

        settings_window.present()

    def create_provider_model_tab(self, persona_state):
        """
        Creates the Provider and Model settings tab.

        Args:
            persona_state (dict): The persona data.

        Returns:
            Gtk.Box: The container with provider and model settings.
        """
        provider_model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Provider
        provider_label = Gtk.Label(label="Provider")
        provider_label.set_xalign(0.0)
        provider_label.set_tooltip_text("LLM provider key, e.g., 'OpenAI', 'Anthropic', 'HuggingFace', etc.")
        self.provider_entry = Gtk.Entry()
        self.provider_entry.set_placeholder_text("e.g., OpenAI")
        provider_defaults = persona_state.get('provider', {})
        self.provider_entry.set_text(provider_defaults.get("provider", "openai"))
        self.provider_entry.set_tooltip_text("Set which backend/provider this persona uses by default.")
        provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        provider_box.append(provider_label)
        provider_box.append(self.provider_entry)
        provider_model_box.append(provider_box)

        # Model
        model_label = Gtk.Label(label="Model")
        model_label.set_xalign(0.0)
        model_label.set_tooltip_text("Model identifier, e.g., 'gpt-4o', 'claude-3-opus', 'meta-llama-3-70b'.")
        self.model_entry = Gtk.Entry()
        self.model_entry.set_placeholder_text("e.g., gpt-4o")
        self.model_entry.set_text(provider_defaults.get("model", "gpt-4"))
        self.model_entry.set_tooltip_text("Exact model name to request from the provider.")
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_box.append(model_label)
        model_box.append(self.model_entry)
        provider_model_box.append(model_box)

        return provider_model_box

    def create_speech_voice_tab(self, persona_state):
        """
        Creates the Speech Provider and Voice settings tab.

        Args:
            persona_state (dict): The persona data.

        Returns:
            Gtk.Box: The container with speech and voice settings.
        """
        speech_voice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Speech Provider
        speech_provider_label = Gtk.Label(label="Speech Provider")
        speech_provider_label.set_xalign(0.0)
        speech_provider_label.set_tooltip_text("TTS/STT provider key, e.g., '11labs', 'openai_tts', 'google_tts'.")
        self.speech_provider_entry = Gtk.Entry()
        self.speech_provider_entry.set_placeholder_text("e.g., 11labs")
        speech_defaults = persona_state.get('speech', {})
        self.speech_provider_entry.set_text(speech_defaults.get("Speech_provider", "11labs"))
        self.speech_provider_entry.set_tooltip_text("Default speech provider for this persona.")
        speech_provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        speech_provider_box.append(speech_provider_label)
        speech_provider_box.append(self.speech_provider_entry)
        speech_voice_box.append(speech_provider_box)

        # Voice
        voice_label = Gtk.Label(label="Voice")
        voice_label.set_xalign(0.0)
        voice_label.set_tooltip_text("Voice identifier (depends on the provider).")
        self.voice_entry = Gtk.Entry()
        self.voice_entry.set_placeholder_text("e.g., Jack")
        self.voice_entry.set_text(speech_defaults.get("voice", "jack"))
        self.voice_entry.set_tooltip_text("Default voice to synthesize for this persona.")
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        voice_box.append(voice_label)
        voice_box.append(self.voice_entry)
        speech_voice_box.append(voice_box)

        return speech_voice_box

    def _format_timestamp(self, iso_timestamp: str) -> str:
        if not iso_timestamp:
            return "Unknown time"

        text = str(iso_timestamp)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return iso_timestamp

        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)

        return moment.strftime("%Y-%m-%d %H:%M UTC")

    def _format_tool_list(self, tools: List[str]) -> str:
        if not tools:
            return "none"
        return ", ".join(tools)

    def _format_tool_change(self, old_tools: List[str], new_tools: List[str]) -> str:
        return f"Tools: {self._format_tool_list(old_tools)} → {self._format_tool_list(new_tools)}"

    def _format_rationale(self, rationale: str) -> str:
        text = (rationale or "").strip()
        if not text:
            return "Rationale: Not provided."
        return f"Rationale: {text}"

    def _format_history_entry(self, entry: PersonaAuditEntry) -> str:
        timestamp = self._format_timestamp(entry.timestamp)
        username = entry.username or "unknown"
        return f"{timestamp} — {username}"

    def _build_history_row(self, entry: PersonaAuditEntry) -> Gtk.Widget:
        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        summary_label = Gtk.Label(label=self._format_history_entry(entry))
        summary_label.set_xalign(0.0)
        container.append(summary_label)

        tools_label = Gtk.Label(label=self._format_tool_change(entry.old_tools, entry.new_tools))
        tools_label.set_xalign(0.0)
        container.append(tools_label)

        rationale_label = Gtk.Label(label=self._format_rationale(entry.rationale))
        rationale_label.set_xalign(0.0)
        container.append(rationale_label)

        row = Gtk.ListBoxRow()
        setter = getattr(row, "set_child", None)
        if callable(setter):
            setter(container)
        else:  # pragma: no cover - GTK stubs without set_child
            row.children = [container]
        return row

    def _load_history_page(self, *, reset: bool = False) -> None:
        persona_name = self._history_persona_name
        list_box = self.history_list_box
        if not persona_name or list_box is None:
            return

        if reset:
            existing_children = list(getattr(list_box, "children", []))
            for child in existing_children:
                try:
                    list_box.remove(child)
                except Exception:  # pragma: no cover - stub fallback
                    continue
            self._history_offset = 0
            self._history_total = 0
            self._history_placeholder = None

        try:
            logger = get_persona_audit_logger()
            entries, total = logger.get_history(
                persona_name=persona_name,
                offset=self._history_offset,
                limit=self._history_page_size,
            )
        except Exception:
            self.ATLAS.show_persona_message(
                "error",
                "Unable to load persona change history.",
            )
            return

        self._history_total = total

        if reset and not entries:
            placeholder = Gtk.Label(label="No persona changes recorded yet.")
            placeholder.set_xalign(0.0)
            list_box.append(placeholder)
            self._history_placeholder = placeholder
        else:
            if self._history_placeholder is not None:
                try:
                    list_box.remove(self._history_placeholder)
                except Exception:  # pragma: no cover - stub fallback
                    pass
                self._history_placeholder = None

            for entry in entries:
                row = self._build_history_row(entry)
                list_box.append(row)

            self._history_offset += len(entries)

        has_more = self._history_offset < self._history_total
        if self._history_load_more_button is not None:
            self._history_load_more_button.set_visible(has_more)
            self._history_load_more_button.set_sensitive(has_more)

    def create_history_tab(self, persona_state) -> Gtk.Box:
        persona_name = ""
        if isinstance(persona_state, dict):
            general = persona_state.get('general')
            if isinstance(general, dict):
                persona_name = str(general.get('name') or "")
            if not persona_name:
                persona_name = str(persona_state.get('original_name') or "")

        history_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        history_container.set_margin_top(10)
        history_container.set_margin_bottom(10)
        history_container.set_margin_start(10)
        history_container.set_margin_end(10)

        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)
        history_container.append(scroll)

        self.history_list_box = Gtk.ListBox()
        self.history_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll.set_child(self.history_list_box)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        controls.set_valign(Gtk.Align.START)
        load_more_btn = Gtk.Button(label="Load More")
        load_more_btn.set_tooltip_text("Load older persona change events.")
        load_more_btn.connect("clicked", lambda _btn: self._load_history_page())
        controls.append(load_more_btn)
        history_container.append(controls)

        self._history_persona_name = persona_name
        self._history_offset = 0
        self._history_total = 0
        self._history_page_size = 20
        self._history_load_more_button = load_more_btn
        self._history_placeholder = None

        self._load_history_page(reset=True)

        return history_container

    def _format_tool_hint(self, metadata: Dict[str, Any]) -> str:
        safety = str(metadata.get('safety_level') or 'unspecified').strip()
        if safety:
            safety = safety.capitalize()
        cost_per_call = metadata.get('cost_per_call')
        cost_unit = str(metadata.get('cost_unit') or '').strip()
        if cost_per_call is None:
            cost_text = "n/a"
        else:
            try:
                cost_value = float(cost_per_call)
                if cost_value.is_integer():
                    cost_text = f"{int(cost_value)}"
                else:
                    cost_text = f"{cost_value:.3f}".rstrip('0').rstrip('.')
            except (TypeError, ValueError):
                cost_text = str(cost_per_call)
        if cost_unit:
            cost_display = f"{cost_text} {cost_unit}/call"
        else:
            cost_display = f"{cost_text}/call"
        return f"Safety: {safety or 'Unspecified'} • Cost: {cost_display}"

    def _ensure_tool_row_icons(self, icon_name: str) -> Gtk.Widget:
        try:
            image = Gtk.Image.new_from_icon_name(icon_name)
            image.set_pixel_size(14)
            return image
        except Exception:
            return Gtk.Image.new()

    def create_tools_tab(self, persona_state):
        tools_state = persona_state.get('tools') or {}
        available = tools_state.get('available') or []

        sorted_entries = sorted(
            [entry for entry in available if isinstance(entry, dict)],
            key=lambda entry: entry.get('order', 0),
        )

        tools_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        tools_box.set_margin_top(10)
        tools_box.set_margin_bottom(10)
        tools_box.set_margin_start(10)
        tools_box.set_margin_end(10)
        tools_box.set_tooltip_text("Enable, disable, and reorder tools available to this persona.")

        header = Gtk.Label(label="Select which tools this persona can use.")
        header.set_xalign(0.0)
        header.set_wrap(True)
        header.set_tooltip_text("Toggle tools to enable or disable them. Use the arrows to change invocation order.")
        tools_box.append(header)

        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.add_css_class("boxed-list")
        self.tool_list_box = list_box

        for entry in sorted_entries:
            name = entry.get('name')
            if not name:
                continue
            if name not in self._tool_order:
                self._tool_order.append(name)
            metadata = entry.get('metadata') if isinstance(entry, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            display_name = metadata.get('display_name') or metadata.get('title') or metadata.get('name') or name
            enabled = bool(entry.get('enabled'))

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

            check = Gtk.CheckButton.new_with_label(str(display_name))
            check.set_active(enabled)
            description = metadata.get('description')
            if isinstance(description, str) and description.strip():
                check.set_tooltip_text(description.strip())
            controls.append(check)

            button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            up_btn = Gtk.Button()
            up_btn.set_tooltip_text("Move up")
            up_btn.set_child(self._ensure_tool_row_icons("go-up-symbolic"))
            up_btn.connect("clicked", lambda _btn, tool=name: self._move_tool_row(tool, -1))
            button_box.append(up_btn)

            down_btn = Gtk.Button()
            down_btn.set_tooltip_text("Move down")
            down_btn.set_child(self._ensure_tool_row_icons("go-down-symbolic"))
            down_btn.connect("clicked", lambda _btn, tool=name: self._move_tool_row(tool, 1))
            button_box.append(down_btn)

            controls.append(button_box)
            row_box.append(controls)

            hint = Gtk.Label(label=self._format_tool_hint(metadata))
            hint.set_xalign(0.0)
            hint.get_style_context().add_class("dim-label")
            row_box.append(hint)

            list_row = Gtk.ListBoxRow()
            list_row.set_child(row_box)
            list_box.append(list_row)

            self.tool_rows[name] = {
                'row': list_row,
                'check': check,
                'up_button': up_btn,
                'down_button': down_btn,
                'metadata': metadata,
            }

        self._refresh_tool_reorder_controls()

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(list_box)
        tools_box.append(scroller)

        return tools_box

    def _refresh_tool_reorder_controls(self):
        total = len(self._tool_order)
        for index, name in enumerate(self._tool_order):
            info = self.tool_rows.get(name)
            if not info:
                continue
            up_button = info.get('up_button')
            down_button = info.get('down_button')
            if isinstance(up_button, Gtk.Button):
                up_button.set_sensitive(index > 0)
            if isinstance(down_button, Gtk.Button):
                down_button.set_sensitive(index < total - 1)

    def _move_tool_row(self, tool_name: str, direction: int):
        if tool_name not in self._tool_order or not self.tool_list_box:
            return

        current_index = self._tool_order.index(tool_name)
        new_index = current_index + direction
        if new_index < 0 or new_index >= len(self._tool_order):
            return

        self._tool_order.pop(current_index)
        self._tool_order.insert(new_index, tool_name)

        info = self.tool_rows.get(tool_name)
        if not info:
            return

        row = info.get('row')
        if not isinstance(row, Gtk.ListBoxRow):
            return

        self.tool_list_box.remove(row)
        self.tool_list_box.insert(row, new_index)
        if hasattr(self.tool_list_box, "invalidate_sort"):
            self.tool_list_box.invalidate_sort()
        self._refresh_tool_reorder_controls()

    def _collect_tool_payload(self) -> List[str]:
        allowed: List[str] = []
        for name in self._tool_order:
            info = self.tool_rows.get(name)
            if not info:
                continue
            check = info.get('check')
            if isinstance(check, Gtk.CheckButton) and check.get_active():
                allowed.append(name)
        return allowed

    def save_persona_settings(self, settings_window):
        """
        Gathers settings from the General, Persona Type, Provider/Model,
        and Speech/Voice tabs, updates the persona, and then saves the changes.

        Args:
            settings_window (Gtk.Window): The settings window to close after saving.
        """
        if not self._current_editor_state:
            self.ATLAS.show_persona_message(
                "system",
                "Persona data is unavailable; cannot save.",
            )
            return
        general_payload = {
            'name': self.general_tab.get_name(),
            'meaning': self.general_tab.get_meaning(),
            'content': {
                'start_locked': self.general_tab.get_start_locked(),
                'editable_content': self.general_tab.get_editable_content(),
                'end_locked': self.general_tab.get_end_locked(),
            },
        }

        persona_type_payload = self.persona_type_tab.get_values() or {}

        provider_payload = {
            'provider': self.provider_entry.get_text(),
            'model': self.model_entry.get_text(),
        }

        speech_payload = {
            'Speech_provider': self.speech_provider_entry.get_text(),
            'voice': self.voice_entry.get_text(),
        }

        tools_payload = self._collect_tool_payload()

        result = self.ATLAS.update_persona_from_editor(
            self._current_editor_state.get('original_name'),
            general_payload,
            persona_type_payload,
            provider_payload,
            speech_payload,
            tools_payload,
        )

        if result.get('success'):
            persona_result = result.get('persona')
            target_name = general_payload['name']
            if persona_result:
                target_name = persona_result.get('name') or target_name
            refreshed_state = self.ATLAS.get_persona_editor_state(target_name)
            if refreshed_state is not None:
                self._current_editor_state.clear()
                self._current_editor_state.update(refreshed_state)
            print(f"Settings for {general_payload['name']} saved!")
            settings_window.destroy()
        else:
            error_text = "; ".join(result.get('errors', [])) or "Unable to save persona settings."
            self.ATLAS.show_persona_message("system", f"Failed to save persona: {error_text}")
