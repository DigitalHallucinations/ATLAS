# GTKUI/Persona_manager/persona_management.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

import os

from .General_Tab.general_tab import GeneralTab
from .Persona_Type_Tab.persona_type_tab import PersonaTypeTab
# Import the centralized CSS application function
from GTKUI.Utils.utils import apply_css


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
        self._persona_message_handler = self._handle_persona_message
        self.ATLAS.register_message_dispatcher(self._persona_message_handler)

    # --------------------------- Helpers ---------------------------

    def _abs_icon(self, rel_path: str) -> str:
        """Resolve absolute path to an icon from project root."""
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base, rel_path)

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

    # --------------------------- Menu ---------------------------

    def show_persona_menu(self):
        """
        Displays the "Select Persona" window. This window lists all available
        personas with a label and a settings icon next to each name. The window
        is styled to match the sidebar by applying the same CSS and style class.
        """
        self.persona_window = Gtk.Window(title="Select Persona")
        self.persona_window.set_default_size(220, 600)
        self.persona_window.set_transient_for(self.parent_window)
        self.persona_window.set_modal(True)

        # Apply the same CSS as the sidebar and set class
        apply_css()
        self.persona_window.get_style_context().add_class("sidebar")
        self.persona_window.set_tooltip_text("Choose a persona or open its settings.")

        # Container inside a scrolled window so long persona lists remain usable
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_propagate_natural_height(True)
        scroll.set_hexpand(True)
        scroll.set_vexpand(True)
        self.persona_window.set_child(scroll)

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

        # Retrieve persona names from ATLAS
        persona_names = self.ATLAS.get_persona_names() or []

        for persona_name in persona_names:
            # Each row is an HBox with a "Select" button (label-styled) and a gear button
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            # Select button for persona (improves keyboard/accessibility over raw label+gesture)
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
            name_lbl.get_style_context().add_class("provider-label")  # reuse bold-ish style if present
            name_lbl.set_halign(Gtk.Align.START)
            name_lbl.set_hexpand(True)
            select_btn.set_child(name_lbl)

            # When clicked, select persona
            select_btn.connect("clicked", lambda _b, pname=persona_name: self.select_persona(pname))

            # Settings button (gear icon)
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

            # Put row into a ListBoxRow for nicer selection/hover
            lrow = Gtk.ListBoxRow()
            lrow.set_child(row)
            list_box.append(lrow)

        # Hint footer
        hint = Gtk.Label(label="Tip: double-click a row to select.")
        hint.set_tooltip_text("You can also use arrow keys to move and Enter to activate.")
        hint.set_margin_top(6)
        outer.append(hint)

        # Double-click behavior on rows (optional)
        def on_row_activated(_lb, lbrow):
            # Find the select button inside the row and activate it
            box = lbrow.get_child()
            if isinstance(box, Gtk.Box):
                child = box.get_first_child()
                if isinstance(child, Gtk.Button):
                    child.emit("clicked")

        list_box.connect("row-activated", on_row_activated)

        self.persona_window.present()

    def select_persona(self, persona):
        """
        Loads the specified persona.

        Args:
            persona (str): The name of the persona to select.
        """
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.persona_manager.current_system_prompt}")

    # --------------------------- Settings ---------------------------

    def open_persona_settings(self, persona_name):
        """
        Opens the persona settings window for the specified persona.

        Args:
            persona_name (str): The name of the persona.
        """
        if self.persona_window:
            self.persona_window.close()

        persona = self.ATLAS.persona_manager.get_persona(persona_name)
        self.show_persona_settings(persona)

    def show_persona_settings(self, persona):
        """
        Displays the settings window for a given persona. This window includes
        tabs for General settings, Persona Type settings, and other configuration
        options. The styling is applied consistently with the rest of the UI.

        Args:
            persona (dict): The persona data.
        """
        title = f"Settings for {persona.get('name')}"
        settings_window = Gtk.Window(title=title)
        settings_window.set_default_size(560, 820)
        settings_window.set_transient_for(self.parent_window)
        settings_window.set_modal(True)
        settings_window.set_tooltip_text("Configure the persona's details, provider/model, and speech options.")

        # Apply the centralized CSS to ensure consistent styling.
        apply_css()
        # Optionally, add a style class to match the sidebar.
        settings_window.get_style_context().add_class("sidebar")

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
        self.general_tab = GeneralTab(persona, self.ATLAS.persona_manager)
        general_box = self.general_tab.get_widget()
        scrolled_general_tab = Gtk.ScrolledWindow()
        scrolled_general_tab.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_general_tab.set_child(general_box)
        stack.add_titled(scrolled_general_tab, "general", "General")
        scrolled_general_tab.set_tooltip_text("Edit persona name, meaning, and prompt content.")

        # Persona Type Tab (with scrollable box)
        self.persona_type_tab = PersonaTypeTab(persona, self.general_tab)
        type_box = self.persona_type_tab.get_widget()
        scrolled_persona_type = Gtk.ScrolledWindow()
        scrolled_persona_type.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_persona_type.set_child(type_box)
        scrolled_persona_type.set_tooltip_text("Enable persona roles and configure their specific options.")
        stack.add_titled(scrolled_persona_type, "persona_type", "Persona Type")

        # Provider and Model Tab
        provider_model_box = self.create_provider_model_tab(persona)
        provider_model_box.set_tooltip_text("Select which provider/model this persona should use.")
        stack.add_titled(provider_model_box, "provider_model", "Provider & Model")

        # Speech Provider and Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona)
        speech_voice_box.set_tooltip_text("Select speech provider and voice defaults for this persona.")
        stack.add_titled(speech_voice_box, "speech_voice", "Speech & Voice")

        # Save Button at the bottom
        save_button = Gtk.Button(label="Save")
        save_button.set_tooltip_text("Save all changes to this persona.")
        save_button.connect("clicked", lambda widget: self.save_persona_settings(persona, settings_window))
        main_vbox.append(save_button)

        settings_window.present()

    def create_provider_model_tab(self, persona):
        """
        Creates the Provider and Model settings tab.

        Args:
            persona (dict): The persona data.

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
        self.provider_entry.set_text(persona.get("provider", "openai"))
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
        self.model_entry.set_text(persona.get("model", "gpt-4"))
        self.model_entry.set_tooltip_text("Exact model name to request from the provider.")
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_box.append(model_label)
        model_box.append(self.model_entry)
        provider_model_box.append(model_box)

        return provider_model_box

    def create_speech_voice_tab(self, persona):
        """
        Creates the Speech Provider and Voice settings tab.

        Args:
            persona (dict): The persona data.

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
        self.speech_provider_entry.set_text(persona.get("Speech_provider", "11labs"))
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
        self.voice_entry.set_text(persona.get("voice", "jack"))
        self.voice_entry.set_tooltip_text("Default voice to synthesize for this persona.")
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        voice_box.append(voice_label)
        voice_box.append(self.voice_entry)
        speech_voice_box.append(voice_box)

        return speech_voice_box

    def save_persona_settings(self, persona, settings_window):
        """
        Gathers settings from the General, Persona Type, Provider/Model,
        and Speech/Voice tabs, updates the persona, and then saves the changes.

        Args:
            persona (dict): The persona data to update.
            settings_window (Gtk.Window): The settings window to close after saving.
        """
        general_payload = {
            'name': self.general_tab.get_name(),
            'meaning': self.general_tab.get_meaning(),
            'content': {
                'start_locked': self.general_tab.get_start_locked(),
                'editable_content': self.general_tab.get_editable_content(),
                'end_locked': self.general_tab.get_end_locked(),
            },
        }

        values = self.persona_type_tab.get_values() or {}
        persona_type_payload = {
            'sys_info_enabled': values.get('sys_info_enabled', False),
            'user_profile_enabled': values.get('user_profile_enabled', False),
            'type': {},
        }
        for type_name, type_values in (values.get('type') or {}).items():
            entry = dict(type_values)
            enabled_value = entry.get('enabled', False)
            if isinstance(enabled_value, str):
                entry['enabled'] = enabled_value.lower() == 'true'
            else:
                entry['enabled'] = bool(enabled_value)
            persona_type_payload['type'][type_name] = entry

        provider_payload = {
            'provider': self.provider_entry.get_text(),
            'model': self.model_entry.get_text(),
        }

        speech_payload = {
            'Speech_provider': self.speech_provider_entry.get_text(),
            'voice': self.voice_entry.get_text(),
        }

        result = self.ATLAS.persona_manager.update_persona_from_form(
            persona.get('name'),
            general_payload,
            persona_type_payload,
            provider_payload,
            speech_payload,
        )

        if result.get('success'):
            if result.get('persona') is not None:
                persona.clear()
                persona.update(result['persona'])
            print(f"Settings for {general_payload['name']} saved!")
            settings_window.destroy()
        else:
            error_text = "; ".join(result.get('errors', [])) or "Unable to save persona settings."
            self.ATLAS.persona_manager.show_message("system", f"Failed to save persona: {error_text}")
