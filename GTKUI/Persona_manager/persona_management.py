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

    def show_persona_menu(self):
        """
        Displays the "Select Persona" window. This window lists all available
        personas with a label and a settings icon next to each name. The window
        is styled to match the sidebar by applying the same CSS and style class.
        """
        self.persona_window = Gtk.Window(title="Select Persona")
        # Set a default size; adjust as needed
        self.persona_window.set_default_size(150, 600)
        # Apply the same CSS as the sidebar
        apply_css()
        # Add the 'sidebar' style class to ensure matching background and font colors
        self.persona_window.get_style_context().add_class("sidebar")

        # Create a vertical box container with uniform spacing and margins
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        self.persona_window.set_child(box)

        # Retrieve persona names from ATLAS
        persona_names = self.ATLAS.get_persona_names()

        for persona_name in persona_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            # Create the label for the persona name.
            label = Gtk.Label(label=persona_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)
            hbox.append(label)

            # Create a gesture for clicking on the label to select the persona.
            label_gesture = Gtk.GestureClick.new()
            label_gesture.connect(
                "pressed",
                lambda gesture, n_press, x, y, persona_name=persona_name: self.select_persona(persona_name)
            )
            label.add_controller(label_gesture)

            # Create a settings icon for the persona.
            settings_icon_path = os.path.join(os.path.dirname(__file__), "../../Icons/settings.png")
            settings_icon_path = os.path.abspath(settings_icon_path)
            try:
                texture = Gdk.Texture.new_from_filename(settings_icon_path)
                settings_icon = Gtk.Picture.new_for_paintable(texture)
                settings_icon.set_size_request(16, 16)
                settings_icon.set_content_fit(Gtk.ContentFit.CONTAIN)
            except GLib.Error as e:
                print(f"Error loading icon {settings_icon_path}: {e}")
                settings_icon = Gtk.Image.new_from_icon_name("image-missing")

            # Attach a gesture to the settings icon to open persona settings.
            settings_gesture = Gtk.GestureClick.new()
            settings_gesture.connect(
                "pressed",
                lambda gesture, n_press, x, y, persona_name=persona_name: self.open_persona_settings(persona_name)
            )
            settings_icon.add_controller(settings_gesture)

            hbox.append(settings_icon)
            box.append(hbox)

        self.persona_window.present()

    def select_persona(self, persona):
        """
        Loads the specified persona.

        Args:
            persona (str): The name of the persona to select.
        """
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.persona_manager.current_system_prompt}")

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
        settings_window = Gtk.Window(title=f"Settings for {persona.get('name')}")
        settings_window.set_default_size(500, 800)
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

        stack_switcher = Gtk.StackSwitcher()
        stack_switcher.set_stack(stack)
        main_vbox.append(stack_switcher)
        main_vbox.append(stack)

        # General Tab (with scrollable box)
        self.general_tab = GeneralTab(persona)
        general_box = self.general_tab.get_widget()
        scrolled_general_tab = Gtk.ScrolledWindow()
        scrolled_general_tab.set_child(general_box)
        stack.add_titled(scrolled_general_tab, "general", "General")

        # Persona Type Tab (with scrollable box)
        self.persona_type_tab = PersonaTypeTab(persona, self.general_tab)
        type_box = self.persona_type_tab.get_widget()
        scrolled_persona_type = Gtk.ScrolledWindow()
        scrolled_persona_type.set_child(type_box)
        stack.add_titled(scrolled_persona_type, "persona_type", "Persona Type")

        # Provider and Model Tab
        provider_model_box = self.create_provider_model_tab(persona)
        stack.add_titled(provider_model_box, "provider_model", "Provider & Model")

        # Speech Provider and Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona)
        stack.add_titled(speech_voice_box, "speech_voice", "Speech & Voice")

        # Save Button at the bottom
        save_button = Gtk.Button(label="Save")
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
        self.provider_entry = Gtk.Entry()
        self.provider_entry.set_text(persona.get("provider", "openai"))
        provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        provider_box.append(provider_label)
        provider_box.append(self.provider_entry)
        provider_model_box.append(provider_box)

        # Model
        model_label = Gtk.Label(label="Model")
        self.model_entry = Gtk.Entry()
        self.model_entry.set_text(persona.get("model", "gpt-4"))
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
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
        self.speech_provider_entry = Gtk.Entry()
        self.speech_provider_entry.set_text(persona.get("Speech_provider", "11labs"))
        speech_provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        speech_provider_box.append(speech_provider_label)
        speech_provider_box.append(self.speech_provider_entry)
        speech_voice_box.append(speech_provider_box)

        # Voice
        voice_label = Gtk.Label(label="Voice")
        self.voice_entry = Gtk.Entry()
        self.voice_entry.set_text(persona.get("voice", "jack"))
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
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
        name = self.general_tab.get_name()
        meaning = self.general_tab.get_meaning()
        editable_content = self.general_tab.get_editable_content()
        start_locked_content = self.general_tab.get_start_locked()
        end_locked_content = self.general_tab.get_end_locked()

        # Get values from persona type tab
        values = self.persona_type_tab.get_values()
        sys_info_enabled = values.get('sys_info_enabled', False)
        user_profile_enabled = values.get('user_profile_enabled', False)
        persona_type_values = values.get('type', {})

        # Get provider/model values
        provider = self.provider_entry.get_text()
        model = self.model_entry.get_text()

        # Get speech/voice values
        speech_provider = self.speech_provider_entry.get_text()
        voice = self.voice_entry.get_text()

        persona['name'] = name
        persona['meaning'] = meaning
        content = persona.get('content', {})
        content['start_locked'] = start_locked_content
        content['editable_content'] = editable_content
        content['end_locked'] = end_locked_content
        persona['content'] = content
        persona['sys_info_enabled'] = "True" if sys_info_enabled else "False"
        persona['user_profile_enabled'] = "True" if user_profile_enabled else "False"

        persona['type'] = {}
        persona_types = [
            'Agent', 'medical_persona', 'educational_persona', 'fitness_persona', 'language_instructor',
            'legal_persona', 'financial_advisor', 'tech_support', 'personal_assistant', 'therapist',
            'travel_guide', 'storyteller', 'game_master', 'chef'
        ]
        for key in persona_types:
            enabled = persona_type_values.get(key, {}).get('enabled', False)
            persona['type'][key] = {"enabled": str(enabled)}

        # Save additional options under 'type'
        additional_keys = {
            'educational_persona': ['subject_specialization', 'education_level', 'teaching_style'],
            'fitness_persona': ['fitness_goal', 'exercise_preference'],
            'language_instructor': ['target_language', 'proficiency_level'],
            # Add other persona type additional options as needed
        }
        for persona_type, keys in additional_keys.items():
            if persona['type'].get(persona_type, {}).get('enabled') == "True":
                for key in keys:
                    if key in persona_type_values.get(persona_type, {}):
                        persona['type'][persona_type][key] = persona_type_values[persona_type][key]
                    else:
                        persona['type'][persona_type].pop(key, None)

        persona['provider'] = provider
        persona['model'] = model
        persona['Speech_provider'] = speech_provider
        persona['voice'] = voice

        self.ATLAS.persona_manager.update_persona(persona)
        print(f"Settings for {name} saved!")
        settings_window.destroy()
