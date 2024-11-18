# UI/persona_management.py
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from .General_Tab.general_tab import GeneralTab
from .Persona_Type_Tab.persona_type_tab import PersonaTypeTab


class PersonaManagement:
    def __init__(self, ATLAS, parent_window):
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.persona_window = None

    def show_persona_menu(self):
        self.persona_window = Gtk.Window(title="Select Persona")
        self.persona_window.set_default_size(150, 600)

        # Removed set_keep_above(True) as it's deprecated in GTK 4

        # Note: Window positioning methods like move() are not available in GTK 4
        # Window positioning is managed by the window manager
        # self.position_window_next_to_sidebar(self.persona_window, 150)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.persona_window.set_child(box)

        persona_names = self.ATLAS.get_persona_names()

        for persona_name in persona_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            label = Gtk.Label(label=persona_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)

            # Connect click gesture to label
            label_gesture = Gtk.GestureClick()
            label_gesture.connect(
                "pressed",
                lambda gesture, n_press, x, y, persona_name=persona_name: self.select_persona(persona_name)
            )
            label.add_controller(label_gesture)

            settings_icon_path = "Icons/settings.png"
            try:
                settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            except GLib.Error as e:
                print(f"Error loading icon {settings_icon_path}: {e}")
                settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("Icons/default.png", 16, 16, True)  # Fallback icon
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)
            settings_icon.set_margin_start(20)

            # Connect click gesture to settings icon
            settings_gesture = Gtk.GestureClick()
            settings_gesture.connect(
                "pressed",
                lambda gesture, n_press, x, y, persona_name=persona_name: self.open_persona_settings(persona_name)
            )
            settings_icon.add_controller(settings_gesture)

            hbox.append(label)
            hbox.append(settings_icon)

            box.append(hbox)

        self.persona_window.present()

    def select_persona(self, persona):
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.persona_manager.current_system_prompt}")

    def open_persona_settings(self, persona_name):
        if self.persona_window:
            self.persona_window.close()

        persona = self.ATLAS.persona_manager.get_persona(persona_name)
        self.show_persona_settings(persona)

    def show_persona_settings(self, persona):
        settings_window = Gtk.Window(title=f"Settings for {persona.get('name')}")
        settings_window.set_default_size(500, 800)

        self.apply_css_styling()

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        settings_window.set_child(main_vbox)

        # Using Gtk.Stack and Gtk.StackSwitcher instead of Gtk.Notebook
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

        # Save button
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", lambda widget: self.save_persona_settings(persona, settings_window))
        main_vbox.append(save_button)

        settings_window.present()
        # Note: Window positioning is managed by the window manager in GTK 4
        # self.position_window_next_to_sidebar(settings_window, 500)

    # Removed get_entry_text method as we are storing references to entries directly

    # Additional methods for provider and speech settings
    def create_provider_model_tab(self, persona):
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
        # Get values from general_tab
        name = self.general_tab.get_name()
        meaning = self.general_tab.get_meaning()
        editable_content = self.general_tab.get_editable_content()
        start_locked_content = self.general_tab.get_start_locked()
        end_locked_content = self.general_tab.get_end_locked()

        # Get values from persona_type_tab
        values = self.persona_type_tab.get_values()
        sys_info_enabled = values.get('sys_info_enabled', False)
        user_profile_enabled = values.get('user_profile_enabled', False)
        persona_type_values = values.get('type', {})

        # Get values from provider_model_box
        provider = self.provider_entry.get_text()
        model = self.model_entry.get_text()

        # Get values from speech_voice_box
        speech_provider = self.speech_provider_entry.get_text()
        voice = self.voice_entry.get_text()

        # Now save to persona
        persona['name'] = name
        persona['meaning'] = meaning

        # Update content parts
        content = persona.get('content', {})
        content['start_locked'] = start_locked_content
        content['editable_content'] = editable_content
        content['end_locked'] = end_locked_content
        persona['content'] = content

        # Save top-level flags
        persona['sys_info_enabled'] = "True" if sys_info_enabled else "False"
        persona['user_profile_enabled'] = "True" if user_profile_enabled else "False"

        # Save 'type' dictionary
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
            'legal_persona': ['jurisdiction', 'area_of_law', 'disclaimer'],
            'financial_advisor': ['investment_goals', 'risk_tolerance', 'time_horizon'],
            'tech_support': ['product_specialization', 'user_expertise_level', 'access_to_logs'],
            'personal_assistant': ['time_zone', 'communication_style', 'access_to_calendar'],
            'therapist': ['therapy_style', 'session_length', 'confidentiality'],
            'travel_guide': ['destination_preferences', 'travel_style', 'interests'],
            'storyteller': ['genre', 'audience_age_group', 'story_length'],
            'game_master': ['game_type', 'difficulty_level', 'theme'],
            'chef': ['cuisine_preferences', 'dietary_restrictions', 'skill_level']
        }

        for persona_type, keys in additional_keys.items():
            if persona['type'][persona_type]['enabled'] == "True":
                for key in keys:
                    if key in persona_type_values.get(persona_type, {}):
                        persona['type'][persona_type][key] = persona_type_values[persona_type][key]
                    else:
                        persona['type'][persona_type].pop(key, None)

        # Save other settings
        persona['provider'] = provider
        persona['model'] = model
        persona['Speech_provider'] = speech_provider
        persona['voice'] = voice

        self.ATLAS.persona_manager.update_persona(persona)
        print(f"Settings for {name} saved!")
        settings_window.destroy()

    def apply_css_styling(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            * { background-color: #2b2b2b; color: white; }
            entry, textview {
                background-color: #1c1c1c;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
                margin: 0;
            }
            entry {
                min-height: 30px;
            }
            entry:focus {
                outline: none;
            }
            textview {
                caret-color: white;
            }
            textview text {
                background-color: #1c1c1c;
                color: white;
                caret-color: white;
            }
            textview text selection {
                background-color: #4a90d9;
                color: white;
            }
            button {
                background-color: #1c1c1c;
                color: white;
                padding: 5px;
                border-radius: 5px;
                margin: 5px;
            }
            label { margin: 5px; }
            /* Updated styling for Gtk.StackSwitcher */
            stackswitcher {
                background-color: #2b2b2b;
                color: white;
            }
            /* Editable textview styles */
            .editable-textview {
                border: none;
            }
            .editable-textview text {
                caret-color: white;
            }
            scrolledwindow {
                border: none;
                background-color: transparent;
            }
            .editable-area {
                border: 1px solid transparent;
                border-radius: 5px;
                background-color: #1c1c1c;
                transition: all 0.3s ease;
            }
            .editable-area-focused {
                border-color: #4a90d9;
            }
            .editable-textview:focus {
                outline: none;
            }
            .info-button {
                background: none;
                border: none;
                color: #4a90d9;
                font-weight: bold;
                font-size: 14px;
                padding: 0;
                margin: 0;
                min-width: 20px;
                min-height: 20px;
            }
            .info-button:hover {
                color: #3a7ab9;
            }
            .info-popup {
                background-color: #2b2b2b;
                border: 1px solid #4a90d9;
                padding: 5px;
                margin: 0;
            }
            .info-popup label {
                background-color: #2b2b2b;
                color: white;
                padding: 0;
                margin: 0;
                font-size: 14px;
            }
        """)
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1
        )

