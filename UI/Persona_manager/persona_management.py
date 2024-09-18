# UI/persona_management.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

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
        self.persona_window.set_keep_above(True)

        self.position_window_next_to_sidebar(self.persona_window, 150)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.persona_window.add(box)

        persona_names = self.ATLAS.get_persona_names()

        for persona_name in persona_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            label = Gtk.Label(label=persona_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)

            label_event_box = Gtk.EventBox()
            label_event_box.add(label)
            label_event_box.connect(
                "button-press-event",
                lambda widget, event, persona_name=persona_name: self.select_persona(persona_name)
            )

            settings_icon_path = "Icons/settings.png"
            settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)

            settings_event_box = Gtk.EventBox()
            settings_event_box.add(settings_icon)
            settings_event_box.set_margin_start(20)

            settings_event_box.connect(
                "button-press-event",
                lambda widget, event, persona_name=persona_name: self.open_persona_settings(persona_name)
            )

            hbox.pack_start(label_event_box, True, True, 0)
            hbox.pack_end(settings_event_box, False, False, 0)

            box.pack_start(hbox, False, False, 0)

        self.persona_window.show_all()

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
        settings_window.set_default_size(500, 600)
        settings_window.set_keep_above(True)

        self.apply_css_styling()

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        settings_window.add(main_vbox)

        notebook = Gtk.Notebook()
        main_vbox.pack_start(notebook, True, True, 0)

        # General Tab
        self.general_tab = GeneralTab(persona)
        general_box = self.general_tab.get_widget()
        notebook.append_page(general_box, Gtk.Label(label="General"))

        # Persona Type Tab
        self.persona_type_tab = PersonaTypeTab(persona, self.general_tab)
        type_box = self.persona_type_tab.get_widget()
        notebook.append_page(type_box, Gtk.Label(label="Persona Type"))

        # Provider and Model Tab
        self.provider_model_box = self.create_provider_model_tab(persona)
        notebook.append_page(self.provider_model_box, Gtk.Label(label="Provider & Model"))

        # Speech Provider and Voice Tab
        self.speech_voice_box = self.create_speech_voice_tab(persona)
        notebook.append_page(self.speech_voice_box, Gtk.Label(label="Speech & Voice"))

        # Save button
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", lambda widget: self.save_persona_settings(persona, settings_window))
        main_vbox.pack_end(save_button, False, False, 0)

        settings_window.show_all()
        self.position_window_next_to_sidebar(settings_window, 500)

    def get_entry_text(self, box, child_index, entry_index):
        entry_box = box.get_children()[child_index]
        entry = entry_box.get_children()[entry_index]
        return entry.get_text()

    # Additional methods for provider and speech settings
    def create_provider_model_tab(self, persona):
        provider_model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Provider
        provider_label = Gtk.Label(label="Provider")
        provider_entry = Gtk.Entry()
        provider_entry.set_text(persona.get("provider", "openai"))
        provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        provider_box.pack_start(provider_label, False, False, 0)
        provider_box.pack_start(provider_entry, True, True, 0)
        provider_model_box.pack_start(provider_box, False, False, 0)

        # Model
        model_label = Gtk.Label(label="Model")
        model_entry = Gtk.Entry()
        model_entry.set_text(persona.get("model", "gpt-4"))
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        model_box.pack_start(model_label, False, False, 0)
        model_box.pack_start(model_entry, True, True, 0)
        provider_model_box.pack_start(model_box, False, False, 0)

        return provider_model_box

    def create_speech_voice_tab(self, persona):
        speech_voice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Speech Provider
        speech_provider_label = Gtk.Label(label="Speech Provider")
        speech_provider_entry = Gtk.Entry()
        speech_provider_entry.set_text(persona.get("Speech_provider", "11labs"))
        speech_provider_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        speech_provider_box.pack_start(speech_provider_label, False, False, 0)
        speech_provider_box.pack_start(speech_provider_entry, True, True, 0)
        speech_voice_box.pack_start(speech_provider_box, False, False, 0)

        # Voice
        voice_label = Gtk.Label(label="Voice")
        voice_entry = Gtk.Entry()
        voice_entry.set_text(persona.get("voice", "jack"))
        voice_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        voice_box.pack_start(voice_label, False, False, 0)
        voice_box.pack_start(voice_entry, True, True, 0)
        speech_voice_box.pack_start(voice_box, False, False, 0)

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
        provider = self.get_entry_text(self.provider_model_box, 0, 1)  # Provider
        model = self.get_entry_text(self.provider_model_box, 1, 1)  # Model

        # Get values from speech_voice_box
        speech_provider = self.get_entry_text(self.speech_voice_box, 0, 1)  # Speech provider
        voice = self.get_entry_text(self.speech_voice_box, 1, 1)  # Voice

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
        for key in ['Agent', 'medical_persona', 'educational_persona', 'fitness_persona', 'language_instructor']:
            persona['type'][key] = "True" if persona_type_values.get(key, False) else "False"

        # Save additional options under 'type'
        additional_keys = [
            'subject_specialization', 'education_level', 'teaching_style',
            'fitness_goal', 'exercise_preference',
            'target_language', 'proficiency_level'
        ]
        for key in additional_keys:
            if key in persona_type_values:
                persona['type'][key] = persona_type_values[key]
            else:
                persona['type'].pop(key, None)

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
            notebook tab { background-color: #2b2b2b; color: white; }
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
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1
        )

    def position_window_next_to_sidebar(self, window, window_width):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width

        window_x = screen_width - 50 - 10 - window_width
        window.move(window_x, 0)
