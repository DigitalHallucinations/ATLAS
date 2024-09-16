# UI/persona_management.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

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
            label_event_box.connect("button-press-event", lambda widget, event, persona_name=persona_name: self.select_persona(persona_name))

            settings_icon_path = "Icons/settings.png"
            settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)

            settings_event_box = Gtk.EventBox()
            settings_event_box.add(settings_icon)
            settings_event_box.set_margin_start(20)

            settings_event_box.connect("button-press-event", lambda widget, event, persona_name=persona_name: self.open_persona_settings(persona_name))

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
        general_box = self.create_general_tab(persona)
        notebook.append_page(general_box, Gtk.Label(label="General"))

        # Persona Type Tab
        type_box = self.create_persona_type_tab(persona)
        notebook.append_page(type_box, Gtk.Label(label="Persona Type"))

        # Provider and Model Tab
        provider_model_box = self.create_provider_model_tab(persona)
        notebook.append_page(provider_model_box, Gtk.Label(label="Provider & Model"))

        # Speech Provider and Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona)
        notebook.append_page(speech_voice_box, Gtk.Label(label="Speech & Voice"))

        # Save button
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", lambda widget: self.save_persona_settings(
            persona,
            self.name_entry.get_text(),  # Name
            self.meaning_entry.get_text(),  # Meaning
            self.editable_view,             # Editable content view
            provider_model_box.get_children()[0].get_children()[1].get_text(),  # Provider
            provider_model_box.get_children()[1].get_children()[1].get_text(),  # Model
            self.sys_info_switch.get_active(),  # System Info Enabled
            self.agent_switch.get_active(),     # Agent
            self.user_profile_switch.get_active(),  # User Profile Enabled
            self.medical_persona_switch.get_active(),  # Medical Persona
            speech_voice_box.get_children()[0].get_children()[1].get_text(),  # Speech provider
            speech_voice_box.get_children()[1].get_children()[1].get_text(),  # Voice
            settings_window
        ))
        main_vbox.pack_end(save_button, False, False, 0)

        settings_window.show_all()
        self.position_window_next_to_sidebar(settings_window, 500)

    def create_general_tab(self, persona):
        # Create an outer box for margins
        outer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer_box.set_margin_start(10)
        outer_box.set_margin_end(10)
        outer_box.set_margin_top(10)
        outer_box.set_margin_bottom(10)

        # Create a VBox to hold all the general content
        general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        general_box.set_hexpand(True)
        general_box.set_vexpand(True)
        self.general_box = general_box  # Store reference for later use

        # Add the general_box directly to the scrolled window
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.add(general_box)  # Use add instead of add_with_viewport
        scrolled_window.set_size_request(-1, 500)  # Adjust the height as needed

        # Add the scrolled window to the outer box
        outer_box.pack_start(scrolled_window, True, True, 0)

        # Persona name
        name_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        name_label = Gtk.Label(label="Persona Name")
        name_label.set_halign(Gtk.Align.START)
        name_entry = Gtk.Entry()
        name_entry.set_text(persona.get("name", ""))
        name_box.pack_start(name_label, False, False, 0)
        name_box.pack_start(name_entry, True, True, 0)
        general_box.pack_start(name_box, False, False, 0)

        # Name meaning
        meaning_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        meaning_label = Gtk.Label(label="Name Meaning")
        meaning_label.set_halign(Gtk.Align.START)
        meaning_entry = Gtk.Entry()
        meaning_entry.set_text(persona.get("meaning", ""))
        meaning_box.pack_start(meaning_label, False, False, 0)
        meaning_box.pack_start(meaning_entry, True, True, 0)
        general_box.pack_start(meaning_box, False, False, 0)

        # Store entries for later use
        self.name_entry = name_entry
        self.meaning_entry = meaning_entry

        # Function to create a TextView
        def create_textview(editable=True, height=50, css_class=None):
            textview = Gtk.TextView()
            textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            textview.set_editable(editable)
            textview.set_cursor_visible(editable)
            if css_class:
                textview.get_style_context().add_class(css_class)
            textview.set_size_request(-1, height)
            return textview

        # Start Locked TextView (non-editable)
        self.start_view = create_textview(editable=False, height=100)
        general_box.pack_start(Gtk.Label(label="Start Locked Content"), False, False, 0)
        general_box.pack_start(self.start_view, False, False, 0)

        # Update start_locked text
        self.update_start_locked(persona)

        # Connect signals to update start_view in real-time
        self.name_entry.connect("changed", self.on_name_or_meaning_changed)
        self.meaning_entry.connect("changed", self.on_name_or_meaning_changed)

        # Editable Content TextView
        self.editable_view = create_textview(editable=True, height=200, css_class="editable-textview")
        general_box.pack_start(Gtk.Label(label="Editable Content"), False, False, 0)
        scrolled_editable = Gtk.ScrolledWindow()
        scrolled_editable.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_editable.set_size_request(-1, 200)
        scrolled_editable.add(self.editable_view)

        # Use a Frame to capture focus events and show border
        frame = Gtk.Frame()
        frame.add(scrolled_editable)
        frame.get_style_context().add_class("editable-area")
        self.frame = frame  # Store reference to the frame

        general_box.pack_start(frame, True, True, 0)

        # Set the editable content
        content = persona.get("content", {})
        editable_content = content.get("editable_content", "")
        self.editable_view.get_buffer().set_text(editable_content)

        # Connect focus events
        self.editable_view.connect("focus-in-event", self.on_textview_focus_in, frame)
        self.editable_view.connect("focus-out-event", self.on_textview_focus_out, frame)

        # Sysinfo Locked TextView (conditionally displayed)
        sys_info_enabled = persona.get('sys_info_enabled') == "True"
        self.sysinfo_view = None
        if sys_info_enabled:
            self.sysinfo_view = create_textview(editable=False, height=50)
            self.sysinfo_view.get_buffer().set_text("Your current System is <<sysinfo>>. Please make all requests considering these specifications.")
            general_box.pack_start(Gtk.Label(label="Sysinfo Locked Content"), False, False, 0)
            general_box.pack_start(self.sysinfo_view, False, False, 0)

        # End Locked TextView (non-editable)
        self.end_view = create_textview(editable=False, height=100)
        general_box.pack_start(Gtk.Label(label="End Locked Content"), False, False, 0)
        general_box.pack_start(self.end_view, False, False, 0)

        # Update end_locked text
        self.update_end_locked(persona)

        return outer_box


    def update_start_locked(self, persona):
        name = self.name_entry.get_text()
        meaning = self.meaning_entry.get_text()
        if meaning:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}: ({meaning})."
        else:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}."
        self.start_view.get_buffer().set_text(start_locked)

    def update_end_locked(self, persona):
        end_locked = persona.get("content", {}).get("end_locked", "")
        self.end_view.get_buffer().set_text(end_locked)

    def on_name_or_meaning_changed(self, widget):
        self.update_start_locked(None)

    def create_provider_model_tab(self, persona):
        provider_model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        for label, key in [("API Provider", "provider"), ("Model", "model")]:
            entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            entry_label = Gtk.Label(label=label)
            entry = Gtk.Entry()
            entry.set_text(persona.get(key, ""))
            entry_box.pack_start(entry_label, False, False, 0)
            entry_box.pack_start(entry, True, True, 0)
            provider_model_box.pack_start(entry_box, False, False, 0)

        return provider_model_box

    def create_persona_type_tab(self, persona):
        type_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Switches
        self.sys_info_switch = Gtk.Switch()
        self.sys_info_switch.set_active(persona.get("sys_info_enabled", "False") == "True")
        self.agent_switch = Gtk.Switch()
        self.agent_switch.set_active(persona.get("Agent", "False") == "True")
        self.user_profile_switch = Gtk.Switch()
        self.user_profile_switch.set_active(persona.get("user_profile_enabled", "False") == "True")
        self.medical_persona_switch = Gtk.Switch()
        self.medical_persona_switch.set_active(persona.get("medical_persona", "False") == "True")

        switches = [
            ("System Info Enabled", self.sys_info_switch),
            ("Agent", self.agent_switch),
            ("User Profile Enabled", self.user_profile_switch),
            ("Medical Persona", self.medical_persona_switch),
        ]

        for label, switch in switches:
            switch_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            switch_label = Gtk.Label(label=label)
            switch_box.pack_start(switch_label, False, False, 0)
            switch_box.pack_end(switch, False, False, 0)
            type_box.pack_start(switch_box, False, False, 0)

        # Connect signals to update UI
        self.sys_info_switch.connect("notify::active", self.on_sys_info_switch_toggled)
        self.user_profile_switch.connect("notify::active", self.on_user_profile_switch_toggled)
        self.medical_persona_switch.connect("notify::active", self.on_medical_persona_switch_toggled)

        return type_box

    def create_speech_voice_tab(self, persona):
        speech_voice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        for label, key in [("Speech Provider", "Speech_provider"), ("Voice", "voice")]:
            entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            entry_label = Gtk.Label(label=label)
            entry = Gtk.Entry()
            entry.set_text(persona.get(key, ""))
            entry_box.pack_start(entry_label, False, False, 0)
            entry_box.pack_start(entry, True, True, 0)
            speech_voice_box.pack_start(entry_box, False, False, 0)

        return speech_voice_box

    def on_sys_info_switch_toggled(self, switch, gparam):
        sys_info_enabled = switch.get_active()
        if sys_info_enabled:
            if not self.sysinfo_view:
                self.sysinfo_view = self.create_textview(editable=False, height=50)
                self.sysinfo_view.get_buffer().set_text("Your current System is <<sysinfo>>. Please make all requests considering these specifications.")
                self.general_box.pack_start(Gtk.Label(label="Sysinfo Locked Content"), False, False, 0)
                self.general_box.pack_start(self.sysinfo_view, False, False, 0)
                self.general_box.show_all()
        else:
            if self.sysinfo_view:
                self.general_box.remove(self.sysinfo_view)
                self.sysinfo_view = None

    def on_user_profile_switch_toggled(self, switch, gparam):
        self.update_end_locked_content()

    def on_medical_persona_switch_toggled(self, switch, gparam):
        self.update_end_locked_content()

    def update_end_locked_content(self):
        # Update end_locked content based on switches
        end_locked_parts = []
        if self.user_profile_switch.get_active():
            end_locked_parts.append("User Profile: <<Profile>>")
        if self.medical_persona_switch.get_active():
            end_locked_parts.append("User EMR: <<emr>>")
        if end_locked_parts:
            end_locked = " ".join(end_locked_parts)
            end_locked += " Clear responses and relevant information are key for a great user experience. Ask for clarity or offer input as needed."
        else:
            end_locked = ""

        # Update the end_locked TextView
        self.end_view.get_buffer().set_text(end_locked)

    def save_persona_settings(self, persona, name, meaning, editable_view, provider, model,
                              sys_info_enabled, agent, user_profile_enabled, medical_persona,
                              speech_provider, voice, settings_window):
        persona['name'] = name
        persona['meaning'] = meaning

        # Get the editable content
        buffer_start = editable_view.get_buffer().get_start_iter()
        buffer_end = editable_view.get_buffer().get_end_iter()
        editable_content = editable_view.get_buffer().get_text(buffer_start, buffer_end, True)

        # Update content parts
        content = persona.get('content', {})
        content['start_locked'] = self.start_view.get_buffer().get_text(
            self.start_view.get_buffer().get_start_iter(),
            self.start_view.get_buffer().get_end_iter(),
            True
        )
        content['editable_content'] = editable_content
        content['end_locked'] = self.end_view.get_buffer().get_text(
            self.end_view.get_buffer().get_start_iter(),
            self.end_view.get_buffer().get_end_iter(),
            True
        )
        persona['content'] = content

        persona['provider'] = provider
        persona['model'] = model
        persona['sys_info_enabled'] = "True" if sys_info_enabled else "False"
        persona['Agent'] = "True" if agent else "False"
        persona['user_profile_enabled'] = "True" if user_profile_enabled else "False"
        persona['medical_persona'] = "True" if medical_persona else "False"
        persona['Speech_provider'] = speech_provider
        persona['voice'] = voice

        self.ATLAS.persona_manager.update_persona(persona)
        print(f"Settings for {name} saved!")
        settings_window.destroy()

    def on_textview_focus_in(self, textview, event, frame):
        frame.get_style_context().add_class("editable-area-focused")
        return False

    def on_textview_focus_out(self, textview, event, frame):
        frame.get_style_context().remove_class("editable-area-focused")
        return False

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

    def create_textview(self, editable=True, height=50, css_class=None):
        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        textview.set_editable(editable)
        textview.set_cursor_visible(editable)
        if css_class:
            textview.get_style_context().add_class(css_class)
        textview.set_size_request(-1, height)
        return textview
