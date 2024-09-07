# UI/sidebar.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
from ATLAS.ATLAS import ATLAS

class Sidebar(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Sidebar")
        
        # ATLAS instance 
        self.ATLAS = ATLAS()

        # Set window properties to span the full screen height
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        monitor_height = monitor_geometry.height

        self.set_default_size(50, monitor_height)
        self.set_decorated(False)
        self.set_keep_above(False)
        self.stick()

        # Set up CSS for styling
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background-color: #2b2b2b;
            }
            label {
                color: white;
                padding: 5px;
                font-size: 14px;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.add(self.box)

        top_spacer = Gtk.Box()
        self.box.pack_start(top_spacer, False, False, 5)

        # Replace buttons with event boxes containing icons
        self.create_icon("Icons/providers.png", self.show_providers_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42)

        self.box.pack_start(Gtk.Box(), True, True, 0)
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

        bottom_spacer = Gtk.Box()
        self.box.pack_start(bottom_spacer, False, False, 5)

        self.position_sidebar()

    def create_icon(self, icon_path, callback, icon_size):
        # Create an event box to capture click events
        event_box = Gtk.EventBox()

        # Load the icon image
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, icon_size, icon_size, True)
        icon = Gtk.Image.new_from_pixbuf(pixbuf)

        # Add the image to the event box
        event_box.add(icon)

        # Connect the click event to the event box
        event_box.connect("button-press-event", lambda widget, event: callback(widget))

        # Pack the event box into the vertical box
        self.box.pack_start(event_box, False, False, 0)

    def position_sidebar(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        geometry = monitor.get_geometry()
        self.move(geometry.width - self.get_size().width, 0)

    def position_window_next_to_sidebar(self, window, window_width):
        """Helper function to place a window next to the sidebar."""
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width
        
        # Calculate the X position to place the window next to the sidebar
        window_x = screen_width - 50 - 10 - window_width  # 50px sidebar, 10px gap
        window.move(window_x, 0)  # Y = 0 (top of the screen)

    def show_providers_menu(self, widget):
        print("Providers menu clicked")

    def handle_history_button(self, widget):
        self.ATLAS.log_history()

    def show_chat_page(self, widget):
        print("Chat page clicked")

    def select_persona(self, persona):
        """Handle persona selection and print personalized system prompt."""
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.persona_manager.current_system_prompt}")

        def show_persona_settings(self, persona):
            print(f"Settings for {persona}")

    def show_persona_menu(self, widget):
        # Create persona window
        self.persona_window = Gtk.Window(title="Select Persona")  # Save reference to the persona window
        self.persona_window.set_default_size(150, 600)
        self.persona_window.set_keep_above(True)

        # Use helper method to position the window
        self.position_window_next_to_sidebar(self.persona_window, 150)

        # Create box to hold persona labels
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.persona_window.add(box)

        persona_names = self.ATLAS.get_persona_names()

        for persona_name in persona_names:
            # Create a horizontal box to hold the label and the settings icon
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            # Create and configure the persona label
            label = Gtk.Label(label=persona_name)
            label.set_xalign(0.0)  # Align the label to the left
            label.set_yalign(0.5)

            # Add a click event to load the persona when the label is clicked
            label_event_box = Gtk.EventBox()
            label_event_box.add(label)
            label_event_box.connect("button-press-event", lambda widget, event, persona_name=persona_name: self.select_persona(persona_name))

            # Create the settings icon
            settings_icon_path = "Icons/settings.png"
            settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)

            # Create an event box for the settings icon
            settings_event_box = Gtk.EventBox()
            settings_event_box.add(settings_icon)
            settings_event_box.set_margin_start(20)

            # Connect the settings icon to open the settings for the full persona object and close the persona window
            settings_event_box.connect("button-press-event", lambda widget, event, persona_name=persona_name: self.open_persona_settings(persona_name))

            # Add the label and settings icon to the horizontal box
            hbox.pack_start(label_event_box, True, True, 0)  # Let the label expand
            hbox.pack_end(settings_event_box, False, False, 0)  # Pack the icon to the right

            # Add the horizontal box to the main box
            box.pack_start(hbox, False, False, 0)

        self.persona_window.show_all()

    def open_persona_settings(self, persona_name):
        # Close the persona selection window
        if self.persona_window:
            self.persona_window.close()

        # Load the full persona data
        persona = self.ATLAS.persona_manager.get_persona(persona_name)

        # Open the settings window for the selected persona
        self.show_persona_settings(persona)

    def show_persona_settings(self, persona):
        # Create settings window for the persona
        settings_window = Gtk.Window(title=f"Settings for {persona.get('name', 'Unknown Persona')}")
        settings_window.set_default_size(400, 500)
        settings_window.set_keep_above(True)

        # Use helper method to position the window
        self.position_window_next_to_sidebar(settings_window, 400)

        # Create a notebook for tabbed interface
        notebook = Gtk.Notebook()
        settings_window.add(notebook)

        # 1. Name and Basic Info Tab
        basic_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        name_label = Gtk.Label(label="Persona Name")
        name_entry = Gtk.Entry()
        name_entry.set_text(persona.get("name", ""))
        basic_info_box.pack_start(name_label, False, False, 0)
        basic_info_box.pack_start(name_entry, False, False, 0)

        # Add the basic info tab
        notebook.append_page(basic_info_box, Gtk.Label(label="Basic Info"))

        # 2. Content Tab
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        content_label = Gtk.Label(label="Persona Content")
        content_textview = Gtk.TextView()
        content_buffer = content_textview.get_buffer()

        # Convert and display newlines properly
        persona_content = persona.get("content", "").replace("\\n", "\n")
        content_buffer.set_text(persona_content)
        content_textview.set_wrap_mode(Gtk.WrapMode.WORD)
        content_box.pack_start(content_label, False, False, 0)
        content_box.pack_start(content_textview, True, True, 0)

        # Add the content tab
        notebook.append_page(content_box, Gtk.Label(label="Content"))

        # 3. Message Tab
        message_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        message_label = Gtk.Label(label="Persona Message")
        message_entry = Gtk.Entry()
        message_entry.set_text(persona.get("message", ""))
        message_box.pack_start(message_label, False, False, 0)
        message_box.pack_start(message_entry, False, False, 0)

        # Add the message tab
        notebook.append_page(message_box, Gtk.Label(label="Message"))

        # 4. Provider Tab
        provider_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        provider_label = Gtk.Label(label="API Provider")
        provider_entry = Gtk.Entry()
        provider_entry.set_text(persona.get("provider", ""))
        provider_box.pack_start(provider_label, False, False, 0)
        provider_box.pack_start(provider_entry, False, False, 0)

        # Add the provider tab
        notebook.append_page(provider_box, Gtk.Label(label="Provider"))

        # 5. Model Tab
        model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        model_label = Gtk.Label(label="Model")
        model_entry = Gtk.Entry()
        model_entry.set_text(persona.get("model", ""))
        model_box.pack_start(model_label, False, False, 0)
        model_box.pack_start(model_entry, False, False, 0)

        # Add the model tab
        notebook.append_page(model_box, Gtk.Label(label="Model"))

        # 6. System Admin Persona Tab
        sys_admin_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sys_admin_label = Gtk.Label(label="System Admin Persona")
        sys_admin_switch = Gtk.Switch()
        sys_admin_switch.set_active(persona.get("Sys_admin_persona", "False") == "True")
        sys_admin_box.pack_start(sys_admin_label, False, False, 0)
        sys_admin_box.pack_start(sys_admin_switch, False, False, 0)

        # Add the system admin persona tab
        notebook.append_page(sys_admin_box, Gtk.Label(label="Sys Admin"))

        # 7. Medical Persona Tab
        medical_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        medical_label = Gtk.Label(label="Medical Persona")
        medical_switch = Gtk.Switch()
        medical_switch.set_active(persona.get("medical_persona", "False") == "True")
        medical_box.pack_start(medical_label, False, False, 0)
        medical_box.pack_start(medical_switch, False, False, 0)

        # Add the medical persona tab
        notebook.append_page(medical_box, Gtk.Label(label="Medical Persona"))

        # 8. Speech Provider Tab
        speech_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        speech_label = Gtk.Label(label="Speech Provider")
        speech_entry = Gtk.Entry()
        speech_entry.set_text(persona.get("Speech_provider", ""))
        speech_box.pack_start(speech_label, False, False, 0)
        speech_box.pack_start(speech_entry, False, False, 0)

        # Add the speech provider tab
        notebook.append_page(speech_box, Gtk.Label(label="Speech Provider"))

        # 9. Voice Tab
        voice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        voice_label = Gtk.Label(label="Voice")
        voice_entry = Gtk.Entry()
        voice_entry.set_text(persona.get("voice", ""))
        voice_box.pack_start(voice_label, False, False, 0)
        voice_box.pack_start(voice_entry, False, False, 0)

        # Add the voice tab
        notebook.append_page(voice_box, Gtk.Label(label="Voice"))

        # Save button
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", lambda widget: self.save_persona_settings(
            persona,
            name_entry.get_text(),
            content_buffer.get_text(content_buffer.get_start_iter(), content_buffer.get_end_iter(), True),
            message_entry.get_text(),
            provider_entry.get_text(),
            model_entry.get_text(),
            sys_admin_switch.get_active(),
            medical_switch.get_active(),
            speech_entry.get_text(),
            voice_entry.get_text()
        ))

        vbox_save = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_save.pack_start(save_button, False, False, 0)
        notebook.append_page(vbox_save, Gtk.Label(label="Save"))

        settings_window.show_all()


    def save_persona_settings(self, persona, name, content, model):
        # Update the persona with the new settings
        persona['name'] = name
        persona['content'] = content
        persona['OpenAI_model'] = model
        
        # Save updated persona to the file or persona manager
        self.ATLAS.persona_manager.update_persona(persona)
        
        print(f"Settings for {name} saved!")

    def show_settings_page(self, widget):
        # Create settings window
        settings_window = Gtk.Window(title="Settings")
        settings_window.set_default_size(300, 400)
        settings_window.set_keep_above(True)
        
        # Use helper method to position the window
        self.position_window_next_to_sidebar(settings_window, 300)
        
        settings_window.show_all()

    def close_application(self, widget):
        print("Closing application")
        Gtk.main_quit()
