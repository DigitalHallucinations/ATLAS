# UI/Persona_manager/General_Tab/general_tab.py

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


class GeneralTab:
    def __init__(self, persona):
        self.persona = persona
        # Initialize flags for user profile, medical persona, and system info
        self.user_profile_enabled = persona.get("user_profile_enabled", "False") == "True"
        self.medical_persona_enabled = persona.get("medical_persona", "False") == "True"
        self.sys_info_enabled = persona.get("sys_info_enabled", "False") == "True"
        self.educational_persona_enabled = persona.get("educational_persona_enabled", "False") == "True"
        self.fitness_persona_enabled = persona.get("fitness_trainer_enabled", "False") == "True"
        self.language_practice_enabled = persona.get("language_practice_enabled", "False") == "True"
        self.build_ui()

    def build_ui(self):
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
        scrolled_window.add(general_box)

        # Add the scrolled window to the outer box
        outer_box.pack_start(scrolled_window, True, True, 0)

        # Persona name
        name_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        name_label = Gtk.Label(label="Persona Name")
        name_label.set_halign(Gtk.Align.START)
        name_entry = Gtk.Entry()
        name_entry.set_text(self.persona.get("name", ""))
        name_box.pack_start(name_label, False, False, 0)
        name_box.pack_start(name_entry, True, True, 0)
        general_box.pack_start(name_box, False, False, 0)

        # Name meaning
        meaning_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        meaning_label = Gtk.Label(label="Name Meaning")
        meaning_label.set_halign(Gtk.Align.START)
        meaning_entry = Gtk.Entry()
        meaning_entry.set_text(self.persona.get("meaning", ""))
        meaning_box.pack_start(meaning_label, False, False, 0)
        meaning_box.pack_start(meaning_entry, True, True, 0)
        general_box.pack_start(meaning_box, False, False, 0)

        # Store entries for later use
        self.name_entry = name_entry
        self.meaning_entry = meaning_entry

        # Start Locked TextView (non-editable)
        self.start_view = self.create_textview(editable=False, height=100)
        general_box.pack_start(Gtk.Label(label="Start Locked Content"), False, False, 0)
        general_box.pack_start(self.start_view, False, False, 0)

        # Update start_locked text
        self.update_start_locked()

        # Connect signals to update start_view in real-time
        self.name_entry.connect("changed", self.update_start_locked)
        self.meaning_entry.connect("changed", self.update_start_locked)

        # Editable Content TextView
        self.editable_view = self.create_textview(editable=True, height=200, css_class="editable-textview")
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
        content = self.persona.get("content", {})
        editable_content = content.get("editable_content", "")
        self.editable_view.get_buffer().set_text(editable_content)

        # Connect focus events
        self.editable_view.connect("focus-in-event", self.on_textview_focus_in, frame)
        self.editable_view.connect("focus-out-event", self.on_textview_focus_out, frame)

        # End Locked TextView (non-editable)
        self.end_view = self.create_textview(editable=False, height=100)
        self.end_view.set_vexpand(True)  # Allow vertical expansion
        general_box.pack_start(Gtk.Label(label="End Locked Content"), False, False, 0)
        general_box.pack_start(self.end_view, False, False, 0)

        # Update end_locked text based on switches
        self.update_end_locked()

        # Sysinfo Locked TextView (conditionally displayed)
        self.sysinfo_view = None
        self.update_sys_info_content()

        # Store the outer_box as self.main_widget
        self.main_widget = outer_box

    def get_widget(self):
        return self.main_widget

    def create_textview(self, editable=True, height=50, css_class=None):
        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        textview.set_editable(editable)
        textview.set_cursor_visible(editable)
        if css_class:
            textview.get_style_context().add_class(css_class)
        textview.set_size_request(-1, height)
        return textview

    def update_start_locked(self, widget=None):
        name = self.name_entry.get_text()
        meaning = self.meaning_entry.get_text()
        if meaning:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}: ({meaning})."
        else:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}."

        # Update the text if it has changed to avoid duplicate updates
        buffer = self.start_view.get_buffer()
        current_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if current_text.strip() != start_locked.strip():
            buffer.set_text(start_locked)

        # Dynamically resize the TextView based on the content
        self.resize_textview_to_content(self.start_view)

    def update_end_locked(self):
        # Generate end_locked content based on switches
        dynamic_parts = []
        if self.user_profile_enabled:
            dynamic_parts.append("User Profile: <<Profile>>")
        if self.medical_persona_enabled:
            dynamic_parts.append("User EMR: <<emr>>")
        if self.educational_persona_enabled:
            dynamic_parts.append(f"Subject: {self.persona.get('subject_specialization', 'General')}")
            dynamic_parts.append(f"Level: {self.persona.get('education_level', 'High School')}")
            dynamic_parts.append("Provide explanations suitable for the student's level.")
        if self.fitness_persona_enabled:
            dynamic_parts.append(f"Fitness Goal: {self.persona.get('fitness_goal', 'Weight Loss')}")
            dynamic_parts.append(f"Exercise Preference: {self.persona.get('exercise_preference', 'Gym Workouts')}")
            dynamic_parts.append("Offer motivational support and track progress.")
        if self.language_practice_enabled:
            dynamic_parts.append(f"Target Language: {self.persona.get('target_language', 'Spanish')}")
            dynamic_parts.append(f"Proficiency Level: {self.persona.get('proficiency_level', 'Beginner')}")
            dynamic_parts.append("Engage in conversation to practice the target language.")
        if dynamic_parts:
            dynamic_content = " ".join(dynamic_parts)
            dynamic_content += " Clear responses and relevant information are key for a great user experience. Ask for clarity or offer input as needed."
        else:
            dynamic_content = ""

        end_locked = dynamic_content

        # Avoid updating multiple times by checking if content is already set
        buffer_start = self.end_view.get_buffer().get_start_iter()
        buffer_end = self.end_view.get_buffer().get_end_iter()
        current_text = self.end_view.get_buffer().get_text(buffer_start, buffer_end, True)

        if current_text.strip() == end_locked.strip():
            # If content is already set correctly, return early
            return

        # Update the end_locked TextView
        self.end_view.get_buffer().set_text(end_locked)

        # Dynamically resize the TextView based on the content
        self.resize_textview_to_content(self.end_view)

    def update_sys_info_content(self):
        if self.sys_info_enabled:
            if not self.sysinfo_view:
                self.sysinfo_view = self.create_textview(editable=False, height=50)
                self.sysinfo_view.get_buffer().set_text(
                    "Your current System is <<sysinfo>>. Please make all requests considering these specifications."
                )
                self.general_box.pack_start(Gtk.Label(label="Sysinfo Locked Content"), False, False, 0)
                self.general_box.pack_start(self.sysinfo_view, False, False, 0)
                self.general_box.show_all()
        else:
            if self.sysinfo_view:
                # Remove the label and the sysinfo_view
                children = self.general_box.get_children()
                index = children.index(self.sysinfo_view)
                label = children[index - 1]  # Assuming the label is added just before sysinfo_view
                self.general_box.remove(label)
                self.general_box.remove(self.sysinfo_view)
                self.sysinfo_view = None

    def resize_textview_to_content(self, textview):
        buffer = textview.get_buffer()
        line_count = buffer.get_line_count()

        # Set the height of the TextView based on the line count
        height_per_line = 20  # Adjust this value based on font size
        new_height = max(50, line_count * height_per_line)  # Minimum height of 50 pixels

        # Apply the calculated height to the TextView
        textview.set_size_request(-1, new_height)

    def on_textview_focus_in(self, textview, event, frame):
        frame.get_style_context().add_class("editable-area-focused")
        return False

    def on_textview_focus_out(self, textview, event, frame):
        frame.get_style_context().remove_class("editable-area-focused")
        return False

    def set_sys_info_enabled(self, enabled):
        self.sys_info_enabled = enabled
        self.update_sys_info_content()

    def set_user_profile_enabled(self, enabled):
        self.user_profile_enabled = enabled
        self.update_end_locked()

    def set_medical_persona_enabled(self, enabled):
        self.medical_persona_enabled = enabled
        self.update_end_locked()

    def set_educational_persona_enabled(self, enabled):
        self.educational_persona_enabled = enabled
        self.update_end_locked()

    def set_fitness_persona_enabled(self, enabled):
        self.fitness_persona_enabled = enabled
        self.update_end_locked()

    def set_language_practice_enabled(self, enabled):
        self.language_practice_enabled = enabled
        self.update_end_locked()

    def get_name(self):
        return self.name_entry.get_text()

    def get_meaning(self):
        return self.meaning_entry.get_text()

    def get_editable_content(self):
        buffer_start = self.editable_view.get_buffer().get_start_iter()
        buffer_end = self.editable_view.get_buffer().get_end_iter()
        return self.editable_view.get_buffer().get_text(buffer_start, buffer_end, True)

    def get_start_locked(self):
        buffer_start = self.start_view.get_buffer().get_start_iter()
        buffer_end = self.start_view.get_buffer().get_end_iter()
        return self.start_view.get_buffer().get_text(buffer_start, buffer_end, True)

    def get_end_locked(self):
        # Return the dynamic content
        buffer_start = self.end_view.get_buffer().get_start_iter()
        buffer_end = self.end_view.get_buffer().get_end_iter()
        return self.end_view.get_buffer().get_text(buffer_start, buffer_end, True)
