# UI/Persona_manager/General_Tab/general_tab.py

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango

class InfoPopup(Gtk.Window):
    def __init__(self, parent, text):
        Gtk.Window.__init__(self, title="")
        self.set_transient_for(parent)
        self.set_modal(True)
        self.set_decorated(False)
        self.set_position(Gtk.WindowPosition.MOUSE)

        # Create an EventBox to capture mouse events
        event_box = Gtk.EventBox()
        self.add(event_box)

        # Create a label with no wrapping
        label = Gtk.Label(label=text)
        label.set_line_wrap(False)
        label.set_justify(Gtk.Justification.LEFT)
        label.set_halign(Gtk.Align.START)
        event_box.add(label)

        # Connect events
        event_box.connect("button-press-event", self.on_click)
        self.connect("focus-out-event", self.on_focus_out)

        # Show all widgets
        self.show_all()

        # Force the window to resize to its minimum size
        self.resize(1, 1)

    def on_click(self, widget, event):
        self.destroy()

    def on_focus_out(self, widget, event):
        self.destroy()

class GeneralTab:
    def __init__(self, persona):
        self.persona = persona
        self.persona_type = self.persona.get('type', {})
        self.user_profile_enabled = persona.get("user_profile_enabled", "False") == "True"
        self.sys_info_enabled = persona.get("sys_info_enabled", "False") == "True"
        self.medical_persona_enabled = self.persona_type.get("medical_persona", {}).get("enabled", "False") == "True"
        self.educational_persona = self.persona_type.get("educational_persona", {}).get("enabled", "False") == "True"
        self.fitness_persona_enabled = self.persona_type.get("fitness_persona", {}).get("enabled", "False") == "True"
        self.language_instructor = self.persona_type.get("language_instructor", {}).get("enabled", "False") == "True"
        self.build_ui()

    def build_ui(self):
        outer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer_box.set_margin_start(10)
        outer_box.set_margin_end(10)
        outer_box.set_margin_top(10)
        outer_box.set_margin_bottom(10)

        general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        general_box.set_hexpand(True)
        general_box.set_vexpand(True)
        self.general_box = general_box

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.add(general_box)

        outer_box.pack_start(scrolled_window, True, True, 0)

        # Persona name
        name_box = self.create_labeled_entry_with_info("Persona Name", self.persona.get("name", ""),
                                                       "The name of the persona. This is how you'll identify and select this persona.")
        general_box.pack_start(name_box, False, False, 0)

        # Name meaning
        meaning_box = self.create_labeled_entry_with_info("Name Meaning", self.persona.get("meaning", ""),
                                                          "The meaning or expansion of the persona's name. This helps clarify the persona's purpose or role.")
        general_box.pack_start(meaning_box, False, False, 0)

        # Start Locked TextView
        self.start_view = self.create_textview(editable=False)
        general_box.pack_start(self.create_label_with_info("Start Locked Content", 
                               "This content appears at the start of the persona's system prompt. It typically includes the persona's name and role introduction."), False, False, 0)
        general_box.pack_start(self.start_view, False, False, 0)

        self.update_start_locked()

        self.name_entry.connect("changed", self.update_start_locked)
        self.meaning_entry.connect("changed", self.update_start_locked)

        # Editable Content TextView
        self.editable_view = self.create_textview(editable=True, height=200, css_class="editable-textview")
        general_box.pack_start(self.create_label_with_info("Editable Content", 
                               "This is the main content of the persona's system prompt. You can freely edit this to define the persona's behavior, knowledge, and characteristics."), False, False, 0)
        scrolled_editable = Gtk.ScrolledWindow()
        scrolled_editable.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_editable.set_size_request(-1, 200)
        scrolled_editable.add(self.editable_view)

        frame = Gtk.Frame()
        frame.add(scrolled_editable)
        frame.get_style_context().add_class("editable-area")
        self.frame = frame

        general_box.pack_start(frame, True, True, 0)

        content = self.persona.get("content", {})
        editable_content = content.get("editable_content", "")
        self.editable_view.get_buffer().set_text(editable_content)

        self.editable_view.connect("focus-in-event", self.on_textview_focus_in, frame)
        self.editable_view.connect("focus-out-event", self.on_textview_focus_out, frame)

        # End Locked TextView
        self.end_view = self.create_textview(editable=False)
        general_box.pack_start(self.create_label_with_info("End Locked Content", 
                               "This content appears at the end of the persona's system prompt. It typically includes user-specific information and general instructions."), False, False, 0)
        general_box.pack_start(self.end_view, False, False, 0)

        self.update_end_locked()

        self.sysinfo_view = None
        self.update_sys_info_content()

        self.main_widget = outer_box

    def get_widget(self):
        return self.main_widget

    def create_textview(self, editable=True, height=None, css_class=None):
        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        textview.set_editable(editable)
        textview.set_cursor_visible(editable)
        if css_class:
            textview.get_style_context().add_class(css_class)
        if height:
            textview.set_size_request(-1, height)
        textview.connect("size-allocate", self.on_textview_size_allocate)
        return textview

    def create_labeled_entry_with_info(self, label_text, initial_text, info_text):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        box.pack_start(label, False, False, 0)
        
        entry = Gtk.Entry()
        entry.set_text(initial_text)
        entry.set_width_chars(30)  # Set a fixed width
        box.pack_start(entry, True, True, 0)
        
        info_button = Gtk.Button(label="?")
        info_button.set_relief(Gtk.ReliefStyle.NONE) 
        info_button.get_style_context().add_class("info-button")
        info_button.connect("clicked", self.show_info_popup, info_text)
        box.pack_start(info_button, False, False, 0)
        
        if label_text == "Persona Name":
            self.name_entry = entry
        elif label_text == "Name Meaning":
            self.meaning_entry = entry
        
        return box

    def create_label_with_info(self, label_text, info_text):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        box.pack_start(label, True, True, 0)
        
        info_button = Gtk.Button(label="?")
        info_button.set_relief(Gtk.ReliefStyle.NONE) 
        info_button.get_style_context().add_class("info-button")
        info_button.connect("clicked", self.show_info_popup, info_text)
        box.pack_start(info_button, False, False, 0)
        
        return box

    def show_info_popup(self, widget, info_text):
        popup = InfoPopup(self.main_widget.get_toplevel(), info_text)
        popup.get_style_context().add_class("info-popup")
        popup.show_all()

    def on_entry_changed(self, entry):
        layout = entry.get_layout()
        pixel_width, pixel_height = layout.get_pixel_size()
        entry.set_size_request(-1, pixel_height + 10) 

    def on_textview_size_allocate(self, textview, allocation):
        buffer = textview.get_buffer()
        text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        layout = textview.create_pango_layout(text)
        layout.set_width(allocation.width * Pango.SCALE)
        layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        width, height = layout.get_pixel_size()
        textview.set_size_request(-1, height + 10)  

    def update_start_locked(self, widget=None):
        name = self.name_entry.get_text()
        meaning = self.meaning_entry.get_text()
        if meaning:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}: ({meaning})."
        else:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}."

        buffer = self.start_view.get_buffer()
        current_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if current_text.strip() != start_locked.strip():
            buffer.set_text(start_locked)

    def update_end_locked(self):
        dynamic_parts = []
        if self.user_profile_enabled:
            dynamic_parts.append("User Profile: <<Profile>>")
        if self.medical_persona_enabled:
            dynamic_parts.append("User EMR: <<emr>>")
        if self.educational_persona:
            subject_specialization = self.persona_type.get('educational_persona', {}).get('subject_specialization', 'General')
            education_level = self.persona_type.get('educational_persona', {}).get('education_level', 'High School')
            dynamic_parts.append(f"Subject: {subject_specialization}")
            dynamic_parts.append(f"Level: {education_level}")
            dynamic_parts.append("Provide explanations suitable for the student's level.")
        if self.fitness_persona_enabled:
            fitness_goal = self.persona_type.get('fitness_persona', {}).get('fitness_goal', 'Weight Loss')
            exercise_preference = self.persona_type.get('fitness_persona', {}).get('exercise_preference', 'Gym Workouts')
            dynamic_parts.append(f"Fitness Goal: {fitness_goal}")
            dynamic_parts.append(f"Exercise Preference: {exercise_preference}")
            dynamic_parts.append("Offer motivational support and track progress.")
        if self.language_instructor:
            target_language = self.persona_type.get('language_instructor', {}).get('target_language', 'Spanish')
            proficiency_level = self.persona_type.get('language_instructor', {}).get('proficiency_level', 'Beginner')
            dynamic_parts.append(f"Target Language: {target_language}")
            dynamic_parts.append(f"Proficiency Level: {proficiency_level}")
            dynamic_parts.append("Engage in conversation to practice the target language.")
        if dynamic_parts:
            dynamic_content = " ".join(dynamic_parts)
            dynamic_content += " Clear responses and relevant information are key for a great user experience. Ask for clarity or offer input as needed."
        else:
            dynamic_content = ""

        end_locked = dynamic_content

        buffer = self.end_view.get_buffer()
        current_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if current_text.strip() != end_locked.strip():
            buffer.set_text(end_locked)

    def update_sys_info_content(self):
        if self.sys_info_enabled:
            if not self.sysinfo_view:
                self.sysinfo_view = self.create_textview(editable=False)
                self.sysinfo_view.get_buffer().set_text(
                    "Your current System is <<sysinfo>>. Please make all requests considering these specifications."
                )
                self.general_box.pack_start(self.create_label_with_info("Sysinfo Locked Content",
                                            "This content provides information about the user's system specifications."), False, False, 0)
                self.general_box.pack_start(self.sysinfo_view, False, False, 0)
                self.general_box.show_all()
        else:
            if self.sysinfo_view:
                children = self.general_box.get_children()
                index = children.index(self.sysinfo_view)
                label = children[index - 1]
                self.general_box.remove(label)
                self.general_box.remove(self.sysinfo_view)
                self.sysinfo_view = None

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

    def set_educational_persona(self, enabled):
        self.educational_persona = enabled
        self.update_end_locked()

    def set_fitness_persona_enabled(self, enabled):
        self.fitness_persona_enabled = enabled
        self.update_end_locked()

    def set_language_instructor(self, enabled):
        self.language_instructor = enabled
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
        buffer_start = self.end_view.get_buffer().get_start_iter()
        buffer_end = self.end_view.get_buffer().get_end_iter()
        return self.end_view.get_buffer().get_text(buffer_start, buffer_end, True)
    
    def set_agent_enabled(self, enabled):
        self.persona['type']['Agent'] = {'enabled': str(enabled)}

    def get_sys_info_content(self):
        if self.sysinfo_view:
            buffer_start = self.sysinfo_view.get_buffer().get_start_iter()
            buffer_end = self.sysinfo_view.get_buffer().get_end_iter()
            return self.sysinfo_view.get_buffer().get_text(buffer_start, buffer_end, True)
        return ""

    def set_legal_persona_enabled(self, enabled):
        self.persona['type']['legal_persona'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_financial_advisor_enabled(self, enabled):
        self.persona['type']['financial_advisor'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_tech_support_enabled(self, enabled):
        self.persona['type']['tech_support'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_personal_assistant_enabled(self, enabled):
        self.persona['type']['personal_assistant'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_therapist_enabled(self, enabled):
        self.persona['type']['therapist'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_travel_guide_enabled(self, enabled):
        self.persona['type']['travel_guide'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_storyteller_enabled(self, enabled):
        self.persona['type']['storyteller'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_game_master_enabled(self, enabled):
        self.persona['type']['game_master'] = {'enabled': str(enabled)}
        self.update_end_locked()

    def set_chef_enabled(self, enabled):
        self.persona['type']['chef'] = {'enabled': str(enabled)}
        self.update_end_locked()