# UI/Persona_manager/General_Tab/general_tab.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, Pango


class InfoPopup(Gtk.Window):
    def __init__(self, parent, text):
        super().__init__(title="")
        self.set_transient_for(parent)
        self.set_modal(True)
        self.set_decorated(False)

        # Create a label with no wrapping
        label = Gtk.Label(label=text)
        label.set_wrap(False)
        label.set_justify(Gtk.Justification.LEFT)
        label.set_halign(Gtk.Align.START)
        label.set_tooltip_text(text)
        self.set_child(label)

        # Close on click
        gesture_click = Gtk.GestureClick()
        gesture_click.connect("pressed", self.on_click)
        self.add_controller(gesture_click)

        # Close when focus is lost
        self.connect("focus-out-event", self.on_focus_out)

        # Present the window (GTK4)
        self.present()

        # Let GTK compute a minimal size
        self.set_default_size(-1, -1)

    def on_click(self, gesture, n_press, x, y):
        self.destroy()

    def on_focus_out(self, *args):
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

        # Keep references for dynamic widgets/labels
        self.sysinfo_view = None
        self.sysinfo_label = None
        self.counter_label = None

        self.build_ui()

    # ----------------------------- UI build -----------------------------

    def build_ui(self):
        outer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer_box.set_margin_start(10)
        outer_box.set_margin_end(10)
        outer_box.set_margin_top(10)
        outer_box.set_margin_bottom(10)

        general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        general_box.set_hexpand(True)
        general_box.set_vexpand(True)
        general_box.set_tooltip_text(
            "Configure the core prompt blocks for this persona. "
            "Locked sections are auto-generated; Editable Content is freeform."
        )
        self.general_box = general_box

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_child(general_box)
        outer_box.append(scrolled_window)

        # Persona name
        name_box = self.create_labeled_entry_with_info(
            "Persona Name",
            self.persona.get("name", ""),
            "The display name of this persona (used in menus and prompts)."
        )
        general_box.append(name_box)

        # Name meaning
        meaning_box = self.create_labeled_entry_with_info(
            "Name Meaning",
            self.persona.get("meaning", ""),
            "A short explanation or expansion that clarifies the persona’s purpose."
        )
        general_box.append(meaning_box)

        # Start Locked TextView
        start_locked_label_row = self.create_label_with_info(
            "Start Locked Content",
            "Auto-generated preamble that appears at the top of the system prompt."
        )
        general_box.append(start_locked_label_row)

        # Copy button for Start Locked
        start_copy_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        start_copy_btn = Gtk.Button(label="Copy")
        start_copy_btn.set_tooltip_text("Copy the Start Locked content to clipboard.")
        start_copy_btn.add_css_class("flat")
        start_copy_btn.connect("clicked", lambda _b: self.copy_start_locked())
        start_copy_row.append(start_copy_btn)
        general_box.append(start_copy_row)

        self.start_view = self.create_textview(editable=False)
        self.start_view.set_tooltip_text("Read-only preface generated from Name/Meaning.")
        general_box.append(self.start_view)

        self.update_start_locked()
        self.name_entry.connect("changed", self.update_start_locked)
        self.meaning_entry.connect("changed", self.update_start_locked)

        # Editable Content (with mini-toolbar)
        editable_header = self.create_label_with_info(
            "Editable Content",
            "Freeform instructions and traits that define the persona’s voice and behavior."
        )
        general_box.append(editable_header)

        # Toolbar: Clear | Insert template | counter
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        clear_btn = Gtk.Button(label="Clear")
        clear_btn.set_tooltip_text("Clear the editable content.")
        clear_btn.add_css_class("flat")
        clear_btn.connect("clicked", lambda _b: self._set_editable_text(""))

        template_btn = Gtk.Button(label="Insert template")
        template_btn.set_tooltip_text("Insert a starter template for common persona traits.")
        template_btn.add_css_class("flat")
        template_btn.connect("clicked", self._insert_default_template)

        self.counter_label = Gtk.Label(label="0 chars • 0 words")
        self.counter_label.set_xalign(1.0)
        self.counter_label.set_tooltip_text("Live character and word count for the editable content.")
        toolbar.append(clear_btn)
        toolbar.append(template_btn)
        toolbar.append(self._spacer())
        toolbar.append(self.counter_label)
        general_box.append(toolbar)

        self.editable_view = self.create_textview(editable=True, height=200, css_class="editable-textview")
        self.editable_view.set_tooltip_text(
            "Write the main persona instructions here. Keep it concise and specific."
        )
        scrolled_editable = Gtk.ScrolledWindow()
        scrolled_editable.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_editable.set_size_request(-1, 200)
        scrolled_editable.set_child(self.editable_view)

        frame = Gtk.Frame()
        frame.set_child(scrolled_editable)
        frame.get_style_context().add_class("editable-area")
        self.frame = frame
        general_box.append(frame)

        content = self.persona.get("content", {})
        editable_content = content.get("editable_content", "")
        self._set_editable_text(editable_content, update_counter=False)
        self._update_counter()  # initialize counter once

        # Focus styling (GTK4 EventControllerFocus)
        focus_controller = Gtk.EventControllerFocus()
        focus_controller.connect("enter", self.on_textview_focus_in)
        focus_controller.connect("leave", self.on_textview_focus_out)
        self.editable_view.add_controller(focus_controller)

        # Track edits for counter updates
        buf = self.editable_view.get_buffer()
        buf.connect('changed', lambda *_args: self._update_counter())

        # End Locked TextView
        end_locked_label_row = self.create_label_with_info(
            "End Locked Content",
            "Auto-generated closing block built from toggled persona capabilities."
        )
        general_box.append(end_locked_label_row)

        # Copy button for End Locked
        end_copy_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        end_copy_btn = Gtk.Button(label="Copy")
        end_copy_btn.set_tooltip_text("Copy the End Locked content to clipboard.")
        end_copy_btn.add_css_class("flat")
        end_copy_btn.connect("clicked", lambda _b: self.copy_end_locked())
        end_copy_row.append(end_copy_btn)
        general_box.append(end_copy_row)

        self.end_view = self.create_textview(editable=False)
        self.end_view.set_tooltip_text(
            "Read-only closing block influenced by System/User Profile and persona type switches."
        )
        general_box.append(self.end_view)

        self.update_end_locked()

        # Sysinfo locked content (appears only if enabled)
        self.update_sys_info_content()

        self.main_widget = outer_box

    def get_widget(self):
        return self.main_widget

    # ----------------------------- Small helpers -----------------------------

    def _spacer(self):
        s = Gtk.Box()
        s.set_hexpand(True)
        return s

    def _clipboard(self):
        # GTK4: use Gdk.Display to get the clipboard
        display = self.main_widget.get_display()
        return display.get_clipboard()

    def copy_start_locked(self):
        text = self.get_start_locked()
        self._clipboard().set(text)

    def copy_end_locked(self):
        text = self.get_end_locked()
        self._clipboard().set(text)

    def _insert_default_template(self, _btn):
        if self.get_editable_content().strip():
            # Append a blank line before template when text exists
            prefix = "" if self.get_editable_content().endswith("\n\n") else "\n\n"
        else:
            prefix = ""
        template = (
            "Goals:\n"
            "1) Provide accurate, concise answers.\n"
            "2) Ask clarifying questions when needed.\n"
            "Tone:\n"
            "- Friendly, professional, and helpful.\n"
            "Constraints:\n"
            "- Keep responses grounded in provided context.\n"
        )
        self._set_editable_text(self.get_editable_content() + prefix + template)

    def _set_editable_text(self, text: str, update_counter: bool = True):
        buf = self.editable_view.get_buffer()
        buf.set_text(text, -1)
        if update_counter:
            self._update_counter()

    def _update_counter(self):
        text = self.get_editable_content()
        chars = len(text)
        words = len([w for w in text.split() if w.strip()])
        if self.counter_label:
            self.counter_label.set_text(f"{chars} chars • {words} words")

    # ----------------------------- Widget factories -----------------------------

    def create_textview(self, editable=True, height=None, css_class=None):
        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        textview.set_editable(editable)
        textview.set_cursor_visible(editable)
        textview.set_monospace(True if editable else False)

        if css_class:
            textview.get_style_context().add_class(css_class)
        if height:
            textview.set_size_request(-1, height)

        buffer = textview.get_buffer()
        buffer.connect('changed', self.on_textview_size_changed, textview)

        # Tooltips for convenience
        textview.set_tooltip_text(
            "Use paragraphs and short lists. Content wraps automatically."
            if editable else "Read-only text generated from other settings."
        )
        return textview

    def create_labeled_entry_with_info(self, label_text, initial_text, info_text):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_tooltip_text(info_text)
        box.append(label)

        entry = Gtk.Entry()
        entry.set_placeholder_text("Type here…")
        entry.set_text(initial_text)
        entry.set_width_chars(30)
        entry.set_tooltip_text(info_text)
        box.append(entry)

        info_button = Gtk.Button(label="?")
        info_button.add_css_class("flat")
        info_button.get_style_context().add_class("info-button")
        info_button.set_tooltip_text("More info")
        info_button.connect("clicked", self.show_info_popup, info_text)
        box.append(info_button)

        if label_text == "Persona Name":
            self.name_entry = entry
        elif label_text == "Name Meaning":
            self.meaning_entry = entry

        return box

    def create_label_with_info(self, label_text, info_text):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_tooltip_text(info_text)
        row.append(label)

        info_button = Gtk.Button(label="?")
        info_button.add_css_class("flat")
        info_button.get_style_context().add_class("info-button")
        info_button.set_tooltip_text("More info")
        info_button.connect("clicked", self.show_info_popup, info_text)
        row.append(info_button)

        return row

    # ----------------------------- Events & updates -----------------------------

    def show_info_popup(self, widget, info_text):
        popup = InfoPopup(self.main_widget.get_root(), info_text)
        popup.get_style_context().add_class("info-popup")
        popup.present()

    def on_entry_changed(self, entry):
        layout = entry.get_layout()
        pixel_width, pixel_height = layout.get_pixel_size()
        entry.set_size_request(-1, pixel_height + 10)

    def on_textview_size_changed(self, buffer, textview):
        allocated_width = textview.get_allocated_width()
        text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        layout = textview.create_pango_layout(text)
        layout.set_width(allocated_width * Pango.SCALE)
        layout.set_wrap(Pango.WrapMode.WORD_CHAR)
        width, height = layout.get_pixel_size()
        textview.set_size_request(-1, height + 10)

    def update_start_locked(self, widget=None):
        name = self.name_entry.get_text().strip() or "Assistant"
        meaning = self.meaning_entry.get_text().strip()
        if meaning:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}: ({meaning})."
        else:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}."

        buffer = self.start_view.get_buffer()
        current_text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        if current_text.strip() != start_locked.strip():
            buffer.set_text(start_locked, -1)

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
            buffer.set_text(end_locked, -1)

    def update_sys_info_content(self):
        """
        Add/remove the Sysinfo Locked Content block safely with its label tracked.
        """
        if self.sys_info_enabled:
            if not self.sysinfo_view:
                # Label row
                self.sysinfo_label = self.create_label_with_info(
                    "Sysinfo Locked Content",
                    "Auto-inserted description of the user's system. Controlled by the System Info switch."
                )
                self.general_box.append(self.sysinfo_label)

                # View
                self.sysinfo_view = self.create_textview(editable=False)
                self.sysinfo_view.set_tooltip_text(
                    "Read-only content describing the user's system (e.g., OS, specs)."
                )
                self.sysinfo_view.get_buffer().set_text(
                    "Your current System is <<sysinfo>>. Please make all requests considering these specifications.", -1
                )
                self.general_box.append(self.sysinfo_view)
        else:
            # Remove both label and view if present
            if self.sysinfo_view:
                self.general_box.remove(self.sysinfo_view)
                self.sysinfo_view = None
            if self.sysinfo_label:
                self.general_box.remove(self.sysinfo_label)
                self.sysinfo_label = None

    def on_textview_focus_in(self, controller, widget):
        self.frame.get_style_context().add_class("editable-area-focused")

    def on_textview_focus_out(self, controller, widget):
        self.frame.get_style_context().remove_class("editable-area-focused")

    # ----------------------------- External setters -----------------------------

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

    # ----------------------------- Value getters -----------------------------

    def get_name(self):
        return self.name_entry.get_text()

    def get_meaning(self):
        return self.meaning_entry.get_text()

    def get_editable_content(self):
        buffer = self.editable_view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

    def get_start_locked(self):
        buffer = self.start_view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

    def get_end_locked(self):
        buffer = self.end_view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)

    def set_agent_enabled(self, enabled):
        self.persona.setdefault('type', {})
        self.persona['type']['Agent'] = {'enabled': str(enabled)}

    def get_sys_info_content(self):
        if self.sysinfo_view:
            buffer = self.sysinfo_view.get_buffer()
            return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
        return ""

    # Convenience setters for remaining persona flags (keep end-locked fresh)
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
