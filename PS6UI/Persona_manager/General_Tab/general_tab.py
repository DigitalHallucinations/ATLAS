# PS6UI/Persona_manager/General_Tab/general_tab.py

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption

class InfoPopup(QDialog):
    def __init__(self, parent, text):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setModal(True)

        layout = QVBoxLayout()
        label = QLabel(text)
        label.setWordWrap(False)
        label.setAlignment(Qt.AlignLeft)
        layout.addWidget(label)
        self.setLayout(layout)

        # Close when the user clicks anywhere in the dialog
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()

        # Clicking anywhere should close the dialog
        self.grabKeyboard()
        self.grabMouse()

    def mousePressEvent(self, event):
        self.close()
        super().mousePressEvent(event)

    def focusOutEvent(self, event):
        self.close()
        super().focusOutEvent(event)

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
        self.main_widget = QWidget()

        outer_layout = QVBoxLayout(self.main_widget)
        outer_layout.setContentsMargins(10,10,10,10)
        outer_layout.setSpacing(0)

        # Scroll area for the general_box
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        general_container = QWidget()
        self.general_box = QVBoxLayout(general_container)
        self.general_box.setSpacing(10)
        scroll_area.setWidget(general_container)

        # Persona name
        name_box = self.create_labeled_entry_with_info(
            "Persona Name",
            self.persona.get("name", ""),
            "The name of the persona. This is how you'll identify and select this persona."
        )
        self.general_box.addWidget(name_box)

        # Name meaning
        meaning_box = self.create_labeled_entry_with_info(
            "Name Meaning",
            self.persona.get("meaning", ""),
            "The meaning or expansion of the persona's name. This helps clarify the persona's purpose or role."
        )
        self.general_box.addWidget(meaning_box)

        # Start Locked Content
        start_locked_label = self.create_label_with_info(
            "Start Locked Content",
            "This content appears at the start of the persona's system prompt. It typically includes the persona's name and role introduction."
        )
        self.general_box.addWidget(start_locked_label)

        self.start_view = self.create_textview(editable=False)
        self.general_box.addWidget(self.start_view)

        # Update start locked
        self.update_start_locked()

        # Connect signals for name and meaning
        self.name_entry.textChanged.connect(self.update_start_locked)
        self.meaning_entry.textChanged.connect(self.update_start_locked)

        # Editable Content
        editable_content_label = self.create_label_with_info(
            "Editable Content",
            "This is the main content of the persona's system prompt. You can freely edit this to define the persona's behavior, knowledge, and characteristics."
        )
        self.general_box.addWidget(editable_content_label)

        self.editable_view = self.create_textview(editable=True, height=200)

        content = self.persona.get("content", {})
        editable_content = content.get("editable_content", "")
        self.editable_view.setPlainText(editable_content)

        # Frame for editable area
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0,0,0,0)
        frame_layout.addWidget(self.editable_view)
        self.frame = frame
        self.general_box.addWidget(frame)

        # End Locked Content
        end_locked_label = self.create_label_with_info(
            "End Locked Content",
            "This content appears at the end of the persona's system prompt. It typically includes user-specific information and general instructions."
        )
        self.general_box.addWidget(end_locked_label)

        self.end_view = self.create_textview(editable=False)
        self.general_box.addWidget(self.end_view)

        self.update_end_locked()

        self.sysinfo_view = None
        self.update_sys_info_content()

    def get_widget(self):
        return self.main_widget

    def create_textview(self, editable=True, height=None):
        text_edit = QTextEdit()
        text_edit.setWordWrapMode(QTextOption.WordWrap)
        text_edit.setReadOnly(not editable)
        if height:
            text_edit.setFixedHeight(height)
        return text_edit

    def create_labeled_entry_with_info(self, label_text, initial_text, info_text):
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.setSpacing(5)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h_layout.addWidget(label)

        entry = QLineEdit()
        entry.setText(initial_text)
        entry.setFixedWidth(200)
        h_layout.addWidget(entry)

        info_button = QPushButton("?")
        info_button.clicked.connect(lambda: self.show_info_popup(info_text))
        h_layout.addWidget(info_button)

        if label_text == "Persona Name":
            self.name_entry = entry
        elif label_text == "Name Meaning":
            self.meaning_entry = entry

        return widget

    def create_label_with_info(self, label_text, info_text):
        widget = QWidget()
        h_layout = QHBoxLayout(widget)
        h_layout.setSpacing(5)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h_layout.addWidget(label)

        info_button = QPushButton("?")
        info_button.clicked.connect(lambda: self.show_info_popup(info_text))
        h_layout.addWidget(info_button)

        return widget

    def show_info_popup(self, info_text):
        popup = InfoPopup(self.main_widget, info_text)
        popup.exec()

    def update_start_locked(self):
        name = self.name_entry.text()
        meaning = self.meaning_entry.text()
        if meaning:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}: ({meaning})."
        else:
            start_locked = f"The name of the user you are speaking to is <<name>>. Your name is {name}."

        current_text = self.start_view.toPlainText()
        if current_text.strip() != start_locked.strip():
            self.start_view.setPlainText(start_locked)

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

        current_text = self.end_view.toPlainText()
        if current_text.strip() != dynamic_content.strip():
            self.end_view.setPlainText(dynamic_content)

    def update_sys_info_content(self):
        # If sys_info_enabled, add sysinfo_view
        if self.sys_info_enabled:
            if not self.sysinfo_view:
                label_widget = self.create_label_with_info(
                    "Sysinfo Locked Content",
                    "This content provides information about the user's system specifications."
                )
                self.general_box.addWidget(label_widget)

                self.sysinfo_view = self.create_textview(editable=False)
                self.sysinfo_view.setPlainText(
                    "Your current System is <<sysinfo>>. Please make all requests considering these specifications."
                )
                self.general_box.addWidget(self.sysinfo_view)
        else:
            if self.sysinfo_view:
                # Remove sysinfo_view and its label. As we dynamically added them, we must remove them carefully.
                # This code assumes they were appended last. Otherwise, store references.
                # Just rebuild UI or keep references for removal in a real scenario.
                items_to_remove = []
                parent_layout = self.general_box
                # A more robust approach: iterate over widgets and remove the relevant ones.
                for i in range(parent_layout.count()):
                    item = parent_layout.itemAt(i)
                    if item.widget() == self.sysinfo_view:
                        items_to_remove.append(i)
                    # You would also remove the label above it if needed
                for i in reversed(items_to_remove):
                    w = parent_layout.itemAt(i).widget()
                    parent_layout.removeWidget(w)
                    w.deleteLater()
                self.sysinfo_view = None

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
        return self.name_entry.text()

    def get_meaning(self):
        return self.meaning_entry.text()

    def get_editable_content(self):
        return self.editable_view.toPlainText()

    def get_start_locked(self):
        return self.start_view.toPlainText()

    def get_end_locked(self):
        return self.end_view.toPlainText()

    def set_agent_enabled(self, enabled):
        self.persona['type']['Agent'] = {'enabled': str(enabled)}

    def get_sys_info_content(self):
        if self.sysinfo_view:
            return self.sysinfo_view.toPlainText()
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
