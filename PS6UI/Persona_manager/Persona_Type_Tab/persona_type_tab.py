# PS6UI/Persona_manager/Persona_Type_Tab/persona_type_tab.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QTabWidget, QLineEdit, QComboBox
)
from PySide6.QtCore import Qt

class PersonaTypeTab:
    def __init__(self, persona, general_tab):
        self.persona = persona
        self.general_tab = general_tab
        self.persona_type = self.persona.get('type', {})
        self.tabs = {}
        self.build_ui()

    def build_ui(self):
        self.type_widget = QWidget()
        self.type_layout = QVBoxLayout(self.type_widget)
        self.type_layout.setSpacing(10)

        # Create a QTabWidget to hold tabs
        self.sub_notebook = QTabWidget()
        self.type_layout.addWidget(self.sub_notebook)

        # Create switches
        self.create_switches()

        # Create the main switches tab
        self.create_main_switches_tab()

        # Add the main switches tab
        main_tab_widget = QWidget()
        main_tab_layout = QVBoxLayout(main_tab_widget)
        main_tab_layout.setSpacing(10)
        main_tab_layout.setContentsMargins(10,10,10,10)

        for label_text, switch in self.main_switches:
            hbox = QHBoxLayout()
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            hbox.addWidget(label)
            hbox.addWidget(switch)
            main_tab_layout.addLayout(hbox)

        self.sub_notebook.addTab(main_tab_widget, "Main")

        # Initialize persona type tabs
        self.update_persona_type_tabs()

    def get_widget(self):
        return self.type_widget

    def create_switches(self):
        # We'll use QCheckBox as a toggle switch replacement.
        # active=True means checked.

        def create_checkbox(enabled):
            cb = QCheckBox()
            cb.setChecked(enabled)
            return cb

        self.sys_info_switch = create_checkbox(self.persona.get("sys_info_enabled", "False") == "True")
        self.sys_info_switch.stateChanged.connect(lambda: self.on_sys_info_switch_toggled(self.sys_info_switch))

        self.agent_switch = create_checkbox(self.persona_type.get("Agent", {}).get("enabled", "False") == "True")
        self.agent_switch.stateChanged.connect(lambda: self.on_agent_switch_toggled(self.agent_switch))

        self.user_profile_switch = create_checkbox(self.persona.get("user_profile_enabled", "False") == "True")
        self.user_profile_switch.stateChanged.connect(lambda: self.on_user_profile_switch_toggled(self.user_profile_switch))

        self.medical_persona_switch = create_checkbox(self.persona_type.get("medical_persona", {}).get("enabled", "False") == "True")
        self.medical_persona_switch.stateChanged.connect(lambda: self.on_medical_persona_switch_toggled(self.medical_persona_switch))

        self.educational_persona_switch = create_checkbox(self.persona_type.get("educational_persona", {}).get("enabled", "False") == "True")
        self.educational_persona_switch.stateChanged.connect(lambda: self.on_educational_persona_switch_toggled(self.educational_persona_switch))

        self.fitness_trainer_switch = create_checkbox(self.persona_type.get("fitness_persona", {}).get("enabled", "False") == "True")
        self.fitness_trainer_switch.stateChanged.connect(lambda: self.on_fitness_trainer_switch_toggled(self.fitness_trainer_switch))

        self.language_practice_switch = create_checkbox(self.persona_type.get("language_instructor", {}).get("enabled", "False") == "True")
        self.language_practice_switch.stateChanged.connect(lambda: self.on_language_practice_switch_toggled(self.language_practice_switch))

        self.legal_persona_switch = create_checkbox(self.persona_type.get("legal_persona", {}).get("enabled", "False") == "True")
        self.legal_persona_switch.stateChanged.connect(lambda: self.on_legal_persona_switch_toggled(self.legal_persona_switch))

        self.financial_advisor_switch = create_checkbox(self.persona_type.get("financial_advisor", {}).get("enabled", "False") == "True")
        self.financial_advisor_switch.stateChanged.connect(lambda: self.on_financial_advisor_switch_toggled(self.financial_advisor_switch))

        self.tech_support_switch = create_checkbox(self.persona_type.get("tech_support", {}).get("enabled", "False") == "True")
        self.tech_support_switch.stateChanged.connect(lambda: self.on_tech_support_switch_toggled(self.tech_support_switch))

        self.personal_assistant_switch = create_checkbox(self.persona_type.get("personal_assistant", {}).get("enabled", "False") == "True")
        self.personal_assistant_switch.stateChanged.connect(lambda: self.on_personal_assistant_switch_toggled(self.personal_assistant_switch))

        self.therapist_switch = create_checkbox(self.persona_type.get("therapist", {}).get("enabled", "False") == "True")
        self.therapist_switch.stateChanged.connect(lambda: self.on_therapist_switch_toggled(self.therapist_switch))

        self.travel_guide_switch = create_checkbox(self.persona_type.get("travel_guide", {}).get("enabled", "False") == "True")
        self.travel_guide_switch.stateChanged.connect(lambda: self.on_travel_guide_switch_toggled(self.travel_guide_switch))

        self.storyteller_switch = create_checkbox(self.persona_type.get("storyteller", {}).get("enabled", "False") == "True")
        self.storyteller_switch.stateChanged.connect(lambda: self.on_storyteller_switch_toggled(self.storyteller_switch))

        self.game_master_switch = create_checkbox(self.persona_type.get("game_master", {}).get("enabled", "False") == "True")
        self.game_master_switch.stateChanged.connect(lambda: self.on_game_master_switch_toggled(self.game_master_switch))

        self.chef_switch = create_checkbox(self.persona_type.get("chef", {}).get("enabled", "False") == "True")
        self.chef_switch.stateChanged.connect(lambda: self.on_chef_switch_toggled(self.chef_switch))

    def create_main_switches_tab(self):
        self.main_switches = [
            ("System Info Enabled", self.sys_info_switch),
            ("Agent", self.agent_switch),
            ("User Profile Enabled", self.user_profile_switch),
            ("Medical Persona", self.medical_persona_switch),
            ("Educational Persona", self.educational_persona_switch),
            ("Fitness Persona", self.fitness_trainer_switch),
            ("Language Instructor Persona", self.language_practice_switch),
            ("Legal Persona", self.legal_persona_switch),
            ("Financial Advisor Persona", self.financial_advisor_switch),
            ("Tech Support Persona", self.tech_support_switch),
            ("Personal Assistant Persona", self.personal_assistant_switch),
            ("Therapist Persona", self.therapist_switch),
            ("Travel Guide Persona", self.travel_guide_switch),
            ("Storyteller Persona", self.storyteller_switch),
            ("Game Master Persona", self.game_master_switch),
            ("Chef Persona", self.chef_switch),
        ]

    def update_persona_type_tabs(self):
        persona_types = {
            'Medical': self.medical_persona_switch,
            'Educational': self.educational_persona_switch,
            'Fitness': self.fitness_trainer_switch,
            'Language Instructor': self.language_practice_switch,
            'Legal': self.legal_persona_switch,
            'Financial Advisor': self.financial_advisor_switch,
            'Tech Support': self.tech_support_switch,
            'Personal Assistant': self.personal_assistant_switch,
            'Therapist': self.therapist_switch,
            'Travel Guide': self.travel_guide_switch,
            'Storyteller': self.storyteller_switch,
            'Game Master': self.game_master_switch,
            'Chef': self.chef_switch,
        }

        for tab_name, switch in persona_types.items():
            if switch.isChecked():
                self.add_tab(tab_name)
            else:
                self.remove_tab(tab_name)

    def on_sys_info_switch_toggled(self, switch):
        sys_info_enabled = switch.isChecked()
        self.general_tab.set_sys_info_enabled(sys_info_enabled)

    def on_agent_switch_toggled(self, switch):
        agent_enabled = switch.isChecked()
        self.general_tab.set_agent_enabled(agent_enabled)

    def on_user_profile_switch_toggled(self, switch):
        user_profile_enabled = switch.isChecked()
        self.general_tab.set_user_profile_enabled(user_profile_enabled)

    def on_medical_persona_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_medical_persona_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_educational_persona_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_educational_persona(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_fitness_trainer_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_fitness_persona_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_language_practice_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_language_instructor(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_legal_persona_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_legal_persona_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_financial_advisor_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_financial_advisor_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_tech_support_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_tech_support_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_personal_assistant_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_personal_assistant_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_therapist_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_therapist_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_travel_guide_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_travel_guide_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_storyteller_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_storyteller_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_game_master_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_game_master_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def on_chef_switch_toggled(self, switch):
        self.update_persona_type_tabs()
        self.general_tab.set_chef_enabled(switch.isChecked())
        self.general_tab.update_end_locked()

    def add_tab(self, tab_name):
        if tab_name in self.tabs:
            return

        tab = self.create_persona_type_tab(tab_name)
        if tab:
            self.tabs[tab_name] = tab
            self.sub_notebook.addTab(tab, tab_name)

    def remove_tab(self, tab_name):
        if tab_name in self.tabs:
            index = self.sub_notebook.indexOf(self.tabs[tab_name])
            if index != -1:
                self.sub_notebook.removeTab(index)
            del self.tabs[tab_name]

    # Create persona-type-specific tabs (simplified)
    def create_persona_type_tab(self, tab_name):
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10,10,10,10)
        layout.setSpacing(10)

        # Example tabs
        if tab_name == 'Medical':
            label = QLabel("No additional settings for Medical Persona.")
            layout.addWidget(label)
        elif tab_name == 'Educational':
            # Subject specialization
            subject_label = QLabel("Subject Specialization")
            self.subject_entry = QLineEdit()
            self.subject_entry.setText(self.persona_type.get("educational_persona", {}).get("subject_specialization", "General"))
            sub_h = QHBoxLayout()
            sub_h.addWidget(subject_label)
            sub_h.addWidget(self.subject_entry)
            layout.addLayout(sub_h)

            # Education Level
            level_label = QLabel("Education Level")
            self.level_combo = QComboBox()
            levels = ["Elementary", "Middle School", "High School", "College", "Advanced"]
            self.level_combo.addItems(levels)
            level_text = self.persona_type.get("educational_persona", {}).get("education_level", "High School")
            if level_text in levels:
                self.level_combo.setCurrentIndex(levels.index(level_text))
            else:
                self.level_combo.setCurrentIndex(2)
            level_h = QHBoxLayout()
            level_h.addWidget(level_label)
            level_h.addWidget(self.level_combo)
            layout.addLayout(level_h)

            # Teaching Style
            style_label = QLabel("Teaching Style")
            self.style_combo = QComboBox()
            styles = ["Socratic Method", "Lecture Style", "Interactive Exercises"]
            self.style_combo.addItems(styles)
            style_text = self.persona_type.get("educational_persona", {}).get("teaching_style", "Lecture Style")
            if style_text in styles:
                self.style_combo.setCurrentIndex(styles.index(style_text))
            else:
                self.style_combo.setCurrentIndex(1)
            style_h = QHBoxLayout()
            style_h.addWidget(style_label)
            style_h.addWidget(self.style_combo)
            layout.addLayout(style_h)

        elif tab_name == 'Fitness':
            goal_label = QLabel("Fitness Goal")
            self.goal_combo = QComboBox()
            goals = ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"]
            self.goal_combo.addItems(goals)
            goal_text = self.persona_type.get("fitness_persona", {}).get("fitness_goal", "Weight Loss")
            if goal_text in goals:
                self.goal_combo.setCurrentIndex(goals.index(goal_text))
            else:
                self.goal_combo.setCurrentIndex(0)
            goal_h = QHBoxLayout()
            goal_h.addWidget(goal_label)
            goal_h.addWidget(self.goal_combo)
            layout.addLayout(goal_h)

            exercise_label = QLabel("Exercise Preference")
            self.exercise_combo = QComboBox()
            exercises = ["Gym Workouts", "Home Exercises", "Yoga", "Cardio"]
            self.exercise_combo.addItems(exercises)
            exercise_text = self.persona_type.get("fitness_persona", {}).get("exercise_preference", "Gym Workouts")
            if exercise_text in exercises:
                self.exercise_combo.setCurrentIndex(exercises.index(exercise_text))
            else:
                self.exercise_combo.setCurrentIndex(0)
            exercise_h = QHBoxLayout()
            exercise_h.addWidget(exercise_label)
            exercise_h.addWidget(self.exercise_combo)
            layout.addLayout(exercise_h)

        elif tab_name == 'Language Instructor':
            language_label = QLabel("Target Language")
            self.language_entry = QLineEdit()
            self.language_entry.setText(self.persona_type.get("language_instructor", {}).get("target_language", "Spanish"))
            lang_h = QHBoxLayout()
            lang_h.addWidget(language_label)
            lang_h.addWidget(self.language_entry)
            layout.addLayout(lang_h)

            proficiency_label = QLabel("Proficiency Level")
            self.proficiency_combo = QComboBox()
            levels = ["Beginner", "Intermediate", "Advanced"]
            self.proficiency_combo.addItems(levels)
            proficiency_text = self.persona_type.get("language_instructor", {}).get("proficiency_level", "Beginner")
            if proficiency_text in levels:
                self.proficiency_combo.setCurrentIndex(levels.index(proficiency_text))
            else:
                self.proficiency_combo.setCurrentIndex(0)
            prof_h = QHBoxLayout()
            prof_h.addWidget(proficiency_label)
            prof_h.addWidget(self.proficiency_combo)
            layout.addLayout(prof_h)

        else:
            # For other tabs, no extra settings are shown for simplicity.
            label = QLabel(f"No additional settings for {tab_name} Persona.")
            layout.addWidget(label)

        return box

    # Methods to retrieve values remain similar to original - integrate with general_tab logic if needed
    def get_values(self):
        # Similar logic as original code
        values = {
            'sys_info_enabled': self.get_sys_info_enabled(),
            'user_profile_enabled': self.get_user_profile_enabled(),
        }

        type_values = {
            'Agent': {'enabled': str(self.get_agent_enabled())},
            'medical_persona': {'enabled': str(self.get_medical_persona_enabled())},
            'educational_persona': {'enabled': str(self.get_educational_persona())},
            'fitness_persona': {'enabled': str(self.get_fitness_persona())},
            'language_instructor': {'enabled': str(self.get_language_instructor())},
        }

        if type_values['educational_persona']['enabled'] == "True":
            type_values['educational_persona'].update(self.get_educational_options())
        if type_values['fitness_persona']['enabled'] == "True":
            type_values['fitness_persona'].update(self.get_fitness_options())
        if type_values['language_instructor']['enabled'] == "True":
            type_values['language_instructor'].update(self.get_language_practice_options())

        values['type'] = type_values
        return values

    def get_sys_info_enabled(self):
        return self.sys_info_switch.isChecked()

    def get_agent_enabled(self):
        return self.agent_switch.isChecked()

    def get_user_profile_enabled(self):
        return self.user_profile_switch.isChecked()

    def get_medical_persona_enabled(self):
        return self.medical_persona_switch.isChecked()

    def get_educational_persona(self):
        return self.educational_persona_switch.isChecked()

    def get_fitness_persona(self):
        return self.fitness_trainer_switch.isChecked()

    def get_language_instructor(self):
        return self.language_practice_switch.isChecked()

    def get_educational_options(self):
        options = {}
        if hasattr(self, 'subject_entry'):
            options['subject_specialization'] = self.subject_entry.text()
        else:
            options['subject_specialization'] = 'General'
        if hasattr(self, 'level_combo'):
            options['education_level'] = self.level_combo.currentText()
        else:
            options['education_level'] = 'High School'
        if hasattr(self, 'style_combo'):
            options['teaching_style'] = self.style_combo.currentText()
        else:
            options['teaching_style'] = 'Lecture Style'
        return options

    def get_fitness_options(self):
        options = {}
        if hasattr(self, 'goal_combo'):
            options['fitness_goal'] = self.goal_combo.currentText()
        else:
            options['fitness_goal'] = 'Weight Loss'
        if hasattr(self, 'exercise_combo'):
            options['exercise_preference'] = self.exercise_combo.currentText()
        else:
            options['exercise_preference'] = 'Gym Workouts'
        return options

    def get_language_practice_options(self):
        options = {}
        if hasattr(self, 'language_entry'):
            options['target_language'] = self.language_entry.text()
        else:
            options['target_language'] = 'Spanish'
        if hasattr(self, 'proficiency_combo'):
            options['proficiency_level'] = self.proficiency_combo.currentText()
        else:
            options['proficiency_level'] = 'Beginner'
        return options
