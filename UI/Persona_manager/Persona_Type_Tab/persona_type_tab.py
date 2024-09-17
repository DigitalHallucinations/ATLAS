# UI/Persona_manager/Persona_Type_Tab/persona_type_tab.py

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class PersonaTypeTab:
    def __init__(self, persona, general_tab):
        self.persona = persona
        self.general_tab = general_tab
        self.build_ui()

    def build_ui(self):
        self.type_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Switches
        self.sys_info_switch = Gtk.Switch()
        self.sys_info_switch.set_active(self.persona.get("sys_info_enabled", "False") == "True")
        self.agent_switch = Gtk.Switch()
        self.agent_switch.set_active(self.persona.get("Agent", "False") == "True")
        self.user_profile_switch = Gtk.Switch()
        self.user_profile_switch.set_active(self.persona.get("user_profile_enabled", "False") == "True")
        self.medical_persona_switch = Gtk.Switch()
        self.medical_persona_switch.set_active(self.persona.get("medical_persona", "False") == "True")
        self.educational_persona_switch = Gtk.Switch()
        self.educational_persona_switch.set_active(self.persona.get("educational_persona_enabled", "False") == "True")
        self.fitness_trainer_switch = Gtk.Switch()
        self.fitness_trainer_switch.set_active(self.persona.get("fitness_trainer_enabled", "False") == "True")
        self.language_practice_switch = Gtk.Switch()
        self.language_practice_switch.set_active(self.persona.get("language_practice_enabled", "False") == "True")

        switches = [
            ("System Info Enabled", self.sys_info_switch),
            ("Agent", self.agent_switch),
            ("User Profile Enabled", self.user_profile_switch),
            ("Medical Persona", self.medical_persona_switch),
            ("Educational Tutor Persona", self.educational_persona_switch),
            ("Fitness Trainer Persona", self.fitness_trainer_switch),
            ("Language Practice Persona", self.language_practice_switch),
        ]

        for label, switch in switches:
            switch_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            switch_label = Gtk.Label(label=label)
            switch_box.pack_start(switch_label, False, False, 0)
            switch_box.pack_end(switch, False, False, 0)
            self.type_box.pack_start(switch_box, False, False, 0)

        # Connect signals to update UI
        self.sys_info_switch.connect("notify::active", self.on_sys_info_switch_toggled)
        self.user_profile_switch.connect("notify::active", self.on_user_profile_switch_toggled)
        self.medical_persona_switch.connect("notify::active", self.on_medical_persona_switch_toggled)
        self.educational_persona_switch.connect("notify::active", self.on_educational_persona_switch_toggled)
        self.fitness_trainer_switch.connect("notify::active", self.on_fitness_trainer_switch_toggled)
        self.language_practice_switch.connect("notify::active", self.on_language_practice_switch_toggled)

        # Additional options panels
        self.educational_options_box = self.create_educational_options()
        self.type_box.pack_start(self.educational_options_box, False, False, 0)
        self.educational_options_box.set_visible(self.educational_persona_switch.get_active())

        self.fitness_options_box = self.create_fitness_options()
        self.type_box.pack_start(self.fitness_options_box, False, False, 0)
        self.fitness_options_box.set_visible(self.fitness_trainer_switch.get_active())

        self.language_practice_options_box = self.create_language_practice_options()
        self.type_box.pack_start(self.language_practice_options_box, False, False, 0)
        self.language_practice_options_box.set_visible(self.language_practice_switch.get_active())

    def get_widget(self):
        return self.type_box

    # Switch toggled methods
    def on_sys_info_switch_toggled(self, switch, gparam):
        sys_info_enabled = switch.get_active()
        self.general_tab.set_sys_info_enabled(sys_info_enabled)

    def on_user_profile_switch_toggled(self, switch, gparam):
        user_profile_enabled = switch.get_active()
        self.general_tab.set_user_profile_enabled(user_profile_enabled)

    def on_medical_persona_switch_toggled(self, switch, gparam):
        medical_persona_enabled = switch.get_active()
        self.general_tab.set_medical_persona_enabled(medical_persona_enabled)

    def on_educational_persona_switch_toggled(self, switch, gparam):
        educational_persona_enabled = switch.get_active()
        self.educational_options_box.set_visible(educational_persona_enabled)
        self.general_tab.set_educational_persona_enabled(educational_persona_enabled)
        self.general_tab.update_end_locked()

    def on_fitness_trainer_switch_toggled(self, switch, gparam):
        fitness_persona_enabled = switch.get_active()
        self.fitness_options_box.set_visible(fitness_persona_enabled)
        self.general_tab.set_fitness_persona_enabled(fitness_persona_enabled)
        self.general_tab.update_end_locked()

    def on_language_practice_switch_toggled(self, switch, gparam):
        language_practice_enabled = switch.get_active()
        self.language_practice_options_box.set_visible(language_practice_enabled)
        self.general_tab.set_language_practice_enabled(language_practice_enabled)
        self.general_tab.update_end_locked()

    # Options for different personas
    def create_educational_options(self):
        options_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        options_box.set_margin_start(20)

        # Subject Specialization
        subject_label = Gtk.Label(label="Subject Specialization")
        self.subject_entry = Gtk.Entry()
        self.subject_entry.set_text(self.persona.get("subject_specialization", "General"))
        subject_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        subject_box.pack_start(subject_label, False, False, 0)
        subject_box.pack_start(self.subject_entry, True, True, 0)
        options_box.pack_start(subject_box, False, False, 0)

        # Education Level
        level_label = Gtk.Label(label="Education Level")
        self.level_combo = Gtk.ComboBoxText()
        levels = ["Elementary", "Middle School", "High School", "College", "Advanced"]
        for level in levels:
            self.level_combo.append_text(level)
        level_text = self.persona.get("education_level", "High School")
        if level_text in levels:
            self.level_combo.set_active(levels.index(level_text))
        else:
            self.level_combo.set_active(2)  # Default to "High School"
        level_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        level_box.pack_start(level_label, False, False, 0)
        level_box.pack_start(self.level_combo, False, False, 0)
        options_box.pack_start(level_box, False, False, 0)

        # Teaching Style
        style_label = Gtk.Label(label="Teaching Style")
        self.style_combo = Gtk.ComboBoxText()
        styles = ["Socratic Method", "Lecture Style", "Interactive Exercises"]
        for style in styles:
            self.style_combo.append_text(style)
        style_text = self.persona.get("teaching_style", "Lecture Style")
        if style_text in styles:
            self.style_combo.set_active(styles.index(style_text))
        else:
            self.style_combo.set_active(1)  # Default to "Lecture Style"
        style_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        style_box.pack_start(style_label, False, False, 0)
        style_box.pack_start(self.style_combo, False, False, 0)
        options_box.pack_start(style_box, False, False, 0)

        return options_box

    def create_fitness_options(self):
        options_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        options_box.set_margin_start(20)

        # Fitness Goal
        goal_label = Gtk.Label(label="Fitness Goal")
        self.goal_combo = Gtk.ComboBoxText()
        goals = ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"]
        for goal in goals:
            self.goal_combo.append_text(goal)
        goal_text = self.persona.get("fitness_goal", "Weight Loss")
        if goal_text in goals:
            self.goal_combo.set_active(goals.index(goal_text))
        else:
            self.goal_combo.set_active(0)  # Default to "Weight Loss"
        goal_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        goal_box.pack_start(goal_label, False, False, 0)
        goal_box.pack_start(self.goal_combo, False, False, 0)
        options_box.pack_start(goal_box, False, False, 0)

        # Exercise Preference
        exercise_label = Gtk.Label(label="Exercise Preference")
        self.exercise_combo = Gtk.ComboBoxText()
        exercises = ["Gym Workouts", "Home Exercises", "Yoga", "Cardio"]
        for exercise in exercises:
            self.exercise_combo.append_text(exercise)
        exercise_text = self.persona.get("exercise_preference", "Gym Workouts")
        if exercise_text in exercises:
            self.exercise_combo.set_active(exercises.index(exercise_text))
        else:
            self.exercise_combo.set_active(0)  # Default to "Gym Workouts"
        exercise_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        exercise_box.pack_start(exercise_label, False, False, 0)
        exercise_box.pack_start(self.exercise_combo, False, False, 0)
        options_box.pack_start(exercise_box, False, False, 0)

        return options_box

    def create_language_practice_options(self):
        options_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        options_box.set_margin_start(20)

        # Target Language
        language_label = Gtk.Label(label="Target Language")
        self.language_entry = Gtk.Entry()
        self.language_entry.set_text(self.persona.get("target_language", "Spanish"))
        language_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        language_box.pack_start(language_label, False, False, 0)
        language_box.pack_start(self.language_entry, True, True, 0)
        options_box.pack_start(language_box, False, False, 0)

        # Proficiency Level
        proficiency_label = Gtk.Label(label="Proficiency Level")
        self.proficiency_combo = Gtk.ComboBoxText()
        levels = ["Beginner", "Intermediate", "Advanced"]
        for level in levels:
            self.proficiency_combo.append_text(level)
        proficiency_text = self.persona.get("proficiency_level", "Beginner")
        if proficiency_text in levels:
            self.proficiency_combo.set_active(levels.index(proficiency_text))
        else:
            self.proficiency_combo.set_active(0)  # Default to "Beginner"
        proficiency_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        proficiency_box.pack_start(proficiency_label, False, False, 0)
        proficiency_box.pack_start(self.proficiency_combo, False, False, 0)
        options_box.pack_start(proficiency_box, False, False, 0)

        return options_box

    # Methods to retrieve values
    def get_sys_info_enabled(self):
        return self.sys_info_switch.get_active()

    def get_agent_enabled(self):
        return self.agent_switch.get_active()

    def get_user_profile_enabled(self):
        return self.user_profile_switch.get_active()

    def get_medical_persona_enabled(self):
        return self.medical_persona_switch.get_active()

    def get_educational_persona_enabled(self):
        return self.educational_persona_switch.get_active()

    def get_fitness_trainer_enabled(self):
        return self.fitness_trainer_switch.get_active()

    def get_language_practice_enabled(self):
        return self.language_practice_switch.get_active()

    def get_educational_options(self):
        return {
            'subject_specialization': self.subject_entry.get_text(),
            'education_level': self.level_combo.get_active_text(),
            'teaching_style': self.style_combo.get_active_text()
        }

    def get_fitness_options(self):
        return {
            'fitness_goal': self.goal_combo.get_active_text(),
            'exercise_preference': self.exercise_combo.get_active_text()
        }

    def get_language_practice_options(self):
        return {
            'target_language': self.language_entry.get_text(),
            'proficiency_level': self.proficiency_combo.get_active_text()
        }

    def get_values(self):
        values = {
            'sys_info_enabled': self.get_sys_info_enabled(),
            'agent': self.get_agent_enabled(),
            'user_profile_enabled': self.get_user_profile_enabled(),
            'medical_persona': self.get_medical_persona_enabled(),
            'educational_persona_enabled': self.get_educational_persona_enabled(),
            'fitness_trainer_enabled': self.get_fitness_trainer_enabled(),
            'language_practice_enabled': self.get_language_practice_enabled()
        }

        if values['educational_persona_enabled']:
            values.update(self.get_educational_options())
        if values['fitness_trainer_enabled']:
            values.update(self.get_fitness_options())
        if values['language_practice_enabled']:
            values.update(self.get_language_practice_options())

        return values
