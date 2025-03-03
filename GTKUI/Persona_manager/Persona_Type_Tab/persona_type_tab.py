# UI/Persona_manager/Persona_Type_Tab/persona_type_tab.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

class PersonaTypeTab:
    def __init__(self, persona, general_tab):
        self.persona = persona
        self.general_tab = general_tab
        self.persona_type = self.persona.get('type', {})
        self.tabs = {}  # Dictionary to store tabs
        self.build_ui()

    def build_ui(self):
        self.type_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Create a notebook to hold tabs for main switches and each persona type
        self.sub_notebook = Gtk.Notebook()
        self.type_box.append(self.sub_notebook)

        # Create the switches and store them as attributes
        self.create_switches()

        # Create the main switches tab
        self.create_main_switches_tab()

        # Add the main switches tab to the notebook
        self.sub_notebook.append_page(self.main_switches_box, Gtk.Label(label="Main"))

        # Initialize persona type tabs
        self.update_persona_type_tabs()

    def get_widget(self):
        return self.type_box

    def create_switches(self):
        # Switches for each persona type
        self.sys_info_switch = Gtk.Switch()
        self.sys_info_switch.set_active(self.persona.get("sys_info_enabled", "False") == "True")
        self.sys_info_switch.connect("notify::active", self.on_sys_info_switch_toggled)

        self.agent_switch = Gtk.Switch()
        self.agent_switch.set_active(self.persona_type.get("Agent", {}).get("enabled", "False") == "True")
        self.agent_switch.connect("notify::active", self.on_agent_switch_toggled)

        self.user_profile_switch = Gtk.Switch()
        self.user_profile_switch.set_active(self.persona.get("user_profile_enabled", "False") == "True")
        self.user_profile_switch.connect("notify::active", self.on_user_profile_switch_toggled)

        # Existing persona type switches
        self.medical_persona_switch = Gtk.Switch()
        self.medical_persona_switch.set_active(self.persona_type.get("medical_persona", {}).get("enabled", "False") == "True")
        self.medical_persona_switch.connect("notify::active", self.on_medical_persona_switch_toggled)

        self.educational_persona_switch = Gtk.Switch()
        self.educational_persona_switch.set_active(self.persona_type.get("educational_persona", {}).get("enabled", "False") == "True")
        self.educational_persona_switch.connect("notify::active", self.on_educational_persona_switch_toggled)

        self.fitness_trainer_switch = Gtk.Switch()
        self.fitness_trainer_switch.set_active(self.persona_type.get("fitness_persona", {}).get("enabled", "False") == "True")
        self.fitness_trainer_switch.connect("notify::active", self.on_fitness_trainer_switch_toggled)

        self.language_practice_switch = Gtk.Switch()
        self.language_practice_switch.set_active(self.persona_type.get("language_instructor", {}).get("enabled", "False") == "True")
        self.language_practice_switch.connect("notify::active", self.on_language_practice_switch_toggled)

        # New persona type switches
        self.legal_persona_switch = Gtk.Switch()
        self.legal_persona_switch.set_active(self.persona_type.get("legal_persona", {}).get("enabled", "False") == "True")
        self.legal_persona_switch.connect("notify::active", self.on_legal_persona_switch_toggled)

        self.financial_advisor_switch = Gtk.Switch()
        self.financial_advisor_switch.set_active(self.persona_type.get("financial_advisor", {}).get("enabled", "False") == "True")
        self.financial_advisor_switch.connect("notify::active", self.on_financial_advisor_switch_toggled)

        self.tech_support_switch = Gtk.Switch()
        self.tech_support_switch.set_active(self.persona_type.get("tech_support", {}).get("enabled", "False") == "True")
        self.tech_support_switch.connect("notify::active", self.on_tech_support_switch_toggled)

        self.personal_assistant_switch = Gtk.Switch()
        self.personal_assistant_switch.set_active(self.persona_type.get("personal_assistant", {}).get("enabled", "False") == "True")
        self.personal_assistant_switch.connect("notify::active", self.on_personal_assistant_switch_toggled)

        self.therapist_switch = Gtk.Switch()
        self.therapist_switch.set_active(self.persona_type.get("therapist", {}).get("enabled", "False") == "True")
        self.therapist_switch.connect("notify::active", self.on_therapist_switch_toggled)

        self.travel_guide_switch = Gtk.Switch()
        self.travel_guide_switch.set_active(self.persona_type.get("travel_guide", {}).get("enabled", "False") == "True")
        self.travel_guide_switch.connect("notify::active", self.on_travel_guide_switch_toggled)

        self.storyteller_switch = Gtk.Switch()
        self.storyteller_switch.set_active(self.persona_type.get("storyteller", {}).get("enabled", "False") == "True")
        self.storyteller_switch.connect("notify::active", self.on_storyteller_switch_toggled)

        self.game_master_switch = Gtk.Switch()
        self.game_master_switch.set_active(self.persona_type.get("game_master", {}).get("enabled", "False") == "True")
        self.game_master_switch.connect("notify::active", self.on_game_master_switch_toggled)

        self.chef_switch = Gtk.Switch()
        self.chef_switch.set_active(self.persona_type.get("chef", {}).get("enabled", "False") == "True")
        self.chef_switch.connect("notify::active", self.on_chef_switch_toggled)

    def create_main_switches_tab(self):
        self.main_switches_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.main_switches_box.set_margin_start(10)
        self.main_switches_box.set_margin_end(10)
        self.main_switches_box.set_margin_top(10)
        self.main_switches_box.set_margin_bottom(10)

        # List of switches and labels
        switches = [
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

        for label_text, switch in switches:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
            label = Gtk.Label(label=label_text)
            label.set_halign(Gtk.Align.START)
            hbox.append(label)
            hbox.append(switch)
            self.main_switches_box.append(hbox)

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
            if switch.get_active():
                self.add_tab(tab_name)
            else:
                self.remove_tab(tab_name)

    # Switch toggled methods

    def on_sys_info_switch_toggled(self, switch, gparam):
        sys_info_enabled = switch.get_active()
        self.general_tab.set_sys_info_enabled(sys_info_enabled)

    def on_agent_switch_toggled(self, switch, gparam):
        agent_enabled = switch.get_active()
        self.general_tab.set_agent_enabled(agent_enabled)
        # We will use this later to enable tool use by agents

    def on_user_profile_switch_toggled(self, switch, gparam):
        user_profile_enabled = switch.get_active()
        self.general_tab.set_user_profile_enabled(user_profile_enabled)

    def on_medical_persona_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_medical_persona_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_educational_persona_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_educational_persona(switch.get_active())
        self.general_tab.update_end_locked()

    def on_fitness_trainer_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_fitness_persona_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_language_practice_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_language_instructor(switch.get_active())
        self.general_tab.update_end_locked()

    def on_legal_persona_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_legal_persona_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_financial_advisor_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_financial_advisor_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_tech_support_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_tech_support_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_personal_assistant_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_personal_assistant_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_therapist_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_therapist_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_travel_guide_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_travel_guide_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_storyteller_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_storyteller_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_game_master_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_game_master_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    def on_chef_switch_toggled(self, switch, gparam):
        self.update_persona_type_tabs()
        self.general_tab.set_chef_enabled(switch.get_active())
        self.general_tab.update_end_locked()

    # Methods to add or remove tabs
    def add_tab(self, tab_name):
        if tab_name in self.tabs:
            return  # Tab already exists

        if tab_name == 'Medical':
            tab = self.create_medical_persona_tab()
        elif tab_name == 'Educational':
            tab = self.create_educational_persona_tab()
        elif tab_name == 'Fitness':
            tab = self.create_fitness_trainer_tab()
        elif tab_name == 'Language Instructor':
            tab = self.create_language_practice_tab()
        elif tab_name == 'Legal':
            tab = self.create_legal_persona_tab()
        elif tab_name == 'Financial Advisor':
            tab = self.create_financial_advisor_tab()
        elif tab_name == 'Tech Support':
            tab = self.create_tech_support_tab()
        elif tab_name == 'Personal Assistant':
            tab = self.create_personal_assistant_tab()
        elif tab_name == 'Therapist':
            tab = self.create_therapist_tab()
        elif tab_name == 'Travel Guide':
            tab = self.create_travel_guide_tab()
        elif tab_name == 'Storyteller':
            tab = self.create_storyteller_tab()
        elif tab_name == 'Game Master':
            tab = self.create_game_master_tab()
        elif tab_name == 'Chef':
            tab = self.create_chef_tab()
        else:
            return

        self.tabs[tab_name] = tab
        self.sub_notebook.append_page(tab, Gtk.Label(label=tab_name))
        self.sub_notebook.show()

    def remove_tab(self, tab_name):
        if tab_name in self.tabs:
            page_num = self.sub_notebook.page_num(self.tabs[tab_name])
            if page_num != -1:
                self.sub_notebook.remove_page(page_num)
            del self.tabs[tab_name]
            self.sub_notebook.show()

    # Methods to create tabs for each persona type
    def create_medical_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Currently, there are no additional settings for the medical persona
        label = Gtk.Label(label="No additional settings for Medical Persona.")
        label.set_halign(Gtk.Align.START)
        box.append(label)

        return box

    def create_educational_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Subject Specialization
        subject_label = Gtk.Label(label="Subject Specialization")
        self.subject_entry = Gtk.Entry()
        self.subject_entry.set_text(self.persona_type.get("educational_persona", {}).get("subject_specialization", "General"))
        subject_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        subject_box.append(subject_label)
        subject_box.append(self.subject_entry)
        box.append(subject_box)

        # Education Level
        level_label = Gtk.Label(label="Education Level")
        self.level_combo = Gtk.ComboBoxText()
        levels = ["Elementary", "Middle School", "High School", "College", "Advanced"]
        for level in levels:
            self.level_combo.append_text(level)
        level_text = self.persona_type.get("educational_persona", {}).get("education_level", "High School")
        if level_text in levels:
            self.level_combo.set_active(levels.index(level_text))
        else:
            self.level_combo.set_active(2)  # Default to "High School"
        level_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        level_box.append(level_label)
        level_box.append(self.level_combo)
        box.append(level_box)

        # Teaching Style
        style_label = Gtk.Label(label="Teaching Style")
        self.style_combo = Gtk.ComboBoxText()
        styles = ["Socratic Method", "Lecture Style", "Interactive Exercises"]
        for style in styles:
            self.style_combo.append_text(style)
        style_text = self.persona_type.get("educational_persona", {}).get("teaching_style", "Lecture Style")
        if style_text in styles:
            self.style_combo.set_active(styles.index(style_text))
        else:
            self.style_combo.set_active(1)  # Default to "Lecture Style"
        style_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        style_box.append(style_label)
        style_box.append(self.style_combo)
        box.append(style_box)

        return box

    def create_fitness_trainer_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Fitness Goal
        goal_label = Gtk.Label(label="Fitness Goal")
        self.goal_combo = Gtk.ComboBoxText()
        goals = ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"]
        for goal in goals:
            self.goal_combo.append_text(goal)
        goal_text = self.persona_type.get("fitness_persona", {}).get("fitness_goal", "Weight Loss")
        if goal_text in goals:
            self.goal_combo.set_active(goals.index(goal_text))
        else:
            self.goal_combo.set_active(0)  # Default to "Weight Loss"
        goal_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        goal_box.append(goal_label)
        goal_box.append(self.goal_combo)
        box.append(goal_box)

        # Exercise Preference
        exercise_label = Gtk.Label(label="Exercise Preference")
        self.exercise_combo = Gtk.ComboBoxText()
        exercises = ["Gym Workouts", "Home Exercises", "Yoga", "Cardio"]
        for exercise in exercises:
            self.exercise_combo.append_text(exercise)
        exercise_text = self.persona_type.get("fitness_persona", {}).get("exercise_preference", "Gym Workouts")
        if exercise_text in exercises:
            self.exercise_combo.set_active(exercises.index(exercise_text))
        else:
            self.exercise_combo.set_active(0)  # Default to "Gym Workouts"
        exercise_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        exercise_box.append(exercise_label)
        exercise_box.append(self.exercise_combo)
        box.append(exercise_box)

        return box

    def create_language_practice_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Target Language
        language_label = Gtk.Label(label="Target Language")
        self.language_entry = Gtk.Entry()
        self.language_entry.set_text(self.persona_type.get("language_instructor", {}).get("target_language", "Spanish"))
        language_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        language_box.append(language_label)
        language_box.append(self.language_entry)
        box.append(language_box)

        # Proficiency Level
        proficiency_label = Gtk.Label(label="Proficiency Level")
        self.proficiency_combo = Gtk.ComboBoxText()
        levels = ["Beginner", "Intermediate", "Advanced"]
        for level in levels:
            self.proficiency_combo.append_text(level)
        proficiency_text = self.persona_type.get("language_instructor", {}).get("proficiency_level", "Beginner")
        if proficiency_text in levels:
            self.proficiency_combo.set_active(levels.index(proficiency_text))
        else:
            self.proficiency_combo.set_active(0)  # Default to "Beginner"
        proficiency_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        proficiency_box.append(proficiency_label)
        proficiency_box.append(self.proficiency_combo)
        box.append(proficiency_box)

        return box

    # Additional tab creation methods for other personas would be added here.

    # Methods to retrieve values
    def get_values(self):
        values = {
            'sys_info_enabled': self.get_sys_info_enabled(),
            'user_profile_enabled': self.get_user_profile_enabled(),
        }

        # Collect 'type' values
        type_values = {
            'Agent': {'enabled': str(self.get_agent_enabled())},
            'medical_persona': {'enabled': str(self.get_medical_persona_enabled())},
            'educational_persona': {'enabled': str(self.get_educational_persona())},
            'fitness_persona': {'enabled': str(self.get_fitness_persona())},
            'language_instructor': {'enabled': str(self.get_language_instructor())},
            # Add other persona types here...
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
        return self.sys_info_switch.get_active()

    def get_agent_enabled(self):
        return self.agent_switch.get_active()

    def get_user_profile_enabled(self):
        return self.user_profile_switch.get_active()

    def get_medical_persona_enabled(self):
        return self.medical_persona_switch.get_active()

    def get_educational_persona(self):
        return self.educational_persona_switch.get_active()

    def get_fitness_persona(self):
        return self.fitness_trainer_switch.get_active()

    def get_language_instructor(self):
        return self.language_practice_switch.get_active()

    def get_educational_options(self):
        options = {}
        if hasattr(self, 'subject_entry'):
            options['subject_specialization'] = self.subject_entry.get_text()
        else:
            options['subject_specialization'] = 'General'
        if hasattr(self, 'level_combo'):
            options['education_level'] = self.level_combo.get_active_text()
        else:
            options['education_level'] = 'High School'
        if hasattr(self, 'style_combo'):
            options['teaching_style'] = self.style_combo.get_active_text()
        else:
            options['teaching_style'] = 'Lecture Style'
        return options

    def get_fitness_options(self):
        options = {}
        if hasattr(self, 'goal_combo'):
            options['fitness_goal'] = self.goal_combo.get_active_text()
        else:
            options['fitness_goal'] = 'Weight Loss'
        if hasattr(self, 'exercise_combo'):
            options['exercise_preference'] = self.exercise_combo.get_active_text()
        else:
            options['exercise_preference'] = 'Gym Workouts'
        return options

    def get_language_practice_options(self):
        options = {}
        if hasattr(self, 'language_entry'):
            options['target_language'] = self.language_entry.get_text()
        else:
            options['target_language'] = 'Spanish'
        if hasattr(self, 'proficiency_combo'):
            options['proficiency_level'] = self.proficiency_combo.get_active_text()
        else:
            options['proficiency_level'] = 'Beginner'
        return options
