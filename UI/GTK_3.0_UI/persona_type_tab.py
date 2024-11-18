# UI/Persona_manager/Persona_Type_Tab/persona_type_tab.py

import gi
gi.require_version('Gtk', '3.0')
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
        self.type_box.pack_start(self.sub_notebook, True, True, 0)

        # Create the switches and store them as attributes
        self.create_switches()

        # Create the main switches tab
        self.create_main_switches_tab()

        # Add the main switches tab to the notebook
        self.sub_notebook.append_page(self.main_switches_box, Gtk.Label(label="Main"))

        # Now create individual tabs for each persona type, only if the switch is ON
        if self.medical_persona_switch.get_active():
            tab = self.create_medical_persona_tab()
            self.tabs['Medical'] = tab
            self.sub_notebook.append_page(tab, Gtk.Label(label="Medical"))

        if self.educational_persona_switch.get_active():
            tab = self.create_educational_persona_tab()
            self.tabs['Educational'] = tab
            self.sub_notebook.append_page(tab, Gtk.Label(label="Educational"))

        if self.fitness_trainer_switch.get_active():
            tab = self.create_fitness_trainer_tab()
            self.tabs['Fitness'] = tab
            self.sub_notebook.append_page(tab, Gtk.Label(label="Fitness"))

        if self.language_practice_switch.get_active():
            tab = self.create_language_practice_tab()
            self.tabs['Language Instructor'] = tab
            self.sub_notebook.append_page(tab, Gtk.Label(label="Language Instructor"))

        # Ensure the notebook is displayed correctly
        self.sub_notebook.show_all()

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
        self.main_switches_box.set_border_width(10)

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
            hbox.pack_start(label, True, True, 0)
            hbox.pack_end(switch, False, False, 0)
            self.main_switches_box.pack_start(hbox, False, False, 0)

    def update_persona_type_tabs(self):
        # Existing persona types
        if self.medical_persona_switch.get_active():
            self.add_tab('Medical')
        if self.educational_persona_switch.get_active():
            self.add_tab('Educational')
        if self.fitness_trainer_switch.get_active():
            self.add_tab('Fitness')
        if self.language_practice_switch.get_active():
            self.add_tab('Language Instructor')

        # New persona types
        if self.legal_persona_switch.get_active():
            self.add_tab('Legal')
        if self.financial_advisor_switch.get_active():
            self.add_tab('Financial Advisor')
        if self.tech_support_switch.get_active():
            self.add_tab('Tech Support')
        if self.personal_assistant_switch.get_active():
            self.add_tab('Personal Assistant')
        if self.therapist_switch.get_active():
            self.add_tab('Therapist')
        if self.travel_guide_switch.get_active():
            self.add_tab('Travel Guide')
        if self.storyteller_switch.get_active():
            self.add_tab('Storyteller')
        if self.game_master_switch.get_active():
            self.add_tab('Game Master')
        if self.chef_switch.get_active():
            self.add_tab('Chef')

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
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Medical')
        else:
            self.remove_tab('Medical')
        self.general_tab.set_medical_persona_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_educational_persona_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Educational')
        else:
            self.remove_tab('Educational')
        self.general_tab.set_educational_persona(enabled)
        self.general_tab.update_end_locked()

    def on_fitness_trainer_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Fitness')
        else:
            self.remove_tab('Fitness')
        self.general_tab.set_fitness_persona_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_language_practice_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Language Instructor')
        else:
            self.remove_tab('Language Instructor')
        self.general_tab.set_language_instructor(enabled)
        self.general_tab.update_end_locked()

    def on_legal_persona_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Legal')
        else:
            self.remove_tab('Legal')
        self.general_tab.set_legal_persona_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_financial_advisor_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Financial Advisor')
        else:
            self.remove_tab('Financial Advisor')
        self.general_tab.set_financial_advisor_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_tech_support_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Tech Support')
        else:
            self.remove_tab('Tech Support')
        self.general_tab.set_tech_support_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_personal_assistant_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Personal Assistant')
        else:
            self.remove_tab('Personal Assistant')
        self.general_tab.set_personal_assistant_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_therapist_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Therapist')
        else:
            self.remove_tab('Therapist')
        self.general_tab.set_therapist_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_travel_guide_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Travel Guide')
        else:
            self.remove_tab('Travel Guide')
        self.general_tab.set_travel_guide_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_storyteller_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Storyteller')
        else:
            self.remove_tab('Storyteller')
        self.general_tab.set_storyteller_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_game_master_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Game Master')
        else:
            self.remove_tab('Game Master')
        self.general_tab.set_game_master_enabled(enabled)
        self.general_tab.update_end_locked()

    def on_chef_switch_toggled(self, switch, gparam):
        enabled = switch.get_active()
        if enabled:
            self.add_tab('Chef')
        else:
            self.remove_tab('Chef')
        self.general_tab.set_chef_enabled(enabled)
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
        self.sub_notebook.show_all()

    def remove_tab(self, tab_name):
        if tab_name in self.tabs:
            page_num = self.sub_notebook.page_num(self.tabs[tab_name])
            if page_num != -1:
                self.sub_notebook.remove_page(page_num)
            del self.tabs[tab_name]
            self.sub_notebook.show_all()

    # Methods to create tabs for each persona type
    def create_medical_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Currently, there are no additional settings for the medical persona
        label = Gtk.Label(label="No additional settings for Medical Persona.")
        label.set_halign(Gtk.Align.START)
        box.pack_start(label, False, False, 0)

        return box

    def create_educational_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Subject Specialization
        subject_label = Gtk.Label(label="Subject Specialization")
        self.subject_entry = Gtk.Entry()
        self.subject_entry.set_text(self.persona_type.get("educational_persona", {}).get("subject_specialization", "General"))
        subject_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        subject_box.pack_start(subject_label, False, False, 0)
        subject_box.pack_start(self.subject_entry, True, True, 0)
        box.pack_start(subject_box, False, False, 0)

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
        level_box.pack_start(level_label, False, False, 0)
        level_box.pack_start(self.level_combo, False, False, 0)
        box.pack_start(level_box, False, False, 0)

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
        style_box.pack_start(style_label, False, False, 0)
        style_box.pack_start(self.style_combo, False, False, 0)
        box.pack_start(style_box, False, False, 0)

        return box

    def create_fitness_trainer_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

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
        goal_box.pack_start(goal_label, False, False, 0)
        goal_box.pack_start(self.goal_combo, False, False, 0)
        box.pack_start(goal_box, False, False, 0)

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
        exercise_box.pack_start(exercise_label, False, False, 0)
        exercise_box.pack_start(self.exercise_combo, False, False, 0)
        box.pack_start(exercise_box, False, False, 0)

        return box

    def create_language_practice_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Target Language
        language_label = Gtk.Label(label="Target Language")
        self.language_entry = Gtk.Entry()
        self.language_entry.set_text(self.persona_type.get("language_instructor", {}).get("target_language", "Spanish"))
        language_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        language_box.pack_start(language_label, False, False, 0)
        language_box.pack_start(self.language_entry, True, True, 0)
        box.pack_start(language_box, False, False, 0)

        # Proficiency Level
        proficiency_label = Gtk.Label(label="Proficiency Level")
        self.proficiency_combo = Gtk.ComboBoxText()
        levels = ["Beginner", "Intermediate", "Advanced"]
        for level in levels:
            self.proficiency_combo.append_text(level)
        
        # Corrected line: Specify the key and default value
        proficiency_text = self.persona_type.get("language_instructor", {}).get("proficiency_level", "Beginner")

        if proficiency_text in levels:
            self.proficiency_combo.set_active(levels.index(proficiency_text))
        else:
            self.proficiency_combo.set_active(0)  # Default to "Beginner"

        proficiency_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        proficiency_box.pack_start(proficiency_label, False, False, 0)
        proficiency_box.pack_start(self.proficiency_combo, False, False, 0)
        box.pack_start(proficiency_box, False, False, 0)

        return box


    def create_legal_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Jurisdiction
        jurisdiction_label = Gtk.Label(label="Jurisdiction")
        self.jurisdiction_entry = Gtk.Entry()
        self.jurisdiction_entry.set_text(self.persona_type.get("legal_persona", {}).get("jurisdiction", ""))
        jurisdiction_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        jurisdiction_box.pack_start(jurisdiction_label, False, False, 0)
        jurisdiction_box.pack_start(self.jurisdiction_entry, True, True, 0)
        box.pack_start(jurisdiction_box, False, False, 0)

        # Area of Law
        area_label = Gtk.Label(label="Area of Law")
        self.area_entry = Gtk.Entry()
        self.area_entry.set_text(self.persona_type.get("legal_persona", {}).get("area_of_law", ""))
        area_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        area_box.pack_start(area_label, False, False, 0)
        area_box.pack_start(self.area_entry, True, True, 0)
        box.pack_start(area_box, False, False, 0)

        # Disclaimer
        disclaimer_label = Gtk.Label(label="Disclaimer Notice")
        self.disclaimer_entry = Gtk.Entry()
        self.disclaimer_entry.set_text(self.persona_type.get("legal_persona", {}).get("disclaimer", "This is not legal advice."))
        disclaimer_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        disclaimer_box.pack_start(disclaimer_label, False, False, 0)
        disclaimer_box.pack_start(self.disclaimer_entry, True, True, 0)
        box.pack_start(disclaimer_box, False, False, 0)

        return box

    def create_tech_support_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_border_width(10)

        # Product Specialization
        product_label = Gtk.Label(label="Product Specialization")
        self.product_entry = Gtk.Entry()
        self.product_entry.set_text(self.persona_type.get("tech_support", {}).get("product_specialization", ""))
        product_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        product_box.pack_start(product_label, False, False, 0)
        product_box.pack_start(self.product_entry, True, True, 0)
        box.pack_start(product_box, False, False, 0)

        # User Expertise Level
        expertise_label = Gtk.Label(label="User Expertise Level")
        self.expertise_combo = Gtk.ComboBoxText()
        levels = ["Beginner", "Intermediate", "Advanced"]
        for level in levels:
            self.expertise_combo.append_text(level)
        expertise_text = self.persona_type.get("tech_support", {}).get("user_expertise_level", "Beginner")
        if expertise_text in levels:
            self.expertise_combo.set_active(levels.index(expertise_text))
        else:
            self.expertise_combo.set_active(0)
        expertise_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        expertise_box.pack_start(expertise_label, False, False, 0)
        expertise_box.pack_start(self.expertise_combo, False, False, 0)
        box.pack_start(expertise_box, False, False, 0)

        # Access to Logs
        access_label = Gtk.Label(label="Access to Logs")
        self.access_switch = Gtk.Switch()
        access_state = self.persona_type.get("tech_support", {}).get("access_to_logs", "False") == "True"
        self.access_switch.set_active(access_state)
        access_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        access_box.pack_start(access_label, False, False, 0)
        access_box.pack_start(self.access_switch, False, False, 0)
        box.pack_start(access_box, False, False, 0)

        return box
    
    # In persona_type_tab.py

    def get_values(self):
        values = {
            'sys_info_enabled': self.get_sys_info_enabled(),
            'user_profile_enabled': self.get_user_profile_enabled(),
        }

        # Collect 'type' values
        type_values = {
            'Agent': {'enabled': self.get_agent_enabled()},
            'medical_persona': {'enabled': self.get_medical_persona_enabled()},
            'educational_persona': {'enabled': self.get_educational_persona()},
            'fitness_persona': {'enabled': self.get_fitness_persona()},
            'language_instructor': {'enabled': self.get_language_instructor()},
            # Add other persona types here...
        }

        if type_values['educational_persona']['enabled']:
            type_values['educational_persona'].update(self.get_educational_options())
        if type_values['fitness_persona']['enabled']:
            type_values['fitness_persona'].update(self.get_fitness_options())
        if type_values['language_instructor']['enabled']:
            type_values['language_instructor'].update(self.get_language_practice_options())

        values['type'] = type_values
        return values
    
    # Methods to retrieve values
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