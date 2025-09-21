# UI/Persona_manager/Persona_Type_Tab/persona_type_tab.py

from typing import Callable, Dict, Optional

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk


class PersonaTypeTab:
    def __init__(self, persona, general_tab):
        self.persona = persona
        self.general_tab = general_tab
        self.persona_type = self.persona.get('type', {})
        self.tabs = {}  # name -> page widget (dynamic tabs only; "Main" handled separately)
        self._restoring_tab = False  # guard to avoid races while restoring
        self._switch_handlers: Dict[Gtk.Switch, int] = {}
        self.build_ui()

    # ----------------------------- UI Builders -----------------------------

    def build_ui(self):
        self.type_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.type_box.set_margin_start(10)
        self.type_box.set_margin_end(10)
        self.type_box.set_margin_top(10)
        self.type_box.set_margin_bottom(10)

        # Notebook holds Main (switches) + per-persona sub-tabs
        self.sub_notebook = Gtk.Notebook()
        self.sub_notebook.set_tab_pos(Gtk.PositionType.TOP)
        self.sub_notebook.set_scrollable(True)
        self.sub_notebook.set_tooltip_text(
            "Switch persona types on the Main tab. Tabs appear here when a persona type is enabled."
        )
        self.sub_notebook.connect("switch-page", self._on_switch_page)

        # Header hint
        header = Gtk.Label(
            label="Toggle persona capabilities on the Main tab. "
                  "When a switch is enabled, an options tab appears."
        )
        header.set_wrap(True)
        header.set_xalign(0.0)
        header.set_tooltip_text("Enable a persona type to configure its specific options.")

        self.type_box.append(header)
        self.type_box.append(self.sub_notebook)

        # Create switches and the Main tab
        self.create_switches()
        self.create_main_switches_tab()
        # Keep a handle to the main tab content
        self._main_tab = self.main_switches_box
        self.sub_notebook.append_page(self._main_tab, Gtk.Label(label="Main"))

        # Initialize persona type tabs according to current state
        self.update_persona_type_tabs()

        # Restore last opened tab if present
        self._restore_last_opened_tab()

    def get_widget(self):
        return self.type_box

    # ----------------------------- Helpers -----------------------------

    def _row(self, label_text: str, widget: Gtk.Widget, tooltip: str = "") -> Gtk.Box:
        """Make a horizontal row with a left-aligned label and a widget."""
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        label = Gtk.Label(label=label_text)
        label.set_halign(Gtk.Align.START)
        label.set_xalign(0.0)
        if tooltip:
            label.set_tooltip_text(tooltip)
            widget.set_tooltip_text(tooltip)
        hbox.append(label)
        hbox.append(widget)
        return hbox

    def _switch(self, active: bool, tooltip: str, on_toggle):
        """Create a Gtk.Switch with tooltip and toggle handler."""
        sw = Gtk.Switch()
        sw.set_active(active)
        if tooltip:
            sw.set_tooltip_text(tooltip)
        handler_id = sw.connect("notify::active", on_toggle)
        self._switch_handlers[sw] = handler_id
        return sw

    def _set_switch_active(self, switch: Gtk.Switch, value: bool):
        handler_id = self._switch_handlers.get(switch)
        if handler_id is not None:
            switch.handler_block(handler_id)
        switch.set_active(value)
        if handler_id is not None:
            switch.handler_unblock(handler_id)

    def _set_all_switches(self, value: bool):
        """Set all persona-related switches to the same boolean value."""
        # Global/system switches — leave System Info & User Profile as-is; mass-toggle persona modes + Agent
        persona_switches = [
            self.agent_switch,
            self.medical_persona_switch, self.educational_persona_switch,
            self.fitness_trainer_switch, self.language_practice_switch,
            self.legal_persona_switch, self.financial_advisor_switch,
            self.tech_support_switch, self.personal_assistant_switch,
            self.therapist_switch, self.travel_guide_switch,
            self.storyteller_switch, self.game_master_switch, self.chef_switch
        ]
        for sw in persona_switches:
            # Avoid re-entrancy storms; only flip when different
            if sw.get_active() != value:
                sw.set_active(value)

    def _page_label_text(self, page_widget: Gtk.Widget) -> str:
        """Return the text of the tab label for a given page widget."""
        label_widget = self.sub_notebook.get_tab_label(page_widget)
        if isinstance(label_widget, Gtk.Label):
            return label_widget.get_text()
        return ""

    def _find_page_by_name(self, name: str):
        """Return (page_widget, page_index) for a tab name, or (None, -1)."""
        pages = self.sub_notebook.get_n_pages()
        for i in range(pages):
            child = self.sub_notebook.get_nth_page(i)
            if self._page_label_text(child) == name:
                return child, i
        return None, -1

    def _save_last_opened_tab(self, name: str):
        ui = self.persona.setdefault('ui_state', {})
        ui['persona_type_last_tab'] = name

    def _restore_last_opened_tab(self):
        """Restore last opened sub-tab from persona['ui_state'] if available."""
        ui = self.persona.get('ui_state') or {}
        last = ui.get('persona_type_last_tab')
        # Delay until tabs are populated
        if last:
            # Guard to prevent saving while we are restoring
            self._restoring_tab = True
            try:
                page, idx = self._find_page_by_name(last)
                if page is not None and idx >= 0:
                    self.sub_notebook.set_current_page(idx)
                else:
                    # Fallback to Main if missing
                    page, idx = self._find_page_by_name("Main")
                    if idx >= 0:
                        self.sub_notebook.set_current_page(idx)
            finally:
                self._restoring_tab = False

    # ----------------------------- Switches -----------------------------

    def create_switches(self):
        # Global/system switches
        self.sys_info_switch = self._switch(
            active=self.persona.get("sys_info_enabled", "False") == "True",
            tooltip="Allow the persona to include host/system info context (off by default).",
            on_toggle=self.on_sys_info_switch_toggled,
        )
        self.agent_switch = self._switch(
            active=self.persona_type.get("Agent", {}).get("enabled", "False") == "True",
            tooltip="Enable Agent behavior (tool use / multi-step planning).",
            on_toggle=self.on_agent_switch_toggled,
        )
        self.user_profile_switch = self._switch(
            active=self.persona.get("user_profile_enabled", "False") == "True",
            tooltip="Permit use of stored user profile context for responses.",
            on_toggle=self.on_user_profile_switch_toggled,
        )

        # Persona type switches
        self.medical_persona_switch = self._switch(
            active=self.persona_type.get("medical_persona", {}).get("enabled", "False") == "True",
            tooltip="Medical persona (health info & safety-aware tone).",
            on_toggle=self.on_medical_persona_switch_toggled,
        )
        self.educational_persona_switch = self._switch(
            active=self.persona_type.get("educational_persona", {}).get("enabled", "False") == "True",
            tooltip="Educational persona (teaching-oriented explanations).",
            on_toggle=self.on_educational_persona_switch_toggled,
        )
        self.fitness_trainer_switch = self._switch(
            active=self.persona_type.get("fitness_persona", {}).get("enabled", "False") == "True",
            tooltip="Fitness persona (programs, exercises, goals).",
            on_toggle=self.on_fitness_trainer_switch_toggled,
        )
        self.language_practice_switch = self._switch(
            active=self.persona_type.get("language_instructor", {}).get("enabled", "False") == "True",
            tooltip="Language instructor persona (practice & drills).",
            on_toggle=self.on_language_practice_switch_toggled,
        )
        self.legal_persona_switch = self._switch(
            active=self.persona_type.get("legal_persona", {}).get("enabled", "False") == "True",
            tooltip="Legal persona (general information, not legal advice).",
            on_toggle=self.on_legal_persona_switch_toggled,
        )
        self.financial_advisor_switch = self._switch(
            active=self.persona_type.get("financial_advisor", {}).get("enabled", "False") == "True",
            tooltip="Financial advisor persona (general info, not financial advice).",
            on_toggle=self.on_financial_advisor_switch_toggled,
        )
        self.tech_support_switch = self._switch(
            active=self.persona_type.get("tech_support", {}).get("enabled", "False") == "True",
            tooltip="Tech support persona (troubleshooting & diagnostics).",
            on_toggle=self.on_tech_support_switch_toggled,
        )
        self.personal_assistant_switch = self._switch(
            active=self.persona_type.get("personal_assistant", {}).get("enabled", "False") == "True",
            tooltip="Personal assistant persona (organization & reminders).",
            on_toggle=self.on_personal_assistant_switch_toggled,
        )
        self.therapist_switch = self._switch(
            active=self.persona_type.get("therapist", {}).get("enabled", "False") == "True",
            tooltip="Supportive, reflective style for wellness topics (not a substitute for therapy).",
            on_toggle=self.on_therapist_switch_toggled,
        )
        self.travel_guide_switch = self._switch(
            active=self.persona_type.get("travel_guide", {}).get("enabled", "False") == "True",
            tooltip="Travel guide persona (itineraries, tips, highlights).",
            on_toggle=self.on_travel_guide_switch_toggled,
        )
        self.storyteller_switch = self._switch(
            active=self.persona_type.get("storyteller", {}).get("enabled", "False") == "True",
            tooltip="Storyteller persona (creative writing & narratives).",
            on_toggle=self.on_storyteller_switch_toggled,
        )
        self.game_master_switch = self._switch(
            active=self.persona_type.get("game_master", {}).get("enabled", "False") == "True",
            tooltip="Game master persona (RPG facilitation & encounters).",
            on_toggle=self.on_game_master_switch_toggled,
        )
        self.chef_switch = self._switch(
            active=self.persona_type.get("chef", {}).get("enabled", "False") == "True",
            tooltip="Chef persona (recipes, substitutions, techniques).",
            on_toggle=self.on_chef_switch_toggled,
        )

    def create_main_switches_tab(self):
        self.main_switches_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.main_switches_box.set_margin_start(6)
        self.main_switches_box.set_margin_end(6)
        self.main_switches_box.set_margin_top(6)
        self.main_switches_box.set_margin_bottom(6)

        # Row: Enable all / Disable all
        mass_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        enable_btn = Gtk.Button(label="Enable all")
        enable_btn.set_tooltip_text("Enable Agent and all persona types.")
        enable_btn.connect("clicked", lambda _b: self._set_all_switches(True))
        disable_btn = Gtk.Button(label="Disable all")
        disable_btn.set_tooltip_text("Disable Agent and all persona types.")
        disable_btn.connect("clicked", lambda _b: self._set_all_switches(False))
        mass_box.append(enable_btn)
        mass_box.append(disable_btn)
        self.main_switches_box.append(mass_box)

        # Global
        self.main_switches_box.append(self._row(
            "System Info Enabled", self.sys_info_switch,
            "Allow the persona to use host/system info context."
        ))
        self.main_switches_box.append(self._row(
            "Agent", self.agent_switch,
            "Enable Agent behavior (tool use / multi-step planning)."
        ))
        self.main_switches_box.append(self._row(
            "User Profile Enabled", self.user_profile_switch,
            "Permit the persona to use stored user profile details."
        ))

        # Persona types
        self.main_switches_box.append(self._row("Medical Persona", self.medical_persona_switch,
                                                "Enable specialized medical tone and safety cues."))
        self.main_switches_box.append(self._row("Educational Persona", self.educational_persona_switch,
                                                "Enable teaching-oriented explanations and scaffolding."))
        self.main_switches_box.append(self._row("Fitness Persona", self.fitness_trainer_switch,
                                                "Enable fitness planning and exercise coaching."))
        self.main_switches_box.append(self._row("Language Instructor Persona", self.language_practice_switch,
                                                "Enable language practice and drills."))
        self.main_switches_box.append(self._row("Legal Persona", self.legal_persona_switch,
                                                "Enable legal-oriented information (not legal advice)."))
        self.main_switches_box.append(self._row("Financial Advisor Persona", self.financial_advisor_switch,
                                                "Enable financial topics (general info, not advice)."))
        self.main_switches_box.append(self._row("Tech Support Persona", self.tech_support_switch,
                                                "Enable troubleshooting and diagnostics flow."))
        self.main_switches_box.append(self._row("Personal Assistant Persona", self.personal_assistant_switch,
                                                "Enable organization, reminders, and task support."))
        self.main_switches_box.append(self._row("Therapist Persona", self.therapist_switch,
                                                "Enable supportive, reflective conversational style."))
        self.main_switches_box.append(self._row("Travel Guide Persona", self.travel_guide_switch,
                                                "Enable travel suggestions and itinerary planning."))
        self.main_switches_box.append(self._row("Storyteller Persona", self.storyteller_switch,
                                                "Enable creative writing and narrative generation."))
        self.main_switches_box.append(self._row("Game Master Persona", self.game_master_switch,
                                                "Enable RPG facilitation, encounters, and narration."))
        self.main_switches_box.append(self._row("Chef Persona", self.chef_switch,
                                                "Enable recipes, substitutions, and kitchen techniques."))

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

        # Dynamic add/remove persona tabs
        for tab_name, switch in persona_types.items():
            if switch.get_active():
                self.add_tab(tab_name)
            else:
                self.remove_tab(tab_name)

        # Agent -> "Tools" placeholder sub-tab
        if self.agent_switch.get_active():
            self.add_tools_tab()
        else:
            self.remove_tools_tab()

    # ----------------------------- Signal Handlers -----------------------------

    def _on_switch_page(self, _notebook, _page, page_num):
        # Save the tab name when user switches pages
        if self._restoring_tab:
            return
        child = self.sub_notebook.get_nth_page(page_num)
        name = self._page_label_text(child)
        if name:
            self._save_last_opened_tab(name)

    def on_sys_info_switch_toggled(self, switch, _gparam):
        self._process_toggle(switch, lambda: self.general_tab.set_sys_info_enabled(switch.get_active()))

    def on_agent_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_agent_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_user_profile_switch_toggled(self, switch, _gparam):
        self._process_toggle(switch, lambda: self.general_tab.set_user_profile_enabled(switch.get_active()))

    def on_medical_persona_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_medical_persona_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_educational_persona_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_educational_persona(
                switch.get_active(),
                self.get_educational_options() if switch.get_active() else None,
            ),
            update_tabs=True,
        )

    def on_fitness_trainer_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_fitness_persona_enabled(
                switch.get_active(),
                self.get_fitness_options() if switch.get_active() else None,
            ),
            update_tabs=True,
        )

    def on_language_practice_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_language_instructor(
                switch.get_active(),
                self.get_language_practice_options() if switch.get_active() else None,
            ),
            update_tabs=True,
        )

    def on_legal_persona_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_legal_persona_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_financial_advisor_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_financial_advisor_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_tech_support_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_tech_support_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_personal_assistant_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_personal_assistant_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_therapist_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_therapist_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_travel_guide_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_travel_guide_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_storyteller_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_storyteller_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_game_master_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_game_master_enabled(switch.get_active()),
            update_tabs=True,
        )

    def on_chef_switch_toggled(self, switch, _gparam):
        self._process_toggle(
            switch,
            lambda: self.general_tab.set_chef_enabled(switch.get_active()),
            update_tabs=True,
        )

    def _process_toggle(
        self,
        switch: Gtk.Switch,
        action: Callable[[], bool],
        update_tabs: bool = False,
    ) -> None:
        desired_state = switch.get_active()
        success = action()
        if not success:
            self._set_switch_active(switch, not desired_state)
        self.refresh_from_persona()
        if success and update_tabs:
            self.update_persona_type_tabs()

    def refresh_from_persona(self):
        self.persona_type = self.persona.get('type', {})

        def _enabled(flag: str) -> bool:
            return self.persona_type.get(flag, {}).get('enabled', 'False') == 'True'

        self._set_switch_active(self.sys_info_switch, self.persona.get("sys_info_enabled", "False") == "True")
        self._set_switch_active(self.user_profile_switch, self.persona.get("user_profile_enabled", "False") == "True")
        self._set_switch_active(self.agent_switch, _enabled('Agent'))
        self._set_switch_active(self.medical_persona_switch, _enabled('medical_persona'))
        self._set_switch_active(self.educational_persona_switch, _enabled('educational_persona'))
        self._set_switch_active(self.fitness_trainer_switch, _enabled('fitness_persona'))
        self._set_switch_active(self.language_practice_switch, _enabled('language_instructor'))
        self._set_switch_active(self.legal_persona_switch, _enabled('legal_persona'))
        self._set_switch_active(self.financial_advisor_switch, _enabled('financial_advisor'))
        self._set_switch_active(self.tech_support_switch, _enabled('tech_support'))
        self._set_switch_active(self.personal_assistant_switch, _enabled('personal_assistant'))
        self._set_switch_active(self.therapist_switch, _enabled('therapist'))
        self._set_switch_active(self.travel_guide_switch, _enabled('travel_guide'))
        self._set_switch_active(self.storyteller_switch, _enabled('storyteller'))
        self._set_switch_active(self.game_master_switch, _enabled('game_master'))
        self._set_switch_active(self.chef_switch, _enabled('chef'))

        educational = self.persona_type.get('educational_persona', {})
        if hasattr(self, 'subject_entry'):
            self.subject_entry.set_text(educational.get('subject_specialization', 'General'))
        if hasattr(self, 'level_combo'):
            self._set_combo_active(self.level_combo, educational.get('education_level', 'High School'))
        if hasattr(self, 'style_combo'):
            self._set_combo_active(self.style_combo, educational.get('teaching_style', 'Lecture Style'))

        fitness = self.persona_type.get('fitness_persona', {})
        if hasattr(self, 'goal_combo'):
            self._set_combo_active(self.goal_combo, fitness.get('fitness_goal', 'Weight Loss'))
        if hasattr(self, 'exercise_combo'):
            self._set_combo_active(self.exercise_combo, fitness.get('exercise_preference', 'Gym Workouts'))

        language = self.persona_type.get('language_instructor', {})
        if hasattr(self, 'language_entry'):
            self.language_entry.set_text(language.get('target_language', 'Spanish'))
        if hasattr(self, 'proficiency_combo'):
            self._set_combo_active(self.proficiency_combo, language.get('proficiency_level', 'Beginner'))

    def _set_combo_active(self, combo: Gtk.ComboBoxText, text: Optional[str]):
        if combo is None or text is None:
            return
        if combo.get_active_text() == text:
            return
        model = combo.get_model()
        if model is None:
            return
        for index, row in enumerate(model):
            if row[0] == text:
                combo.set_active(index)
                return

    # ----------------------------- Tab Mgmt -----------------------------

    def add_tab(self, tab_name):
        if tab_name in self.tabs:
            return  # already exists

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
        elif tab_name == 'Tools':
            tab = self.create_tools_tab()
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

    # ----------------------------- Persona Tabs -----------------------------

    def _simple_info_tab(self, text: str, tooltip: str = "") -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        label = Gtk.Label(label=text)
        label.set_wrap(True)
        label.set_xalign(0.0)
        if tooltip:
            label.set_tooltip_text(tooltip)
            box.set_tooltip_text(tooltip)
        box.append(label)
        return box

    def create_medical_persona_tab(self):
        return self._simple_info_tab(
            "No additional settings for Medical Persona.",
            "Medical persona focuses on careful, safety-aware responses."
        )

    def create_educational_persona_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Subject Specialization
        self.subject_entry = Gtk.Entry()
        self.subject_entry.set_placeholder_text("e.g., Mathematics, Biology, Literature")
        self.subject_entry.set_text(self.persona_type.get("educational_persona", {}).get("subject_specialization", "General"))
        box.append(self._row("Subject Specialization", self.subject_entry,
                             "Primary subject area for teaching."))

        # Education Level
        self.level_combo = Gtk.ComboBoxText()
        levels = ["Elementary", "Middle School", "High School", "College", "Advanced"]
        for level in levels:
            self.level_combo.append_text(level)
        level_text = self.persona_type.get("educational_persona", {}).get("education_level", "High School")
        self.level_combo.set_active(levels.index(level_text) if level_text in levels else 2)
        self.level_combo.set_tooltip_text("Typical learner level to target.")
        box.append(self._row("Education Level", self.level_combo,
                             "Choose the learner level to tailor explanations."))

        # Teaching Style
        self.style_combo = Gtk.ComboBoxText()
        styles = ["Socratic Method", "Lecture Style", "Interactive Exercises"]
        for style in styles:
            self.style_combo.append_text(style)
        style_text = self.persona_type.get("educational_persona", {}).get("teaching_style", "Lecture Style")
        self.style_combo.set_active(styles.index(style_text) if style_text in styles else 1)
        self.style_combo.set_tooltip_text("Preferred teaching approach.")
        box.append(self._row("Teaching Style", self.style_combo,
                             "Select the instructional style for lessons."))

        return box

    def create_fitness_trainer_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Fitness Goal
        self.goal_combo = Gtk.ComboBoxText()
        goals = ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"]
        for goal in goals:
            self.goal_combo.append_text(goal)
        goal_text = self.persona_type.get("fitness_persona", {}).get("fitness_goal", "Weight Loss")
        self.goal_combo.set_active(goals.index(goal_text) if goal_text in goals else 0)
        self.goal_combo.set_tooltip_text("Primary training target.")
        box.append(self._row("Fitness Goal", self.goal_combo,
                             "What outcome the plan should optimize for."))

        # Exercise Preference
        self.exercise_combo = Gtk.ComboBoxText()
        exercises = ["Gym Workouts", "Home Exercises", "Yoga", "Cardio"]
        for exercise in exercises:
            self.exercise_combo.append_text(exercise)
        exercise_text = self.persona_type.get("fitness_persona", {}).get("exercise_preference", "Gym Workouts")
        self.exercise_combo.set_active(exercises.index(exercise_text) if exercise_text in exercises else 0)
        self.exercise_combo.set_tooltip_text("Preferred training modality.")
        box.append(self._row("Exercise Preference", self.exercise_combo,
                             "Choose the style of workouts to prioritize."))

        return box

    def create_language_practice_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_start(10)
        box.set_margin_end(10)
        box.set_margin_top(10)
        box.set_margin_bottom(10)

        # Target Language
        self.language_entry = Gtk.Entry()
        self.language_entry.set_placeholder_text("e.g., Spanish, Japanese, French")
        self.language_entry.set_text(self.persona_type.get("language_instructor", {}).get("target_language", "Spanish"))
        box.append(self._row("Target Language", self.language_entry,
                             "Language the user wants to practice."))

        # Proficiency Level
        self.proficiency_combo = Gtk.ComboBoxText()
        levels = ["Beginner", "Intermediate", "Advanced"]
        for level in levels:
            self.proficiency_combo.append_text(level)
        proficiency_text = self.persona_type.get("language_instructor", {}).get("proficiency_level", "Beginner")
        self.proficiency_combo.set_active(levels.index(proficiency_text) if proficiency_text in levels else 0)
        self.proficiency_combo.set_tooltip_text("Current learner proficiency.")
        box.append(self._row("Proficiency Level", self.proficiency_combo,
                             "Used to scale difficulty and vocabulary."))

        return box

    # Optional simple tabs (placeholders) for personas without extra settings yet
    def create_legal_persona_tab(self):
        return self._simple_info_tab(
            "No additional settings for Legal Persona.",
            "Provides general legal information; not a substitute for professional legal advice."
        )

    def create_financial_advisor_tab(self):
        return self._simple_info_tab(
            "No additional settings for Financial Advisor.",
            "Provides general financial information; not investment advice."
        )

    def create_tech_support_tab(self):
        return self._simple_info_tab(
            "No additional settings for Tech Support.",
            "Focus on troubleshooting steps and system diagnostics."
        )

    def create_personal_assistant_tab(self):
        return self._simple_info_tab(
            "No additional settings for Personal Assistant.",
            "Helps with organization, notes, and reminders."
        )

    def create_therapist_tab(self):
        return self._simple_info_tab(
            "No additional settings for Therapist.",
            "Supportive and reflective style; not a substitute for therapy."
        )

    def create_travel_guide_tab(self):
        return self._simple_info_tab(
            "No additional settings for Travel Guide.",
            "Trip suggestions, highlights, and logistics guidance."
        )

    def create_storyteller_tab(self):
        return self._simple_info_tab(
            "No additional settings for Storyteller.",
            "Creative writing, prompts, and narrative arcs."
        )

    def create_game_master_tab(self):
        return self._simple_info_tab(
            "No additional settings for Game Master.",
            "Encounters, narration, and rules facilitation for RPGs."
        )

    def create_chef_tab(self):
        return self._simple_info_tab(
            "No additional settings for Chef.",
            "Recipes, substitutions, and cooking techniques."
        )

    def create_tools_tab(self):
        return self._simple_info_tab(
            "This is a placeholder for Agent Tools.\n"
            "Future updates may expose tool permissions, allowed actions, sandboxing, and safety settings.",
            "Configure which tools the Agent persona can use."
        )

    # Agent "Tools" special-case
    def add_tools_tab(self):
        if 'Tools' not in self.tabs:
            tab = self.create_tools_tab()
            self.tabs['Tools'] = tab
            self.sub_notebook.append_page(tab, Gtk.Label(label='Tools'))
            self.sub_notebook.show()

    def remove_tools_tab(self):
        self.remove_tab('Tools')

    # ----------------------------- Value Getters -----------------------------

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
            # Add other persona types here if you later add options to them.
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
