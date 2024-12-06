# PS6UI/persona_management.py

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QLineEdit, QDialog, QTabWidget
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt

from PS6UI.Persona_manager.General_Tab.general_tab import GeneralTab
from PS6UI.Persona_manager.Persona_Type_Tab.persona_type_tab import PersonaTypeTab

class PersonaManagement:
    def __init__(self, ATLAS, parent_window):
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.persona_window = None

    def show_persona_menu(self):
        self.persona_window = QDialog(self.parent_window)
        self.persona_window.setWindowTitle("Select Persona")
        self.persona_window.resize(150, 600)

        box = QVBoxLayout(self.persona_window)
        box.setSpacing(10)

        persona_names = self.ATLAS.get_persona_names()

        for persona_name in persona_names:
            hbox = QHBoxLayout()
            label = QLabel(persona_name)
            # Click on label to select persona
            label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            # Instead of Gtk.GestureClick, we can use mousePressEvent or we can wrap label in a button.
            # Simpler: use a QPushButton with the persona name. Let's do that:
            persona_button = QPushButton(persona_name)
            persona_button.setStyleSheet("text-align: left; background: none; border: none;")
            persona_button.clicked.connect(lambda checked, p=persona_name: self.select_persona(p))
            hbox.addWidget(persona_button)

            # Settings icon
            settings_icon_path = "Icons/settings.png"
            if not os.path.exists(settings_icon_path):
                settings_icon_path = "Icons/default.png"  # fallback
            settings_icon_btn = QPushButton()
            settings_icon_btn.setFlat(True)
            pixmap = QPixmap(settings_icon_path).scaled(16,16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            settings_icon_btn.setIcon(QIcon(pixmap))
            settings_icon_btn.setIconSize(pixmap.size())
            settings_icon_btn.clicked.connect(lambda checked, p=persona_name: self.open_persona_settings(p))
            hbox.addWidget(settings_icon_btn)

            # Wrap this hbox in a QWidget to add to layout
            w = QWidget()
            w.setLayout(hbox)
            box.addWidget(w)

        self.persona_window.exec()

    def select_persona(self, persona):
        self.ATLAS.load_persona(persona)
        print(f"Persona '{persona}' selected with system prompt:\n{self.ATLAS.persona_manager.current_system_prompt}")
        if self.persona_window:
            self.persona_window.close()

    def open_persona_settings(self, persona_name):
        if self.persona_window:
            self.persona_window.close()

        persona = self.ATLAS.persona_manager.get_persona(persona_name)
        self.show_persona_settings(persona)

    def show_persona_settings(self, persona):
        settings_window = QDialog(self.parent_window)
        settings_window.setWindowTitle(f"Settings for {persona.get('name')}")
        settings_window.resize(500, 800)

        # apply_css_styling() -> Convert to Qt styling if needed.
        # self.apply_css_styling()  # Comment out as this is GTK CSS
        settings_window.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-size:14px;
            }
            QLineEdit, QTextEdit {
                background-color: #1c1c1c;
                color:white;
                border:none;
                border-radius:5px;
                padding:5px;
            }
            QPushButton {
                background-color: #1c1c1c;
                color: white;
                border-radius:5px;
                margin:5px;
            }
            QPushButton:hover {
                background-color:#4a90d9;
            }
        """)

        main_layout = QVBoxLayout(settings_window)
        main_layout.setSpacing(10)

        # Replace Gtk.Stack with QTabWidget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # General Tab
        self.general_tab = GeneralTab(persona)
        general_widget = self.general_tab.get_widget()

        scroll_area_general = QScrollArea()
        scroll_area_general.setWidgetResizable(True)
        scroll_area_general.setWidget(general_widget)
        tab_widget.addTab(scroll_area_general, "General")

        # Persona Type Tab
        self.persona_type_tab = PersonaTypeTab(persona, self.general_tab)
        type_widget = self.persona_type_tab.get_widget()

        scroll_area_type = QScrollArea()
        scroll_area_type.setWidgetResizable(True)
        scroll_area_type.setWidget(type_widget)
        tab_widget.addTab(scroll_area_type, "Persona Type")

        # Provider & Model Tab
        provider_model_box = self.create_provider_model_tab(persona)
        tab_widget.addTab(provider_model_box, "Provider & Model")

        # Speech & Voice Tab
        speech_voice_box = self.create_speech_voice_tab(persona)
        tab_widget.addTab(speech_voice_box, "Speech & Voice")

        save_button = QPushButton("Save")
        save_button.clicked.connect(lambda: self.save_persona_settings(persona, settings_window))
        main_layout.addWidget(save_button)

        settings_window.exec()

    def create_provider_model_tab(self, persona):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Provider
        provider_h = QHBoxLayout()
        provider_label = QLabel("Provider")
        self.provider_entry = QLineEdit()
        self.provider_entry.setText(persona.get("provider", "openai"))
        provider_h.addWidget(provider_label)
        provider_h.addWidget(self.provider_entry)
        layout.addLayout(provider_h)

        # Model
        model_h = QHBoxLayout()
        model_label = QLabel("Model")
        self.model_entry = QLineEdit()
        self.model_entry.setText(persona.get("model", "gpt-4"))
        model_h.addWidget(model_label)
        model_h.addWidget(self.model_entry)
        layout.addLayout(model_h)

        return widget

    def create_speech_voice_tab(self, persona):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Speech Provider
        speech_h = QHBoxLayout()
        speech_provider_label = QLabel("Speech Provider")
        self.speech_provider_entry = QLineEdit()
        self.speech_provider_entry.setText(persona.get("Speech_provider", "11labs"))
        speech_h.addWidget(speech_provider_label)
        speech_h.addWidget(self.speech_provider_entry)
        layout.addLayout(speech_h)

        # Voice
        voice_h = QHBoxLayout()
        voice_label = QLabel("Voice")
        self.voice_entry = QLineEdit()
        self.voice_entry.setText(persona.get("voice", "jack"))
        voice_h.addWidget(voice_label)
        voice_h.addWidget(self.voice_entry)
        layout.addLayout(voice_h)

        return widget

    def save_persona_settings(self, persona, settings_window):
        # Get values from general_tab
        name = self.general_tab.get_name()
        meaning = self.general_tab.get_meaning()
        editable_content = self.general_tab.get_editable_content()
        start_locked_content = self.general_tab.get_start_locked()
        end_locked_content = self.general_tab.get_end_locked()

        # Get values from persona_type_tab
        values = self.persona_type_tab.get_values()
        sys_info_enabled = values.get('sys_info_enabled', False)
        user_profile_enabled = values.get('user_profile_enabled', False)
        persona_type_values = values.get('type', {})

        # Provider & Model
        provider = self.provider_entry.text()
        model = self.model_entry.text()

        # Speech & Voice
        speech_provider = self.speech_provider_entry.text()
        voice = self.voice_entry.text()

        # Now save to persona
        persona['name'] = name
        persona['meaning'] = meaning

        # Update content parts
        content = persona.get('content', {})
        content['start_locked'] = start_locked_content
        content['editable_content'] = editable_content
        content['end_locked'] = end_locked_content
        persona['content'] = content

        # Save top-level flags
        persona['sys_info_enabled'] = "True" if sys_info_enabled else "False"
        persona['user_profile_enabled'] = "True" if user_profile_enabled else "False"

        # Save 'type' dictionary
        persona['type'] = {}
        persona_types = [
            'Agent', 'medical_persona', 'educational_persona', 'fitness_persona', 'language_instructor',
            'legal_persona', 'financial_advisor', 'tech_support', 'personal_assistant', 'therapist',
            'travel_guide', 'storyteller', 'game_master', 'chef'
        ]
        for key in persona_types:
            enabled = persona_type_values.get(key, {}).get('enabled', False)
            persona['type'][key] = {"enabled": str(enabled)}

        # Additional keys logic
        additional_keys = {
            'educational_persona': ['subject_specialization', 'education_level', 'teaching_style'],
            'fitness_persona': ['fitness_goal', 'exercise_preference'],
            'language_instructor': ['target_language', 'proficiency_level'],
            'legal_persona': ['jurisdiction', 'area_of_law', 'disclaimer'],
            'financial_advisor': ['investment_goals', 'risk_tolerance', 'time_horizon'],
            'tech_support': ['product_specialization', 'user_expertise_level', 'access_to_logs'],
            'personal_assistant': ['time_zone', 'communication_style', 'access_to_calendar'],
            'therapist': ['therapy_style', 'session_length', 'confidentiality'],
            'travel_guide': ['destination_preferences', 'travel_style', 'interests'],
            'storyteller': ['genre', 'audience_age_group', 'story_length'],
            'game_master': ['game_type', 'difficulty_level', 'theme'],
            'chef': ['cuisine_preferences', 'dietary_restrictions', 'skill_level']
        }

        for persona_type, keys in additional_keys.items():
            if persona['type'][persona_type]['enabled'] == "True":
                for key in keys:
                    if key in persona_type_values.get(persona_type, {}):
                        persona['type'][persona_type][key] = persona_type_values[persona_type][key]
                    else:
                        persona['type'][persona_type].pop(key, None)

        persona['provider'] = provider
        persona['model'] = model
        persona['Speech_provider'] = speech_provider
        persona['voice'] = voice

        self.ATLAS.persona_manager.update_persona(persona)
        print(f"Settings for {name} saved!")
        settings_window.close()

    # If needed, you can implement a Qt-based styling method here.
    def apply_css_styling(self):
        # This was GTK-specific. With Qt, we use setStyleSheet as shown above.
        pass
