# SCOUT\persona_manager.py

import os
import json
from typing import List, Optional, Dict
from modules.user_accounts.user_data_manager import UserDataManager
from SCOUT.config import ConfigManager


class PersonaManager:
    """
    The PersonaManager class is responsible for managing personas within the application.
    It loads persona data from corresponding folders and updates the current persona based on user selection.

    Attributes:
        master (object): A reference to the master application object.
        user (str): The username of the current user.
        personas (dict): A cache of loaded personas.
        current_persona (dict): The currently selected persona.
        current_system_prompt (str): The system prompt generated based on the selected persona.
    """
    
    def __init__(self, master, user: str):
        self.master = master
        self.user = user
        self.config_manager = ConfigManager
        self.logger = self.config_manager.logger
        self.persona_base_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'Personas')
        self.persona_names: List[str] = self.load_persona_names(self.persona_base_path)
        self.personas: Dict[str, dict] = {}  # Cache for loaded personas
        self.default_persona_name = "SCOUT"
        self.current_persona = None
        self.current_system_prompt = None

        # Load the default persona and generate the system prompt
        self.load_default_persona()

    def load_default_persona(self):
        """Loads and personalizes the default persona."""
        persona = self.load_persona(self.default_persona_name)
        if persona:
            self.current_persona = persona
            self.current_system_prompt = self.build_system_prompt(persona)
            self.logger.info(f"Default persona {self.default_persona_name} loaded with personalized system prompt.")
        else:
            self.logger.error(f"Failed to load default persona: {self.default_persona_name}")

    def load_persona_names(self, persona_path: str) -> List[str]:
        """Get the list of persona directories."""
        try:
            persona_names = [name for name in os.listdir(persona_path) if os.path.isdir(os.path.join(persona_path, name))]
            self.logger.info(f"Persona names loaded: {persona_names}")
            return persona_names
        except OSError as e:
            self.logger.error(f"Error loading persona names: {e}")
            return []

    def load_persona(self, persona_name: str) -> Optional[dict]:
        """
        Load a single persona from its respective folder.

        Args:
            persona_name (str): The name of the persona to load.

        Returns:
            dict: The loaded persona data, or None if loading fails.
        """
        if persona_name in self.personas:  # Check cache first
            return self.personas[persona_name]
        
        persona_folder = os.path.join(self.persona_base_path, persona_name, 'Persona')
        json_file = os.path.join(persona_folder, f'{persona_name}.json')

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as file:
                    persona_data = json.load(file)
                    self.personas[persona_name] = persona_data["persona"][0]  # Cache the loaded persona
                    self.logger.info(f"Persona {persona_name} loaded successfully from {json_file}")
                    return self.personas[persona_name]
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading persona {persona_name}: {e}")
                return None
        else:
            self.logger.error(f"JSON file for persona {persona_name} not found at {json_file}")
            return None

    def get_persona(self, persona_name: str) -> Optional[dict]:
        """Retrieve the persona by name, loading it from disk if necessary."""
        return self.load_persona(persona_name)

    def updater(self, selected_persona_name: str):
        """Update the current persona and manage conversation ID."""
        self.logger.info(f"Attempting to update persona to {selected_persona_name}.")
        
        persona = self.get_persona(selected_persona_name)
        if not persona:
            self.logger.error(f"Failed to update persona: {selected_persona_name} not found.")
            return

        self.current_persona = persona
        self.current_system_prompt = self.build_system_prompt(persona)

        self.master.system_name = selected_persona_name
        self.master.system_name_tag = selected_persona_name

        if hasattr(self.master, 'database'):
            self.master.database.generate_new_conversation_id()
            self.logger.info(f"Conversation ID updated due to persona change to {selected_persona_name}")
        else:
            self.logger.warning("ConversationHistory instance not found in master.")
        
        self.logger.info(f"Persona switched to {selected_persona_name} with new system prompt.")

    def build_system_prompt(self, persona: dict) -> str:
        """
        Builds the system prompt using user-specific information, 
        including the EMR only for medical personas and sysinfo only for sys_admin_personas.
        
        Args:
            persona (dict): The persona data to personalize.

        Returns:
            str: The personalized system prompt.
        """
        self.logger.info(f"Building system prompt for persona {persona.get('name')}")
        user_data_manager = UserDataManager(self.user)

        # General user data available to all personas
        user_data = {
            "<<name>>": self.user,
            "<<Profile>>": user_data_manager.get_profile_text(),
        }

        # Only add EMR if the persona is a medical persona
        if persona.get("medical_persona") == "True":
            self.logger.info(f"Adding EMR to medical persona: {persona['name']}")
            user_data["<<emr>>"] = user_data_manager.get_emr()
        else:
            self.logger.info(f"EMR not added for non-medical persona: {persona['name']}")
            user_data["<<emr>>"] = "EMR not available for this persona"

        # Only add sysinfo if the persona is a sys_admin_persona
        if persona.get("Sys_admin_persona") == "True":
            self.logger.info(f"Adding system info to sys_admin_persona: {persona['name']}")
            user_data["<<sysinfo>>"] = user_data_manager.get_system_info()
        else:
            self.logger.info(f"System info not added for non-sys_admin_persona: {persona['name']}")
            user_data["<<sysinfo>>"] = "System info not available for this persona"

        # Replace placeholders in persona content with user data
        personalized_content = persona["content"]
        for placeholder, data in user_data.items():
            personalized_content = personalized_content.replace(placeholder, data)

        return personalized_content

    def show_message(self, role: str, message: str):
        """Display a message using the chat component, if available."""
        if hasattr(self.master, 'chat_component'):
            self.master.chat_component.show_message(role, message)
        else:
            self.logger.warning("ChatComponent instance not found in master. Unable to display message.")
