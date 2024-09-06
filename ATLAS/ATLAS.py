# ATLAS/ATLAS.py

from ATLAS.config import ConfigManager
from ATLAS.persona_manager import PersonaManager

class ATLAS:
    def __init__(self):
        self.config_manager = ConfigManager
        self.logger = self.config_manager.logger
        self.persona_path = "/home/bib/Projects/LB/modules/Personas"
        self.user = "bib"  # Example user, adjust as needed

        # Instantiate PersonaManager
        self.persona_manager = PersonaManager(master=self, user=self.user)
        
    def get_persona_names(self):
        """Retrieve persona names from the PersonaManager."""
        return self.persona_manager.persona_names

    def load_persona(self, persona):
        """Delegate loading persona to PersonaManager."""
        self.logger.info(f"Loading persona: {persona}")
        self.persona_manager.updater(persona, self.user)

    def log_history(self):
        """Example of handling history-related functionality."""
        self.logger.info("History button clicked")
        print("History button clicked")

    def show_settings(self):
        """Example of handling settings-related functionality."""
        self.logger.info("Settings page clicked")
        print("Settings page clicked")
