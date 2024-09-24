# ATLAS/ATLAS.py

from typing import List
from ATLAS.config import ConfigManager
from ATLAS.persona_manager import PersonaManager
from ATLAS.provider_manager import ProviderManager
from ATLAS.chat_session import ChatSession

class ATLAS:
    """
    The main ATLAS application class that manages configurations, providers, and personas.
    """

    def __init__(self):
        """
        Initialize the ATLAS instance with synchronous initialization.
        """
        self.config_manager = ConfigManager()
        self.logger = self.config_manager.logger
        self.persona_path = "/home/bib/Projects/LB/modules/Personas"  # Adjust as needed
        self.user = "Bib"  # Example user, adjust as needed
        self.provider_manager = ProviderManager(self.config_manager)
        self.persona_manager = PersonaManager(master=self, user=self.user)
        self.chat_session = ChatSession(self)  # Initialize ChatSession
        self.logger.info("ATLAS initialized successfully.")

    def get_persona_names(self) -> List[str]:
        """
        Retrieve persona names from the PersonaManager.

        Returns:
            List[str]: A list of persona names.
        """
        return self.persona_manager.persona_names

    def load_persona(self, persona: str):
        """
        Delegate loading a persona to the PersonaManager.

        Args:
            persona (str): The name of the persona to load.
        """
        self.logger.info(f"Loading persona: {persona}")
        self.persona_manager.updater(persona)

    def get_available_providers(self) -> List[str]:
        """
        Retrieve all available providers from the ProviderManager.

        Returns:
            List[str]: A list of provider names.
        """
        return self.provider_manager.get_available_providers()
    
    async def set_current_provider(self, provider: str):
        """
        Asynchronously set the current provider in the ProviderManager.
        """
        await self.provider_manager.set_current_provider(provider)
        current_model = self.provider_manager.get_current_model()
        self.logger.info(f"Current provider set to {provider} with model {current_model}")
            
    def log_history(self):
        """
        Handle history-related functionality.
        """
        self.logger.info("History button clicked")
        print("History button clicked")

    def show_settings(self):
        """
        Handle settings-related functionality.
        """
        self.logger.info("Settings page clicked")
        print("Settings page clicked")
