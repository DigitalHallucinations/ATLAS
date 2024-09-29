# ATLAS/ATLAS.py

from typing import List
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger
from ATLAS.provider_manager import ProviderManager
from ATLAS.persona_manager import PersonaManager
from modules.Chat.chat_session import ChatSession

class ATLAS:
    """
    The main ATLAS application class that manages configurations, providers, and personas.
    """

    def __init__(self):
        """
        Initialize the ATLAS instance with synchronous initialization.
        """
        self.config_manager = ConfigManager()
        self.logger = setup_logger(__name__)
        self.persona_path = self.config_manager.get_app_root()  
        self.current_persona = None
        self.user = "Bib"  # Placeholder; replace with system user retrieval
        self.provider_manager = None
        self.persona_manager = None
        self.chat_session = None
        self._initialized = False

    async def initialize(self):
        """
        Asynchronously initialize the ATLAS instance.
        """
        self.provider_manager = await ProviderManager.create(self.config_manager)
        self.persona_manager = PersonaManager(master=self, user=self.user)
        self.chat_session = ChatSession(self)
        
        default_provider = self.config_manager.get_default_provider()
        await self.provider_manager.set_current_provider(default_provider)
        
        self.logger.info(f"Default provider set to: {self.provider_manager.get_current_provider()}")
        self.logger.info(f"Default model set to: {self.provider_manager.get_current_model()}")
        self.logger.info("ATLAS initialized successfully.")
        self._initialized = True

    def is_initialized(self) -> bool:
        """
        Check if ATLAS is fully initialized.

        Returns:
            bool: True if ATLAS is initialized, False otherwise.
        """
        return self._initialized

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
        self.chat_session.set_provider(provider)
        current_model = self.provider_manager.get_current_model()
        self.chat_session.set_model(current_model)
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

    def get_default_provider(self) -> str:
        """
        Get the default provider from the ProviderManager.

        Returns:
            str: The name of the default provider.
        """
        return self.provider_manager.get_current_provider()

    def get_default_model(self) -> str:
        """
        Get the default model from the ProviderManager.

        Returns:
            str: The name of the default model.
        """
        return self.provider_manager.get_current_model()
    
    async def close(self):
        """
        Perform cleanup operations.
        """
        await self.provider_manager.close()
        self.logger.info("ATLAS closed and all providers unloaded.")
