# ATLAS/ATLAS.py

from ATLAS.config import ConfigManager
from ATLAS.persona_manager import PersonaManager
from ATLAS.provider_manager import ProviderManager

class ATLAS:
    def __init__(self, provider_manager: ProviderManager):
        """
        Initialize the ATLAS instance with a given ProviderManager.

        Args:
            provider_manager (ProviderManager): An instance of ProviderManager.
        """
        self.config_manager = provider_manager.config_manager
        self.logger = self.config_manager.logger
        self.persona_path = "/home/bib/Projects/LB/modules/Personas"
        self.user = "Bib"  # Example user, adjust as needed
        self.provider_manager = provider_manager
        self.persona_manager = PersonaManager(master=self, user=self.user)
        self.logger.info("ATLAS initialized successfully.")

    @classmethod
    async def create(cls):
        """
        Asynchronous factory method to create an ATLAS instance.

        Returns:
            ATLAS: An initialized instance of ATLAS.
        """
        config_manager = ConfigManager()
        provider_manager = await ProviderManager.create(config_manager)
        atlas_instance = cls(provider_manager)
        atlas_instance.logger.info("ATLAS created with ProviderManager.")
        return atlas_instance

    def get_persona_names(self):
        """
        Retrieve persona names from the PersonaManager.

        Returns:
            List[str]: A list of persona names.
        """
        return self.persona_manager.persona_names

    def load_persona(self, persona):
        """
        Delegate loading a persona to the PersonaManager.

        Args:
            persona (str): The name of the persona to load.
        """
        self.logger.info(f"Loading persona: {persona}")
        self.persona_manager.updater(persona)

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
