# ATLAS/model_manager.py

import json
import os
from typing import Dict, List, Tuple
from ATLAS.config import ConfigManager

class ModelManager:
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the ModelManager with a given ConfigManager.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        """
        self.config_manager = config_manager
        self.logger = self.config_manager.logger
        self.current_model = None
        self.current_provider = None
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, List[str]]:
        """
        Load available models for each provider from configuration files.

        Returns:
            Dict[str, List[str]]: A dictionary mapping providers to their available models.
        """
        models = {}
        providers = ['OpenAI', 'Mistral', 'Google', 'HuggingFace', 'Anthropic', 'Grok']
        base_path = os.path.dirname(os.path.abspath(__file__))  # This should point to the 'Providers' directory

        for provider in providers:
            if provider == 'HuggingFace':
                # Use the installed_models.json file in the model cache directory for HuggingFace
                file_path = os.path.join(self.config_manager.get_model_cache_dir(), 'installed_models.json')
            elif provider == 'Grok':
                # Grok doesn't need a models file, we just define the models here
                models[provider] = ["grok-2", "grok-2-mini"]
                continue
            else:
                file_name = f"{provider[0]}_models.json"
                file_path = os.path.join(base_path, provider, file_name)
            
            self.logger.debug(f"Attempting to load model file for {provider} from: {file_path}")

            try:
                with open(file_path, 'r') as f:
                    if provider == 'HuggingFace':
                        provider_models = json.load(f)
                    else:
                        provider_models = json.load(f).get('models', [])
                models[provider] = provider_models
                self.logger.info(f"Successfully loaded models for {provider}: {models[provider]}")
            except FileNotFoundError:
                self.logger.warning(f"Model file not found for provider {provider} at {file_path}")
                models[provider] = []
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding JSON in model file for provider {provider} at {file_path}")
                models[provider] = []
            except Exception as e:
                self.logger.error(f"Unexpected error loading model file for provider {provider}: {str(e)}")
                models[provider] = []

        return models

    def set_model(self, model_name: str, provider: str) -> None:
        """
        Set the current model for the specified provider.

        Args:
            model_name (str): The name of the model to set.
            provider (str): The provider of the model.
        """
        if provider not in self.models:
            self.models[provider] = []
        if model_name not in self.models[provider]:
            self.models[provider].append(model_name)
        self.current_model = model_name
        self.current_provider = provider
        self.logger.info(f"Model set to {model_name} for provider {provider}")

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str: The name of the current model.
        """
        return self.current_model

    def get_current_provider(self) -> str:
        """
        Get the current provider being used.

        Returns:
            str: The name of the current provider.
        """
        return self.current_provider

    def get_available_models(self, provider: str = None) -> Dict[str, List[str]]:
        """
        Get available models for a specific provider or all providers.

        Args:
            provider (str, optional): The name of the provider. Defaults to None.

        Returns:
            Dict[str, List[str]]: A dictionary of available models.
        """
        if provider:
            return {provider: self.models.get(provider, [])}
        return self.models

    def get_token_limits_for_model(self, model_name: str) -> Tuple[int, int]:
        """
        Get the token limits for a specific model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Tuple[int, int]: A tuple containing (input_tokens, output_tokens).
        """
        # Define token limits for different models
        token_limits = {
            # OpenAI models
            "gpt-4o": (4000, 128000),
            "gpt-4o-mini": (12000, 128000),
            # Google models
            "gemini-1.5-pro-latest": (8192, 32768),
            # Anthropic models
            "claude-3-opus-20240229": (4096, 200000),
            "claude-3-sonnet-20240229": (4096, 200000),
            "claude-2.1": (4096, 200000),
            "claude-2.0": (4096, 200000),
            "claude-3-haiku-20240229": (4096, 100000),
            "claude-instant-1.2": (4096, 100000),
            # Mistral models
            "mistral-small-latest": (4096, 8192),
            "mistral-medium-latest": (4096, 8192),
            "mistral-large-latest": (4096, 8192),
            # HuggingFace models
        }

        # Return token limits for the specified model, or default values if not found
        return token_limits.get(model_name, (2000, 4000))

    def get_default_model_for_provider(self, provider: str) -> str:
        """
        Get the default model for a specific provider.

        Args:
            provider (str): The name of the provider.

        Returns:
            str: The default model name, or None if not available.
        """
        if provider not in self.models:
            return None
        return self.models[provider][0] if self.models[provider] else None
