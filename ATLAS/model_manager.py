# ATLAS/model_manager.py

import threading
from typing import Dict, List, Tuple
from ATLAS.config import ConfigManager
from modules.logging.logger import setup_logger

class ModelManager:
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the ModelManager with a given ConfigManager.

        Args:
            config_manager (ConfigManager): An instance of ConfigManager.
        """
        self.config_manager = config_manager
        self.logger = setup_logger(__name__)
        self.current_model = None
        self.current_provider = None
        self.models = {}
        self.lock = threading.Lock()  # Add a lock for thread safety
        self.load_models()

    def _normalize_models(self, models: List[str]) -> List[str]:
        """Return a de-duplicated, stripped list of model names."""

        seen = set()
        normalized: List[str] = []
        for entry in models:
            if not isinstance(entry, str):
                continue

            name = entry.strip()
            if name and name not in seen:
                normalized.append(name)
                seen.add(name)

        return normalized

    def load_models(self) -> None:
        """
        Load available models for each provider from configuration files.
        """
        with self.lock:
            providers = ['OpenAI', 'Mistral', 'Google', 'HuggingFace', 'Anthropic', 'Grok']

            cached_models: Dict[str, List[str]] = {}
            getter = getattr(self.config_manager, "get_cached_models", None)
            if callable(getter):
                try:
                    cached_models = getter()
                except Exception as exc:
                    self.logger.warning(
                        "Failed to load cached provider models from configuration: %s",
                        exc,
                        exc_info=True,
                    )
            else:
                self.logger.debug(
                    "Config manager does not expose cached model accessor; starting with empty cache."
                )

            self.models = {}

            for provider, models in cached_models.items():
                if isinstance(models, list):
                    raw_models = list(models)
                elif isinstance(models, (tuple, set)):
                    raw_models = [str(entry) for entry in models]
                else:
                    raw_models = []

                if provider == 'HuggingFace':
                    normalized = [
                        entry.strip()
                        for entry in raw_models
                        if isinstance(entry, str) and entry.strip()
                    ]
                else:
                    normalized = self._normalize_models(raw_models)

                self.models[provider] = normalized

            for provider in providers:
                self.models.setdefault(provider, [])

            self.logger.debug(
                "Initialized provider model cache for: %s",
                ", ".join(sorted(self.models.keys())),
            )

    def set_model(self, model_name: str, provider: str) -> None:
        """
        Set the current model for the specified provider.

        Args:
            model_name (str): The name of the model to set.
            provider (str): The provider of the model.
        """
        with self.lock:
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
        with self.lock:
            return self.current_model

    def get_current_provider(self) -> str:
        """
        Get the current provider being used.

        Returns:
            str: The name of the current provider.
        """
        with self.lock:
            return self.current_provider

    def get_available_models(self, provider: str = None) -> Dict[str, List[str]]:
        """
        Get available models for a specific provider or all providers.

        Args:
            provider (str, optional): The name of the provider. Defaults to None.

        Returns:
            Dict[str, List[str]]: A dictionary of available models.
        """
        with self.lock:
            if provider:
                return {provider: self.models.get(provider, [])}
            return self.models.copy()

    def update_models_for_provider(self, provider: str, models: List[str]) -> List[str]:
        """Replace the cached models for a provider while preserving known fallbacks."""

        normalized = self._normalize_models(models)

        with self.lock:
            existing = self.models.get(provider, []) or []
            default_entry = None
            if existing:
                head = existing[0]
                if isinstance(head, str) and head.strip():
                    default_entry = head.strip()

            if default_entry and default_entry not in normalized:
                normalized.insert(0, default_entry)
            elif default_entry:
                normalized = [default_entry] + [name for name in normalized if name != default_entry]

            seen = set(normalized)
            for name in existing:
                if isinstance(name, str):
                    trimmed = name.strip()
                    if trimmed and trimmed not in seen:
                        normalized.append(trimmed)
                        seen.add(trimmed)

            persisted: List[str] | None = None
            setter = getattr(self.config_manager, "set_cached_models", None)
            if callable(setter):
                try:
                    persisted = setter(provider, normalized)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to persist cached models for %s: %s", provider, exc
                    )

            if persisted is not None:
                normalized = list(persisted)

            self.models[provider] = normalized
            self.logger.info(
                "Updated cached models for %s. %d entries available.", provider, len(normalized)
            )

            return list(normalized)

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
            "gpt-4o": (128000, 16384),
            "gpt-4o-mini": (128000, 16384),
            "gpt-4o-mini-tts": (4096, 4096),
            "gpt-4o-transcribe": (4096, 4096),
            "gpt-4o-mini-transcribe": (4096, 4096),
            "gpt-4.1": (128000, 16384),
            "gpt-4.1-mini": (128000, 16384),
            "o1": (200000, 65536),
            "o1-mini": (200000, 32768),
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
            # Add token limits for HuggingFace models if needed
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
        with self.lock:
            if provider not in self.models or not self.models[provider]:
                self.logger.error(f"No models found for provider {provider}. Ensure the provider's models are loaded.")
                return None
            return self.models[provider][0]
