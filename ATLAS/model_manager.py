# ATLAS/model_manager.py

import json
import os
import threading
from pathlib import Path
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

            atlas_root = self.config_manager.get_app_root()
            providers_path = Path(atlas_root) / "modules" / "Providers"

            self.logger.debug(f"Atlas root directory: {atlas_root}")
            self.logger.debug(f"Providers path: {providers_path}")

            for provider in providers:
                file_candidates: List[Path] = []
                normalize = True

                if provider == 'HuggingFace':
                    cache_dir = Path(self.config_manager.get_model_cache_dir())
                    file_candidates.append(cache_dir / 'installed_models.json')
                    normalize = False
                elif provider == 'Grok':
                    self.models[provider] = ["grok-2", "grok-2-mini"]
                    continue
                elif provider == 'Google':
                    # Google models are discovered dynamically via the API helper.
                    self.logger.debug(
                        "Skipping static model list for Google; awaiting API-backed discovery."
                    )
                    self.models[provider] = self._normalize_models(self.models.get(provider, []))
                    continue
                elif provider == 'Anthropic':
                    try:
                        app_root_path = Path(self.config_manager.get_app_root())
                    except Exception:
                        app_root_path = providers_path.parent

                    file_candidates.extend(
                        [
                            app_root_path / "modules" / "Providers" / "Anthropic" / "anthropic_models.json",
                            app_root_path / "modules" / "Providers" / "Anthropic" / "A_models.json",
                            Path(__file__).resolve().parent.parent
                            / "modules"
                            / "Providers"
                            / "Anthropic"
                            / "anthropic_models.json",
                        ]
                    )
                else:
                    file_name = f"{provider[0]}_models.json"  # e.g., 'O_models.json' for OpenAI
                    file_candidates.append(providers_path / provider / file_name)

                unique_candidates: List[Path] = []
                for candidate in file_candidates:
                    if candidate not in unique_candidates:
                        unique_candidates.append(candidate)

                loaded = False
                for candidate in unique_candidates:
                    self.logger.debug(
                        f"Attempting to load model file for {provider} from: {candidate}"
                    )
                    try:
                        with candidate.open('r', encoding='utf-8') as handle:
                            payload = json.load(handle)
                    except FileNotFoundError:
                        continue
                    except json.JSONDecodeError:
                        self.logger.error(
                            f"Error decoding JSON in model file for provider {provider} at {candidate}"
                        )
                        self.models[provider] = []
                        loaded = True
                        break
                    except Exception as exc:
                        self.logger.error(
                            f"Unexpected error loading model file for provider {provider} at {candidate}: {str(exc)}"
                        )
                        self.models[provider] = []
                        loaded = True
                        break

                    if provider == 'HuggingFace':
                        provider_models = payload if isinstance(payload, list) else []
                    else:
                        provider_models = payload.get('models', []) if isinstance(payload, dict) else []

                    self.models[provider] = (
                        provider_models if not normalize else self._normalize_models(provider_models)
                    )
                    self.logger.info(
                        f"Successfully loaded models for {provider}: {self.models[provider]}"
                    )
                    loaded = True
                    break

                if not loaded:
                    if provider == 'Anthropic':
                        self.logger.info(
                            "No cached Anthropic model list found; run discovery to populate it."
                        )
                    else:
                        missing_path = unique_candidates[0] if unique_candidates else None
                        self.logger.warning(
                            f"Model file not found for provider {provider} at {missing_path}"
                        )
                    self.models[provider] = []

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
