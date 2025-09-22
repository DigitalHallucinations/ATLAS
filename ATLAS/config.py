# ATLAS/config.py

import os
from typing import Dict, Any, Optional
from modules.logging.logger import setup_logger
from dotenv import load_dotenv, set_key, find_dotenv
import yaml

class ConfigManager:
    """
    Manages configuration settings for the application, including loading
    environment variables and handling API keys for various providers.
    """

    def __init__(self):
        """
        Initializes the ConfigManager by loading environment variables and loading configuration settings.

        Raises:
            ValueError: If the API key for the default provider is not found in environment variables.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Setup logger early to log any issues
        self.logger = setup_logger(__name__)
        
        # Load configurations from .env and config.yaml
        self.env_config = self._load_env_config()
        self._yaml_path = self._compute_yaml_path()
        self.yaml_config = self._load_yaml_config()
        
        # Merge configurations, with YAML config overriding env config if there's overlap
        self.config = {**self.env_config, **self.yaml_config}

        # Derive other paths from APP_ROOT
        self.config['MODEL_CACHE_DIR'] = os.path.join(
            self.config.get('APP_ROOT', '.'),
            'modules',
            'Providers',
            'HuggingFace',
            'model_cache'
        )
        # Ensure the model_cache directory exists
        os.makedirs(self.config['MODEL_CACHE_DIR'], exist_ok=True)

        # Validate the API key for the default provider
        default_provider = self.config.get('DEFAULT_PROVIDER', 'OpenAI')
        if not self._is_api_key_set(default_provider):
            raise ValueError(f"{default_provider} API key not found in environment variables")

    def _get_provider_env_keys(self) -> Dict[str, str]:
        """Return the mapping between provider display names and environment keys."""

        return {
            "OpenAI": "OPENAI_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Google": "GOOGLE_API_KEY",
            "HuggingFace": "HUGGINGFACE_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Grok": "GROK_API_KEY",
            "ElevenLabs": "XI_API_KEY",
        }

    def _compute_yaml_path(self) -> str:
        """Return the absolute path to the persistent YAML configuration file."""

        return os.path.join(
            self.env_config.get('APP_ROOT', '.'),
            'ATLAS',
            'config',
            'atlas_config.yaml',
        )

    def _load_env_config(self) -> Dict[str, Any]:
        """
        Loads environment variables into the configuration dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all loaded environment configuration settings.
        """
        config = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4o'),
            'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
            'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GROK_API_KEY': os.getenv('GROK_API_KEY'),
            'XI_API_KEY': os.getenv('XI_API_KEY'),
            'OPENAI_BASE_URL': os.getenv('OPENAI_BASE_URL'),
            'OPENAI_ORGANIZATION': os.getenv('OPENAI_ORGANIZATION'),
            'APP_ROOT': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        }
        self.logger.info(f"APP_ROOT is set to: {config['APP_ROOT']}")
        return config

    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Loads configuration settings from the config.yaml file.

        Returns:
            Dict[str, Any]: A dictionary containing all loaded YAML configuration settings.
        """
        yaml_path = getattr(self, '_yaml_path', None) or self._compute_yaml_path()

        if not os.path.exists(yaml_path):
            self.logger.error(f"Configuration file not found: {yaml_path}")
            return {}
        
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file) or {}
                self.logger.info(f"Loaded configuration from {yaml_path}")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            return {}

    def _persist_env_value(self, env_key: str, value: Optional[str]):
        """Persist an environment-backed configuration value."""

        env_path = find_dotenv()
        if not env_path:
            raise FileNotFoundError("`.env` file not found.")

        # Persist the value to the .env file and refresh the loaded environment.
        set_key(env_path, env_key, value or "")
        load_dotenv(env_path, override=True)

        # Synchronize in-memory state and environment variables.
        if value is None or value == "":
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = value

        self.env_config[env_key] = value
        if value is None:
            self.config.pop(env_key, None)
        else:
            self.config[env_key] = value

    def set_google_credentials(self, credentials_path: str):
        """Persist Google application credentials and refresh process state."""

        if not credentials_path:
            raise ValueError("Google credentials path cannot be empty.")

        self._persist_env_value("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
        self.logger.info("Google credentials path updated.")

    def set_hf_token(self, token: str):
        """Persist the Hugging Face access token."""

        if not token:
            raise ValueError("Hugging Face token cannot be empty.")

        self._persist_env_value("HUGGINGFACE_API_KEY", token)
        self.logger.info("Hugging Face token updated.")

    def set_elevenlabs_api_key(self, api_key: str):
        """Persist the ElevenLabs access token."""

        if not api_key:
            raise ValueError("ElevenLabs API key cannot be empty.")

        self._persist_env_value("XI_API_KEY", api_key)
        self.logger.info("ElevenLabs API key updated.")

    def set_openai_speech_config(
        self,
        *,
        api_key: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_provider: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ):
        """Persist OpenAI speech configuration values."""

        if api_key is not None:
            if not api_key:
                raise ValueError("OpenAI API key cannot be empty.")
            self._persist_env_value("OPENAI_API_KEY", api_key)
            self.logger.info("OpenAI API key updated for speech services.")

        config_updates = {
            "OPENAI_STT_PROVIDER": stt_provider,
            "OPENAI_TTS_PROVIDER": tts_provider,
            "OPENAI_LANGUAGE": language,
            "OPENAI_TASK": task,
            "OPENAI_INITIAL_PROMPT": initial_prompt,
        }

        for key, value in config_updates.items():
            if value is not None:
                self.config[key] = value
            elif key in self.config:
                self.config[key] = None

    def set_openai_llm_settings(
        self,
        *,
        model: Optional[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist OpenAI chat-completion defaults and related metadata."""

        if not model:
            raise ValueError("A default OpenAI model must be provided.")

        normalized_temperature = 0.0 if temperature is None else float(temperature)
        if not 0.0 <= normalized_temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        normalized_max_tokens = 4000 if max_tokens is None else int(max_tokens)
        if normalized_max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer.")

        normalized_stream = True if stream is None else bool(stream)

        sanitized_api_key = None
        if api_key is not None:
            sanitized_api_key = api_key.strip()
            if not sanitized_api_key:
                raise ValueError("OpenAI API key cannot be empty.")

        sanitized_org = (organization or "").strip() or None

        settings_block = {}
        existing = self.yaml_config.get('OPENAI_LLM')
        if isinstance(existing, dict):
            settings_block.update(existing)
            settings_block.pop('base_url', None)

        settings_block.update(
            {
                'model': model,
                'temperature': normalized_temperature,
                'max_tokens': normalized_max_tokens,
                'stream': normalized_stream,
                'organization': sanitized_org,
            }
        )

        self.yaml_config['OPENAI_LLM'] = settings_block
        self.config['OPENAI_LLM'] = dict(settings_block)

        # Persist environment-backed values.
        self._persist_env_value('DEFAULT_MODEL', model)
        self.config['DEFAULT_MODEL'] = model

        if sanitized_api_key is not None:
            self._persist_env_value('OPENAI_API_KEY', sanitized_api_key)
            self.config['OPENAI_API_KEY'] = sanitized_api_key

        self._persist_env_value('OPENAI_ORGANIZATION', sanitized_org)

        # Synchronize cached environment map for convenience.
        self.env_config['DEFAULT_MODEL'] = model
        if sanitized_api_key is not None:
            self.env_config['OPENAI_API_KEY'] = sanitized_api_key
        self.env_config['OPENAI_ORGANIZATION'] = sanitized_org

        self._write_yaml_config()

        return dict(settings_block)


    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by its key.

        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): The default value to return if the key is not found.

        Returns:
            Any: The value associated with the key, or the default value if key is absent.
        """
        return self.config.get(key, default)

    def get_model_cache_dir(self) -> str:
        """
        Retrieves the directory path where models are cached.

        Returns:
            str: The path to the model cache directory.
        """
        return self.get_config('MODEL_CACHE_DIR')

    def get_default_provider(self) -> str:
        """
        Retrieves the default provider name from the configuration.

        Returns:
            str: The name of the default provider.
        """
        return self.get_config('DEFAULT_PROVIDER')

    def get_default_model(self) -> str:
        """
        Retrieves the default model name from the configuration.

        Returns:
            str: The name of the default model.
        """
        return self.get_config('DEFAULT_MODEL')

    def get_openai_llm_settings(self) -> Dict[str, Any]:
        """Return persisted OpenAI LLM defaults merged with environment values."""

        defaults = {
            'model': self.get_config('DEFAULT_MODEL', 'gpt-4o'),
            'temperature': 0.0,
            'max_tokens': 4000,
            'stream': True,
            'organization': self.get_config('OPENAI_ORGANIZATION'),
        }

        stored = self.get_config('OPENAI_LLM')
        if isinstance(stored, dict):
            defaults.update({k: stored.get(k, defaults.get(k)) for k in defaults.keys()})

        return defaults


    def get_openai_api_key(self) -> str:
        """
        Retrieves the OpenAI API key from the configuration.

        Returns:
            str: The OpenAI API key.
        """
        return self.get_config('OPENAI_API_KEY')

    def get_mistral_api_key(self) -> str:
        """
        Retrieves the Mistral API key from the configuration.

        Returns:
            str: The Mistral API key.
        """
        return self.get_config('MISTRAL_API_KEY')

    def get_huggingface_api_key(self) -> str:
        """
        Retrieves the HuggingFace API key from the configuration.

        Returns:
            str: The HuggingFace API key.
        """
        return self.get_config('HUGGINGFACE_API_KEY')

    def get_google_api_key(self) -> str:
        """
        Retrieves the Google API key from the configuration.

        Returns:
            str: The Google API key.
        """
        return self.get_config('GOOGLE_API_KEY')

    def get_anthropic_api_key(self) -> str:
        """
        Retrieves the Anthropic API key from the configuration.

        Returns:
            str: The Anthropic API key.
        """
        return self.get_config('ANTHROPIC_API_KEY')

    def get_grok_api_key(self) -> str:
        """
        Retrieves the Grok API key from the configuration.

        Returns:
            str: The Grok API key.
        """
        return self.get_config('GROK_API_KEY')

    def get_app_root(self) -> str:
        """
        Retrieves the application's root directory path.

        Returns:
            str: The path to the application's root directory.
        """
        return self.get_config('APP_ROOT')

    def update_api_key(self, provider_name: str, new_api_key: str):
        """
        Updates the API key for a specified provider in the .env file and reloads
        the environment variables to reflect the changes immediately.

        Args:
            provider_name (str): The name of the provider whose API key is to be updated.
            new_api_key (str): The new API key to set for the provider.

        Raises:
            FileNotFoundError: If the .env file is not found.
            ValueError: If the provider name does not have a corresponding API key mapping.
        """
        provider_env_keys = self._get_provider_env_keys()

        env_key = provider_env_keys.get(provider_name)
        if not env_key:
            raise ValueError(f"No API key mapping found for provider '{provider_name}'.")

        self._persist_env_value(env_key, new_api_key)
        self.logger.info(f"API key for {provider_name} updated successfully.")

    def has_provider_api_key(self, provider_name: str) -> bool:
        """
        Determine whether an API key is configured for the given provider.

        Args:
            provider_name (str): The name of the provider to check.

        Returns:
            bool: True if an API key exists for the provider, False otherwise.
        """

        return self._is_api_key_set(provider_name)

    def _is_api_key_set(self, provider_name: str) -> bool:
        """
        Checks if the API key for a specified provider is set.

        Args:
            provider_name (str): The name of the provider.

        Returns:
            bool: True if the API key is set, False otherwise.
        """
        env_key = self._get_provider_env_keys().get(provider_name)
        if not env_key:
            return False

        api_key = self.get_config(env_key)
        return bool(api_key)

    def get_available_providers(self) -> Dict[str, str]:
        """
        Retrieves a dictionary of available providers and their corresponding API keys.

        Returns:
            Dict[str, str]: A dictionary where keys are provider names and values are their API keys.
        """
        provider_env_keys = self._get_provider_env_keys()
        return {provider: self.get_config(env_key) for provider, env_key in provider_env_keys.items()}

    # Additional methods to handle TTS_ENABLED from config.yaml
    def get_tts_enabled(self) -> bool:
        """
        Retrieves the TTS enabled status from the configuration.

        Returns:
            bool: True if TTS is enabled, False otherwise.
        """
        return self.get_config('TTS_ENABLED', False)

    def set_tts_enabled(self, value: bool):
        """
        Sets the TTS enabled status in the configuration.

        Args:
            value (bool): True to enable TTS, False to disable.
        """
        self.yaml_config['TTS_ENABLED'] = value
        self.config['TTS_ENABLED'] = value
        self.logger.info(f"TTS_ENABLED set to {value}")
        # Optionally, write back to config.yaml if persistence is required
        self._write_yaml_config()

    def _write_yaml_config(self):
        """
        Writes the current YAML configuration back to the config.yaml file.
        """
        yaml_path = getattr(self, '_yaml_path', None) or self._compute_yaml_path()
        try:
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as file:
                yaml.dump(self.yaml_config, file)
            self.logger.info(f"Configuration written to {yaml_path}")
        except Exception as e:
            self.logger.error(f"Failed to write configuration to {yaml_path}: {e}")
