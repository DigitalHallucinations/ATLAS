# modules/config.py

import logging
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self):
        load_dotenv()
        self.logger = self.setup_logger('config_manager')
        self.config = self._load_env_config()
        self.logging_config = self._load_logging_config()
        self.bitnet_config = self._load_bitnet_config()

        # Update the logger level after loading the logging config
        log_level = self.get_log_level()
        self.logger.setLevel(log_level)

        # Derive other paths from APP_ROOT
        self.config['LOG_FILE_PATH'] = os.path.join(self.config['APP_ROOT'], 'logs', 'application.log')
        self.config['MODEL_CACHE_DIR'] = os.path.join(self.config['APP_ROOT'], 'model_cache')

        if not self.config['OPENAI_API_KEY']:
            raise ValueError("OpenAI API key not found in environment variables")

    def _load_env_config(self) -> Dict[str, Any]:
        config = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4'),
            'MONGO_CONNECTION_STRING': os.getenv('MONGO_CONNECTION_STRING'),
            'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
            'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'APP_ROOT': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        }
        self.logger.info(f"APP_ROOT is set to: {config['APP_ROOT']}")
        return config

    def _load_logging_config(self):
        config_path = os.path.join(self.get_app_root(), 'config', 'logging_config.yaml')
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Logging config file not found at {config_path}, using default logging config.")
            return {"log_level": "INFO", "console_enabled": True}  # Default configuration
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing logging config file: {e}")
            return {"log_level": "INFO", "console_enabled": True}  # Default configuration

    def _load_bitnet_config(self):
        config_path = os.path.join(self.get_app_root(), 'config', 'bitnet_config.yaml')
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"BitNet config file not found at {config_path}, using default BitNet config.")
            return self._get_default_bitnet_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing BitNet config file: {e}")
            return self._get_default_bitnet_config()

    def _get_default_bitnet_config(self):
        return {
            "enabled": False,
            "lambda": 0.0,
            "lambda_scheduler": "linear",
            "warmup_steps": 1000,
            "quantization_method": "ternary",
            "group_size": 1,
            "outlier_threshold": 1.0,
            "custom_kernel": False,
            "kernel_type": "cuda"
        }

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def get_log_level(self):
        return getattr(logging, self.logging_config.get('log_level', 'INFO').upper(), logging.INFO)

    def get_model_cache_dir(self):
        return self.get_config('MODEL_CACHE_DIR')

    def get_default_provider(self):
        return self.get_config('DEFAULT_PROVIDER')

    def get_default_model(self):
        return self.get_config('DEFAULT_MODEL')

    def setup_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def set_log_level(self, level):
        self.config['LOG_LEVEL'] = level
        log_level = getattr(logging, level.upper())
        logging.getLogger().setLevel(log_level)
        self.logger.info(f"Log level set to {level}")

    def get_openai_api_key(self):
        return self.get_config('OPENAI_API_KEY')

    def get_mistral_api_key(self):
        return self.get_config('MISTRAL_API_KEY')

    def get_huggingface_api_key(self):
        return self.get_config('HUGGINGFACE_API_KEY')

    def get_google_api_key(self):
        return self.get_config('GOOGLE_API_KEY')

    def get_anthropic_api_key(self):
        return self.get_config('ANTHROPIC_API_KEY')

    def get_app_root(self):
        return self.get_config('APP_ROOT')

    # New methods for BitNet configuration
    def get_bitnet_config(self):
        return self.bitnet_config

    def update_bitnet_config(self, new_config):
        self.bitnet_config.update(new_config)
        self._save_bitnet_config()

    def _save_bitnet_config(self):
        config_path = os.path.join(self.get_app_root(), 'config', 'bitnet_config.yaml')
        try:
            with open(config_path, 'w') as file:
                yaml.dump(self.bitnet_config, file)
            self.logger.info("BitNet configuration saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving BitNet configuration: {e}")

    def get_bitnet_enabled(self):
        return self.bitnet_config.get('enabled', False)

    def get_bitnet_lambda(self):
        return self.bitnet_config.get('lambda', 0.0)

# Sample usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    print(config_manager.get_bitnet_config())
    config_manager.update_bitnet_config({"enabled": True, "lambda": 0.5})
    print(config_manager.get_bitnet_config())
