# modules/config.py ### NEW ###

import logging
import os
import yaml
import aiofiles
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self):
        load_dotenv()
        self.config = self._load_env_config()
        self.logging_config = self._load_logging_config()
        self.logger = self.setup_logger('config_manager')
        #self.logger.info("Initializing ConfigManager")
        
        #self.logger.info("ConfigManager initialization completed")

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
        
        if not config['OPENAI_API_KEY']:
            self.logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found in environment variables")
        
        return config

    def _load_logging_config(self):
        config_path = os.path.join(self.get_app_root(), 'config', 'logging_config.yaml')
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Logging config file not found at {config_path}")
            return {"log_level": "INFO", "console_enabled": True}  # Default configuration
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing logging config file: {e}")
            return {"log_level": "INFO", "console_enabled": True}  # Default configuration

    def get_config(self, key: str, default: Any = None) -> Any:
        value = self.config.get(key, default)
        # self.logger.debug(f"Retrieved config value for {key}: {value}")
        return value

    def get_log_level(self):
        log_level = self.logging_config.get('log_level', 'INFO').upper()
        return getattr(logging, log_level, logging.INFO)

    def get_model_cache_dir(self):
        return self.get_config('MODEL_CACHE_DIR')

    def get_default_provider(self):
        return self.get_config('DEFAULT_PROVIDER')

    def get_default_model(self):
        return self.get_config('DEFAULT_MODEL')

    def setup_logger(self, name):
        logger = logging.getLogger(name)
        log_level = self.get_log_level()
        logger.setLevel(log_level)

        if not logger.hasHandlers():
            formatter = logging.Formatter(self.logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Console handler
            if self.logging_config.get('console_enabled', True):
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self.logging_config.get('console_log_level', log_level))
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            # File handler
            if self.logging_config.get('file_log_level'):
                file_handler = logging.FileHandler(self.logging_config.get('log_file', 'CSSLM.log'))
                file_handler.setLevel(self.logging_config.get('file_log_level', log_level))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        return logger

    def set_log_level(self, level):
        self.config['LOG_LEVEL'] = level
        log_level = getattr(logging, level.upper())
        # Set for root logger
        logging.getLogger().setLevel(log_level)
        # Set for all existing loggers
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(log_level)
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

    def ensure_directories_exist(self):
        directories = [
            self.get_app_root(),
            self.get_model_cache_dir(),
            os.path.dirname(self.get_config('LOG_FILE_PATH'))
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_config(self, key: str, value: Any) -> None:
        # self.logger.debug(f"Updating config: {key} = {value}")
        self.config[key] = value
        config_path = os.path.join(self.get_app_root(), 'config', 'config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)
        # self.logger.debug(f"Updated config saved to {config_path}")

    def update_logging_config(self, console_enabled: bool):
        """
        Update the logging configuration to enable or disable console logging.

        Args:
            console_enabled (bool): Whether to enable console logging.
        """
        self.logging_config['console_enabled'] = console_enabled
        
        # Update all existing loggers
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            self._update_logger_handlers(logger)

        self.logger.info(f"Console logging {'enabled' if console_enabled else 'disabled'}")

    def _update_logger_handlers(self, logger):
        """
        Update the handlers of a given logger based on the current configuration.
        """
        # Remove existing console handlers
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]

        # Add console handler if enabled
        if self.logging_config.get('console_enabled', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.logging_config.get('console_log_level', logger.level))
            formatter = logging.Formatter(self.logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)    

    def __str__(self) -> str:
        return f"ConfigManager(APP_ROOT={self.get_app_root()})"

    def __repr__(self) -> str:
        return self.__str__()