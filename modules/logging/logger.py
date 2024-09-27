# src/components/logging/logger.py

import logging
import os
from concurrent_log_handler import ConcurrentRotatingFileHandler
import yaml

class CustomLogger(logging.Logger):
    def __init__(self, name, config=None):
        super().__init__(name)
        self.config = config or self._load_default_config()
        self._configure_logger()

    def _load_default_config(self):
        default_config = {
            'log_level': 'INFO',
            'file_log_level': 'DEBUG',
            'console_log_level': 'INFO',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'CSSLM.log',
            'max_file_size': 50 * 1024 * 1024,  # 50 MB
            'backup_count': 5
        }
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'logging_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return default_config

    def _configure_logger(self):
        self.setLevel(self._get_log_level(self.config['log_level']))
        formatter = logging.Formatter(self.config['log_format'])

        # File Handler
        log_directory = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'logs')
        os.makedirs(log_directory, exist_ok=True)
        file_handler = ConcurrentRotatingFileHandler(
            os.path.join(log_directory, self.config['log_file']),
            maxBytes=self.config['max_file_size'],
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setLevel(self._get_log_level(self.config['file_log_level']))
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level(self.config['console_log_level']))
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        self.propagate = False

    @staticmethod
    def _get_log_level(level_str):
        return getattr(logging, level_str.upper(), logging.INFO)

def setup_logger(name):
    logging.setLoggerClass(CustomLogger)
    return logging.getLogger(name)

# Set levels for specific loggers
logging.getLogger('components.state_measurement').setLevel(logging.WARNING)
logging.getLogger('components.attention_focus_mechanism').setLevel(logging.WARNING)