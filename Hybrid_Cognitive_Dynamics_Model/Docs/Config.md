# Configuration Manager Documentation

## File: `config.py`

### Class: `ConfigManager`

The `ConfigManager` class is responsible for managing the application's configuration, including environment variables, logging setup, and directory management.

#### Initialization

```python
def __init__(self):
    load_dotenv()
    self.config = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'DEFAULT_PROVIDER': os.getenv('DEFAULT_PROVIDER', 'OpenAI'),
        'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'gpt-4'),
        'MONGO_CONNECTION_STRING': os.getenv('MONGO_CONNECTION_STRING'),
        'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
        'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'REAL_AGENT_ROOT': os.getenv('REAL_AGENT_ROOT', 'c:\\Real Agent'),
    }
    # Additional initialization code for derived paths and logger setup
```

The constructor loads environment variables, sets up default configurations, and initializes logging.

#### Key Methods

1. `get_config(self, key, default=None)`:
   - Retrieves a configuration value by key.

2. `get_model_cache_dir(self)`:
   - Returns the directory path for caching AI models.

3. `get_log_level(self)`:
   - Returns the configured log level.

4. `setup_logger(self, name)`:
   - Sets up and returns a logger with the specified name.

5. `set_log_level(self, level)`:
   - Sets a new log level for the application.

6. `get_mongo_connection_string(self)`:
   - Returns the MongoDB connection string.

7. `get_*_api_key(self)`:
   - Methods for retrieving API keys for various services (OpenAI, Mistral, HuggingFace, Google, Anthropic).

8. `get_real_agent_root(self)`:
   - Returns the root directory of the Real Agent application.

9. `get_unordered_messages_directory(self)`:
   - Returns the directory path for storing unordered messages.

10. `ensure_directories_exist(self)`:
    - Creates necessary directories for the application.

### Key Features

- Environment variable management using python-dotenv.
- Centralized configuration for various API keys and settings.
- Flexible logging setup with both file and console output.
- Directory management for model caching and message storage.
- Default values for essential configuration items.

### Usage Notes

- The class assumes the presence of a .env file or environment variables for sensitive information like API keys.
- Logging is set up to output to both console and file.
- The REAL_AGENT_ROOT setting determines the base directory for various application-specific folders.
- The class provides a single point of access for all configuration-related information.

### Potential Improvements

1. Implement configuration validation to ensure all required settings are present.
2. Add support for configuration profiles (e.g., development, production).
3. Implement a method to update and save configuration changes.
4. Add support for encrypting sensitive configuration data.
5. Implement a configuration reload method to update settings without restarting the application.

This ConfigManager class serves as the central configuration hub for the entire application, ensuring consistent access to settings and credentials across different components. Its design allows for easy expansion to include new configuration options as the application grows.