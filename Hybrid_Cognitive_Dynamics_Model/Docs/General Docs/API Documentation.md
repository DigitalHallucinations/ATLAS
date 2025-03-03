# API Documentation

## ConfigManager (`config.py`)

### `class ConfigManager`

#### Methods:

- `get_config(key: str, default: Any = None) -> Any`
  - Retrieves a configuration value by key.
  - Parameters:
    - `key`: The configuration key to retrieve.
    - `default`: The default value if the key is not found.
  - Returns: The configuration value or the default.

- `get_model_cache_dir() -> str`
  - Returns the directory path for caching AI models.

- `get_log_level() -> int`
  - Returns the configured log level as a Python logging constant.

- `setup_logger(name: str) -> logging.Logger`
  - Sets up and returns a logger with the specified name.
  - Parameters:
    - `name`: The name for the logger.
  - Returns: A configured Logger instance.

- `set_log_level(level: str) -> None`
  - Sets a new log level for the application.
  - Parameters:
    - `level`: The new log level as a string (e.g., "INFO", "DEBUG").

Example usage:
```python
config = ConfigManager()
api_key = config.get_config('OPENAI_API_KEY')
logger = config.setup_logger('my_module')
```

## Chatbot (`Chat_Bot.py`)

### `class Chatbot`

#### Methods:

- `async create(cls, config_manager: ConfigManager) -> Chatbot`
  - Class method to create and initialize a Chatbot instance.
  - Parameters:
    - `config_manager`: An instance of ConfigManager.
  - Returns: An initialized Chatbot instance.

- `async process_user_input(self, user_input: str) -> str`
  - Processes user input and generates a response.
  - Parameters:
    - `user_input`: The user's input string.
  - Returns: The generated response string.

- `async set_model(self, model_name: str) -> None`
  - Sets the current AI model.
  - Parameters:
    - `model_name`: The name of the model to set.

- `async get_available_models(self) -> List[str]`
  - Retrieves a list of available AI models.
  - Returns: A list of model names.

Example usage:
```python
config = ConfigManager()
chatbot = await Chatbot.create(config)
response = await chatbot.process_user_input("Hello, how are you?")
```

## ProviderManager (`provider_manager.py`)

### `class ProviderManager`

#### Methods:

- `async create(cls, config_manager: ConfigManager) -> ProviderManager`
  - Class method to create and initialize a ProviderManager instance.
  - Parameters:
    - `config_manager`: An instance of ConfigManager.
  - Returns: An initialized ProviderManager instance.

- `async switch_llm_provider(self, llm_provider: str) -> None`
  - Switches the current LLM provider.
  - Parameters:
    - `llm_provider`: The name of the provider to switch to.

- `async generate_response(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 4000, temperature: float = 0.0, stream: bool = True) -> Union[str, AsyncIterator[str]]`
  - Generates a response using the current LLM provider and model.
  - Parameters:
    - `messages`: A list of message dictionaries.
    - `model`: The model to use (optional).
    - `max_tokens`: Maximum number of tokens in the response.
    - `temperature`: Sampling temperature.
    - `stream`: Whether to stream the response.
  - Returns: A string response or an async iterator for streaming.

Example usage:
```python
config = ConfigManager()
provider_manager = await ProviderManager.create(config)
await provider_manager.switch_llm_provider("OpenAI")
response = await provider_manager.generate_response([{"role": "user", "content": "Hello"}])
```

This API documentation provides an overview of the key classes and methods in the chat application. For more detailed information about each module, refer to the inline documentation and docstrings in the source code.