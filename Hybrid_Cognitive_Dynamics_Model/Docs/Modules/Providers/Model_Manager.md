# Model Manager Documentation

## File: `modules/Providers/model_manager.py`

### Class: `ModelManager`

The `ModelManager` class is responsible for managing AI models across different providers. It handles loading model configurations, setting current models, and providing information about available models.

#### Initialization

```python
def __init__(self, config_manager=None):
    self.config_manager = config_manager
    self.logger = self.config_manager.logger
    self.current_model = None
    self.current_provider = None
    self.models = self._load_models()
```

The constructor initializes the ModelManager with a ConfigManager instance, sets up logging, and loads the available models.

#### Key Methods

1. `_load_models(self) -> Dict[str, List[str]]`:
   - Loads model configurations for different providers.
   - Reads JSON files containing model lists for each provider.
   - Handles HuggingFace models separately, reading from an installed_models.json file.
   - Returns a dictionary with providers as keys and lists of model names as values.

2. `set_model(self, model_name: str, provider: str) -> None`:
   - Sets the current model and provider.
   - Adds the model to the provider's list if it's not already present.

3. `get_current_model(self) -> str`:
   - Returns the name of the currently set model.

4. `get_current_provider(self) -> str`:
   - Returns the name of the currently set provider.

5. `get_available_models(self, provider: str = None) -> Dict[str, List[str]]`:
   - Returns available models for a specific provider or all providers if none specified.

6. `get_token_limits_for_model(self, model_name: str) -> Tuple[int, int]`:
   - Returns the input and output token limits for a given model.
   - Contains predefined limits for various models across different providers.

### Key Features

- Supports multiple AI providers: OpenAI, Mistral, Google, HuggingFace, and Anthropic.
- Dynamically loads model configurations from JSON files.
- Handles special cases for HuggingFace models, reading from a separate installed models file.
- Provides methods to set and retrieve current models and providers.
- Maintains a dictionary of token limits for different models.

### Usage Notes

- The class relies on a proper directory structure and JSON files for each provider.
- Error handling is implemented for file reading and JSON parsing.
- The class uses logging to record information and errors during operation.
- Token limits are hardcoded and may need updates as models change or new models are added.

### Potential Improvements

1. Implement a method to update model configurations dynamically.
2. Add functionality to save changes to model lists back to JSON files.
3. Implement a more flexible way to update token limits for models.
4. Consider adding support for custom model configurations.

This class serves as a central point for managing AI models across the application, providing a unified interface for working with multiple AI providers and their respective models.