# AI Provider Integration Guide

This guide provides an overview of the AI providers integrated into the chat application and instructions for adding new providers.

## Supported AI Providers

### 1. OpenAI

- **File**: `modules/Providers/OpenAI/OA_gen_response.py`
- **Key Features**:
  - Supports GPT models (e.g., GPT-4, GPT-3.5-turbo)
  - Handles both streaming and non-streaming responses
- **Configuration**:
  - Requires `OPENAI_API_KEY` in the `.env` file

### 2. Mistral

- **File**: `modules/Providers/Mistral/Mistral_gen_response.py`
- **Key Features**:
  - Supports Mistral's language models
  - Converts messages to Mistral's specific format
- **Configuration**:
  - Requires `MISTRAL_API_KEY` in the `.env` file

### 3. Google (Gemini)

- **File**: `modules/Providers/Google/GG_gen_response.py`
- **Key Features**:
  - Supports Google's Gemini model
  - Handles prompt conversion for Gemini
- **Configuration**:
  - Requires `GOOGLE_API_KEY` in the `.env` file

### 4. HuggingFace

- **File**: `modules/Providers/HuggingFace/HF_gen_response.py`
- **Key Features**:
  - Supports local model loading and inference
  - Handles model quantization and fine-tuning
- **Configuration**:
  - Requires `HUGGINGFACE_API_KEY` in the `.env` file
  - Uses local model cache for downloaded models

### 5. Anthropic

- **File**: `modules/Providers/Anthropic/Anthropic_gen_response.py`
- **Key Features**:
  - Supports Anthropic's Claude models
  - Handles message conversion for Claude's format
- **Configuration**:
  - Requires `ANTHROPIC_API_KEY` in the `.env` file

## Adding a New AI Provider

To add a new AI provider, follow these steps:

1. Create a new directory under `modules/Providers/` for your provider (e.g., `NewProvider`).

2. Create a new Python file for your provider (e.g., `NP_gen_response.py`).

3. Implement a class for your provider with the following methods:
   - `__init__(self, config_manager: ConfigManager)`: Initialize the provider
   - `generate_response(self, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, stream: bool) -> Union[str, AsyncIterator[str]]`: Generate responses
   - `process_response(self, response) -> str`: Process the response (if needed)

4. Update `modules/Providers/provider_manager.py`:
   - Add your provider to the `provider_config` dictionary
   - Implement any provider-specific logic in the `switch_llm_provider` method

5. Update `modules/Providers/model_manager.py`:
   - Add a new JSON file for your provider's models (e.g., `NP_models.json`)
   - Update the `_load_models` method to include your provider

6. Update `config.py`:
   - Add a method to get the API key for your new provider
   - Update the `__init__` method to load the new API key from the `.env` file

7. Update the `.env` file and `example.env`:
   - Add a new entry for your provider's API key

Here's a template for your provider's response generator:

```python
from typing import List, Dict, Union, AsyncIterator
from config import ConfigManager

class NewProviderGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.api_key = self.config_manager.get_newprovider_api_key()
        # Initialize your provider's client here

    async def generate_response(self, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, stream: bool) -> Union[str, AsyncIterator[str]]:
        # Implement the response generation logic here
        pass

    def process_response(self, response) -> str:
        # Implement response processing if needed
        pass

# Convenience functions
def setup_newprovider_generator(config_manager: ConfigManager):
    return NewProviderGenerator(config_manager)

async def generate_response(config_manager: ConfigManager, messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float, stream: bool):
    generator = setup_newprovider_generator(config_manager)
    return await generator.generate_response(messages, model, max_tokens, temperature, stream)

def process_response(response):
    generator = NewProviderGenerator(ConfigManager())
    return generator.process_response(response)
```

Remember to handle both streaming and non-streaming responses, implement proper error handling, and follow the existing code style and conventions.

## Provider-Specific Features and Limitations

When integrating a new provider, consider the following:

1. **Token Limits**: Different providers and models have varying token limits. Ensure your implementation respects these limits.

2. **Streaming Support**: If the provider supports streaming, implement it for a better user experience.

3. **Rate Limiting**: Implement proper rate limiting and error handling to comply with the provider's API usage guidelines.

4. **Model-Specific Parameters**: Some providers may have unique parameters for their models. Ensure your implementation can handle these.

5. **Response Format**: Providers may return responses in different formats. Ensure your `process_response` method handles this correctly.

By following this guide, you can successfully integrate new AI providers into the chat application, extending its capabilities and giving users more options for AI-powered conversations.